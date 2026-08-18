"""
RxRx1 — Recursion Cellular Image Classification (`rxrx1-us-central1` GCS bucket).

All access goes directly against the public GCS bucket via `gcsfs` — nothing
is downloaded to local disk. 51 experiments across 4 cell lines (HUVEC, RPE,
HEPG2, U2OS), 4 plates per experiment, 2 imaging sites per well, 6
fluorescence channels per site (Hoechst/ConA/Phalloidin/Syto14/MitoTracker/
WGA), each a 512x512 grayscale PNG. Task: classify which of 1,108 siRNA
reagents was applied to a well, from its 6-channel image.

The bucket is public and requires no credentials — `gcsfs` is used with
`token="anon"` throughout.

Reference: https://www.kaggle.com/competitions/recursion-cellular-image-classification
           https://rxrx.ai/rxrx1
"""

from __future__ import annotations
from typing import Literal

import io
import os
import warnings

# grpc is an unused transitive dependency here (GCS reads go over plain HTTP
# via aiohttp, not grpc) that still registers a fork handler on import. That
# handler logs an "Other threads are currently calling into gRPC" / "FD from
# fork parent" line, at INFO level, every time a DataLoader worker forks --
# harmless, but noisy across a real training loop. Must be set before gcsfs
# (and grpc, transitively) is imported below; only silences grpc's own
# logging, doesn't touch its actual fork-safety behavior.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# google-api-core warns on every one of its submodules imported under Python
# 3.10 (google.api_core, google.cloud._storage_v2, google.cloud.storage_
# control_v2 -- gcsfs pulls all three in transitively), so this fires
# several times per process, once per spawned DataLoader worker on top of
# that. Filtered by the module that actually calls warnings.warn(), not a
# blanket FutureWarning ignore, so unrelated FutureWarnings elsewhere still
# surface normally.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"google\.api_core\._python_version_support"
)

import gcsfs
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

RXRX1_BUCKET = "rxrx1-us-central1"

N_SITES = 2
N_CHANNELS = 6
IMG_SIZE = 512
N_SIRNA_CLASSES = 1108  # dense 0..1107, same panel applied across all 4 cell lines
CHANNEL_NAMES = {
    1: "Hoechst (nucleus)",
    2: "ConA (endoplasmic reticulum)",
    3: "Phalloidin (actin cytoskeleton)",
    4: "Syto14 (nucleolus)",
    5: "MitoTracker (mitochondria)",
    6: "WGA (golgi apparatus)",
}


# ---------------------------------------------------------------------------
# GCS filesystem
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int) -> None:
    """
    Pass as DataLoader(..., worker_init_fn=worker_init_fn) whenever num_workers > 0.

    RxRx1Dataset.__init__ already builds a `self.fs` for direct/num_workers=0
    use, but that instance was built in the main process. Re-calling
    gcsfs.GCSFileSystem(token="anon") here, in each freshly forked worker,
    does correctly get a new instance rather than fsspec's usual same-args
    instance cache handing back the parent's (potentially stale) one —
    verified directly with os.fork(): fsspec.asyn registers
    os.register_at_fork(after_in_child=reset_after_fork), and each instance
    checks self._pid against os.getpid(), so the cache is fork-aware and
    resets itself in the child instead of serving up a connection that was
    live in a different process. This call is what triggers that rebuild.

    Doesn't silence the "gRPC"/"FD from fork parent" warnings printed at the
    moment DataLoader forks each worker — those fire from background threads
    that exist in the parent process the instant fork() happens (grpc is a
    transitive dependency pulled in for GCS auth), before this function ever
    runs. Cosmetic only; this function is about correctness, not that noise.
    """
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.fs = gcsfs.GCSFileSystem(token="anon")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def fetch_metadata(
    split: Literal["train", "test"] = "train",
    cell_types: list[str] | None = None,
    bucket: str = RXRX1_BUCKET,
) -> pd.DataFrame:
    """
    Return the id_code/experiment/plate/well(/sirna) metadata table for a split.

    `split` is "train" or "test" — "test" has no `sirna` column (that's the
    label withheld for the Kaggle leaderboard). Adds a `cell_type` column
    parsed from `experiment` (e.g. "HEPG2-01" -> "HEPG2"); pass `cell_types`
    to restrict to a subset (["HUVEC"], etc.) up front rather than filtering
    the full table yourself downstream.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    meta = pd.read_csv(
        f"gs://{bucket}/metadata/{split}.csv",
        storage_options={"token": "anon"},
    )
    meta["cell_type"] = meta["experiment"].str.split("-").str[0]

    if cell_types is not None:
        meta = meta[meta["cell_type"].isin(cell_types)].reset_index(drop=True)

    return meta


def compute_channel_stats(
    cell_types: list[str] | None = None,
    bucket: str = RXRX1_BUCKET,
) -> dict[int, tuple[float, float]]:
    """
    Return {channel: (mean, std)} for standardization, from the bucket's
    precomputed `pixel_stats.csv` — no image data is fetched.

    `pixel_stats.csv` has one row per (image, channel) with that image's
    pixel mean/std (all images are the same 512x512 size, so every row is an
    equally-weighted group). Per-channel mean is the mean of per-image means;
    per-channel variance uses the law of total variance
    (`mean(var_i) + var(mean_i)`) rather than naively averaging `std`, which
    would ignore between-image variance entirely and understate the true
    spread.
    """
    stats = pd.read_csv(
        f"gs://{bucket}/metadata/pixel_stats.csv",
        storage_options={"token": "anon"},
    )
    if cell_types is not None:
        stats = stats[stats["experiment"].str.split("-").str[0].isin(cell_types)]

    out = {}
    for channel, grp in stats.groupby("channel"):
        channel_mean = grp["mean"].mean()
        channel_var = (grp["std"] ** 2).mean() + grp["mean"].var(ddof=0)
        out[int(channel)] = (float(channel_mean), float(channel_var ** 0.5))
    return out


def build_image_path(
    experiment: str,
    plate: int,
    well: str,
    site: int,
    channel: int,
    split: str = "train",
    bucket: str = RXRX1_BUCKET,
) -> str:
    """
    Return the bucket-relative path to one channel of one site's image, e.g.
    "rxrx1-us-central1/images/train/HEPG2-01/Plate1/B03_s1_w1.png".

    No "gs://" prefix — this is what `gcsfs.GCSFileSystem.open`/`.info`
    expect; `pd.read_csv` (above) takes the "gs://" form instead, since that
    goes through fsspec's URL-based dispatch rather than a filesystem object.
    """
    return f"{bucket}/images/{split}/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png"


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class RxRx1Dataset(Dataset):
    """
    Streams 6-channel fluorescence microscopy images directly from GCS —
    nothing is downloaded to disk. Each item is one (well, site): the
    requested channels' PNGs for that site are fetched and decoded on the
    fly and stacked into a (n_channels, 512, 512) uint8 array.

    Map-style (not IterableDataset, unlike the Census pipeline in
    cellxgene.py) — GCS serves random single-object reads fine at this
    dataset size, so plain index-based access works, and it's what
    DistributedSampler expects for multi-GPU training.

    Each metadata row expands to `len(sites)` items, since a well is imaged
    at 2 non-overlapping sites and both are normally used as independent
    training examples.

    Pass `worker_init_fn=worker_init_fn` (this module's) to DataLoader
    whenever num_workers > 0 — see its docstring for why.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        split: str = "train",
        sites: tuple[int, ...] = (1, 2),
        channels: tuple[int, ...] = tuple(range(1, N_CHANNELS + 1)),
        bucket: str = RXRX1_BUCKET,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.split = split
        self.sites = sites
        self.channels = channels
        self.bucket = bucket
        self.has_labels = "sirna" in self.metadata.columns
        # Flat (row position, site) index, built once, so __getitem__ doesn't
        # redo the row x site expansion on every call.
        self.index = [(i, s) for i in range(len(self.metadata)) for s in sites]
        # Default for direct access / num_workers=0. With num_workers > 0,
        # worker_init_fn (above) replaces this per worker at process startup.
        self.fs = gcsfs.GCSFileSystem(token="anon")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        row_pos, site = self.index[idx]
        row = self.metadata.iloc[row_pos]

        # One (well, site) is n_channels separate objects in GCS -- fetching
        # them one at a time in a loop serializes n_channels network round
        # trips (measured ~2.7s for 6 channels). fs.cat() fetches the whole
        # list concurrently over gcsfs's async backend instead (~0.35s for
        # the same 6, ~7.7x) -- the fix here is concurrency, not a faster
        # client library.
        paths = [
            build_image_path(
                row["experiment"], int(row["plate"]), row["well"], site, channel,
                split=self.split, bucket=self.bucket,
            )
            for channel in self.channels
        ]
        data = self.fs.cat(paths)
        planes = [np.array(Image.open(io.BytesIO(data[p])), dtype=np.uint8) for p in paths]
        X = np.stack(planes, axis=0)  # (n_channels, 512, 512)

        label = int(row["sirna"]) if self.has_labels else -1
        return X, label


class RxRx1CollateFn:
    """
    Picklable collate_fn for RxRx1Dataset.

    Converts a batch of uint8 (n_channels, 512, 512) arrays to a single
    float32 (B, n_channels, 512, 512) tensor. If `channel_stats` is given
    (see compute_channel_stats), standardizes each channel to zero
    mean/unit variance — RxRx1 has strong batch effects between experiments
    (different confocal runs, reagent lots), so per-channel standardization
    is the minimum treatment before training, playing the same role
    log1p-normalization does for scRNA-seq counts in cellxgene.py. Without
    channel_stats, falls back to a plain /255 scale to [0, 1].

    Assumes `channel_stats` keys match the dataset's `channels` in the same
    order (default: both are 1..6) — a `channels` subset must be paired with
    channel_stats computed/filtered the same way.
    """

    def __init__(self, channel_stats: dict[int, tuple[float, float]] | None = None):
        self.channel_stats = channel_stats

    def __call__(self, batch: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(np.stack([x for x, _ in batch]).astype(np.float32))
        y = torch.tensor([label for _, label in batch], dtype=torch.long)

        if self.channel_stats is not None:
            channels = sorted(self.channel_stats)
            mean = torch.tensor([self.channel_stats[c][0] for c in channels]).view(1, -1, 1, 1)
            std = torch.tensor([self.channel_stats[c][1] for c in channels]).view(1, -1, 1, 1)
            X = (X - mean) / std
        else:
            X = X / 255.0

        return X, y
