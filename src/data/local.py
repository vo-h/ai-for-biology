"""
PyTorch dataset for training directly against local per-donor h5ad files
(see download_donor_h5ads / scripts/download_dataset.py).

Donors are the unit of both storage and memory: only one donor's h5ad is
opened and materialized at a time, shuffled and split into mini-batches, then
discarded before the next donor is opened. The full directory is never loaded
into memory at once.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


def list_available_donors(data_dir: Path) -> list[str]:
    """Return donor_ids for every "{donor_id}.h5ad" file in data_dir."""
    return sorted(p.stem for p in Path(data_dir).glob("*.h5ad"))


class LocalDonorDataset(IterableDataset):
    """
    Streams (X_batch, obs_batch) mini-batches from local per-donor h5ad files.

    One donor is opened and materialized into memory at a time — its cells
    are shuffled in-memory (if `shuffle`) and split into `batch_size`
    mini-batches before the next donor is opened. Never holds more than one
    donor's data at once, and even that donor's X is only densified per
    mini-batch (not all at once), since Census raw counts read back as sparse.

    Yields raw (X_ndarray, obs_df) pairs — same shape/interface as
    tiledbsoma_ml's ExperimentDataset — so CensusCollateFn works unchanged as
    the collate_fn. Wrap in a plain torch.utils.data.DataLoader (not
    tiledbsoma_ml's experiment_dataloader, which is Census-specific), with
    batch_size=None since batching is handled here.

    By default every cell in each listed donor's file is used. Pass
    `cell_indices` (donor_id -> row positions within that donor's file) to
    restrict to a specific subset of cells per donor instead — needed for
    splits that cut across donor boundaries (e.g. a random cell-level split,
    as opposed to the whole-donor splits cross-donor CV uses).
    """

    def __init__(
        self,
        data_dir: Path,
        donor_ids: list[str] | None = None,
        cell_indices: dict[str, np.ndarray] | None = None,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.donor_ids = donor_ids if donor_ids is not None else list_available_donors(self.data_dir)
        self.cell_indices = cell_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reseed donor visit order and in-donor shuffling for a new epoch."""
        self.epoch = epoch

    def __iter__(self):
        donor_ids = list(self.donor_ids)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(donor_ids)

        # Partition donors across DataLoader workers so each worker handles a
        # disjoint subset — otherwise every worker would redundantly stream
        # every donor.
        worker_info = get_worker_info()
        if worker_info is not None:
            donor_ids = donor_ids[worker_info.id::worker_info.num_workers]

        for donor_id in donor_ids:
            path = self.data_dir / f"{donor_id}.h5ad"
            adata = ad.read_h5ad(path, backed="r")

            base_idx = (
                np.asarray(self.cell_indices[donor_id])
                if self.cell_indices is not None
                else np.arange(adata.n_obs)
            )
            n = len(base_idx)
            if n < 2:
                # A single-cell batch crashes BatchNorm1d in training mode
                # ("Expected more than 1 value per channel"); a 1-cell donor
                # can't form a valid batch on its own, so skip it entirely.
                adata.file.close()
                del adata
                continue

            order = base_idx[rng.permutation(n)] if self.shuffle else base_idx
            X, obs = adata.X, adata.obs

            starts = list(range(0, n, self.batch_size))
            # Batches never span donor boundaries, so a donor whose cell count
            # leaves a remainder of exactly 1 would otherwise yield a trailing
            # singleton batch — same BatchNorm crash as above. Merge it into
            # the previous batch instead of dropping data.
            if len(starts) > 1 and n - starts[-1] == 1:
                starts.pop()

            for i, start in enumerate(starts):
                end = starts[i + 1] if i + 1 < len(starts) else n
                idx = order[start:end]
                X_batch = X[idx]
                if hasattr(X_batch, "toarray"):
                    X_batch = X_batch.toarray()
                yield X_batch, obs.iloc[idx]

            adata.file.close()
            del adata, X
