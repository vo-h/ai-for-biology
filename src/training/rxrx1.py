"""
DDP training loop for RxRx1ResNet18.

Launch with torchrun (not `python` directly) — torchrun sets the
RANK/LOCAL_RANK/WORLD_SIZE env vars setup_ddp() reads:

    torchrun --standalone --nproc_per_node=4 scripts/rxrx1/train_resnet.py --cell-types HEPG2

`--batch-size` is per-GPU/per-process — effective global batch size is
`batch_size * world_size`. Gradient sync across ranks happens automatically
inside DDP's backward() hook; nothing in this file does it by hand.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data.rxrx1 import (
    compute_channel_stats,
    fetch_metadata,
    N_SIRNA_CLASSES,
    RxRx1CollateFn,
    RxRx1Dataset,
    worker_init_fn,
)
from src.models.resnet import RxRx1ResNet18


@dataclass
class TrainConfig:
    cell_types: list[str] | None = None
    n_epochs: int = 10
    batch_size: int = 32
    """Per-GPU/per-process batch size."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_experiments: int = 2
    """Number of whole experiments held out for validation."""
    sites: tuple[int, ...] = (1, 2)
    seed: int = 42
    output_dir: Path = Path("results/rxrx1")
    limit: int | None = None
    """If set, only use the first `limit` metadata rows — for a quick smoke
    test of the DDP wiring rather than a real training run."""


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    """
    Initialize the process group from torchrun's env vars.

    Returns (rank, local_rank, world_size, device). NCCL backend for GPU
    training (the only backend that supports GPU-to-GPU collectives);
    falls back to gloo on CPU, e.g. for a local `--nproc_per_node=2` smoke
    test with no GPU available.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: float, device: torch.device) -> float:
    """All-reduce a scalar (e.g. a rank-local mean loss) to its mean across ranks."""
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_by_experiment(
    meta: pd.DataFrame,
    n_val_experiments: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out whole experiments (batches) for validation, not a random
    well/image split.

    RxRx1 has strong batch effects between experiments — different
    confocal runs, reagent lots (see docs/rxrx1.md) — so a random split
    would let the model partly learn "which experiment is this" as a
    shortcut, the same accuracy-inflation risk a random donor split poses
    for the Census cell-type classifier (see projects/split-strategies.md).
    """
    experiments = sorted(meta["experiment"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(experiments)
    val_experiments = set(experiments[:n_val_experiments])

    val_mask = meta["experiment"].isin(val_experiments)
    train_meta = meta[~val_mask].reset_index(drop=True)
    val_meta = meta[val_mask].reset_index(drop=True)
    return train_meta, val_meta


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss, n_batches, n_images = 0.0, 0, 0
    t0 = time.time()

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch", disable=not is_main_process())
    for X, y in pbar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        n_images += len(X)
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    elapsed = time.time() - t0
    return {
        "loss": reduce_mean(total_loss / max(n_batches, 1), device),
        "n_images": n_images,
        "n_batches": n_batches,
        "elapsed": elapsed,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss, n_correct, n_images, n_batches = 0.0, 0, 0, 0

    for X, y in tqdm(loader, desc="   eval", leave=False, unit="batch", disable=not is_main_process()):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total_loss += criterion(logits, y).item()
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        n_images += len(X)
        n_batches += 1

    # Each rank only sees its DistributedSampler shard of the val set, so a
    # per-rank accuracy is a biased estimate of the true global accuracy --
    # sum the raw counts across ranks before dividing, not the per-rank
    # ratios (accuracy is exactly reducible this way; macro-F1 would not
    # be, since it isn't a simple mean of per-shard values).
    correct_t = torch.tensor(float(n_correct), device=device)
    images_t = torch.tensor(float(n_images), device=device)
    dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(images_t, op=dist.ReduceOp.SUM)

    return {
        "loss": reduce_mean(total_loss / max(n_batches, 1), device),
        "accuracy": (correct_t / images_t).item(),
        "n_images": int(images_t.item()),
    }


def run_training(cfg: TrainConfig) -> None:
    rank, local_rank, world_size, device = setup_ddp()
    main = is_main_process()

    if main:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"World size: {world_size} | device: {device}")
        print(f"Fetching metadata (cell_types={cfg.cell_types or 'all'})...")

    meta = fetch_metadata(split="train", cell_types=cfg.cell_types)
    if cfg.limit is not None:
        meta = meta.head(cfg.limit)
    train_meta, val_meta = split_by_experiment(meta, cfg.val_experiments, cfg.seed)

    if main:
        print(f"{len(train_meta):,} train wells | {len(val_meta):,} val wells "
              f"({val_meta['experiment'].nunique()} held-out experiment(s)) | "
              f"{N_SIRNA_CLASSES} classes")
        print("Computing channel stats (pixel_stats.csv, no image fetch)...")
    channel_stats = compute_channel_stats(cell_types=cfg.cell_types)

    train_ds = RxRx1Dataset(train_meta, split="train", sites=cfg.sites)
    val_ds = RxRx1Dataset(val_meta, split="train", sites=cfg.sites)  # still the "train" bucket path -- has labels

    # DistributedSampler, not a manual rank-based slice of the dataset: it
    # guarantees every rank gets the same number of samples/batches per
    # epoch (padding the shard if the dataset size isn't evenly divisible
    # by world_size). DDP requires every rank to make the same number of
    # backward() calls each epoch -- a rank that runs out of batches early
    # would leave the others hanging on an all-reduce that never comes.
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False,
    )

    collate = RxRx1CollateFn(channel_stats)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)

    model = RxRx1ResNet18(n_classes=N_SIRNA_CLASSES).to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    if main:
        print(f"Model params: {model.n_params:,}")

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    criterion = nn.CrossEntropyLoss()

    if main:
        (cfg.output_dir / "model_config.json").write_text(json.dumps({
            "n_channels": model.net.conv1.in_channels,
            "n_classes": N_SIRNA_CLASSES,
            "cell_types": cfg.cell_types,
            "channel_stats": channel_stats,
        }, indent=2))

    best_acc = 0.0
    for epoch in range(cfg.n_epochs):
        # Reshuffles differently each epoch while keeping every rank's
        # shuffle in sync with the others -- without this, every rank would
        # replay the exact same order every epoch (DistributedSampler seeds
        # deterministically from `seed`, not from wall-clock state).
        train_sampler.set_epoch(epoch)

        train_stats = train_epoch(ddp_model, train_loader, optimizer, criterion, device)
        val_stats = eval_epoch(ddp_model, val_loader, criterion, device)
        scheduler.step()

        if main:
            print(f"epoch {epoch}: train_loss={train_stats['loss']:.4f}  "
                  f"val_loss={val_stats['loss']:.4f}  val_acc={val_stats['accuracy']:.4f}  "
                  f"({train_stats['elapsed']:.1f}s, {train_stats['n_images']:,} images/rank)")
            if val_stats["accuracy"] > best_acc:
                best_acc = val_stats["accuracy"]
                torch.save(model.state_dict(), cfg.output_dir / "best_model.pt")

    if main:
        print(f"\nBest val accuracy: {best_acc:.4f}")

    cleanup_ddp()
