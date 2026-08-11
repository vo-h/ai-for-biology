#!/usr/bin/env python3
"""
DDP training entry point for RxRx1ResNet18.

Must be launched via torchrun, not `python` directly -- torchrun sets the
RANK/LOCAL_RANK/WORLD_SIZE env vars setup_ddp() (src/training/rxrx1.py)
reads.

Example:
    # Single machine, 4 GPUs
    torchrun --standalone --nproc_per_node=4 scripts/rxrx1/train_resnet.py \
        --cell-types HEPG2 --n-epochs 10 --batch-size 32

    # Local smoke test of the DDP wiring itself -- CPU, 2 processes, a
    # handful of images, one epoch
    torchrun --standalone --nproc_per_node=2 scripts/rxrx1/train_resnet.py \
        --cell-types HEPG2 --n-epochs 1 --batch-size 4 --limit 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.rxrx1 import run_training, TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-types", nargs="+", default=None,
                   help="Restrict to these cell lines (e.g. HEPG2 HUVEC RPE U2OS). "
                        "Default: all.")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Per-GPU/per-process batch size -- effective global batch "
                        "size is this x world_size.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-experiments", type=int, default=2,
                   help="Number of whole experiments held out for validation -- not "
                        "a random well/image split (see split_by_experiment).")
    p.add_argument("--sites", type=int, nargs="+", default=[1, 2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results/rxrx1"))
    p.add_argument("--limit", type=int, default=None,
                   help="Only use the first N metadata rows -- for a quick smoke "
                        "test of the DDP wiring instead of a real training run.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        cell_types=args.cell_types,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_experiments=args.val_experiments,
        sites=tuple(args.sites),
        seed=args.seed,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
