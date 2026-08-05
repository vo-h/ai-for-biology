#!/usr/bin/env python3
"""
Benchmark RxRx1Dataset throughput by looping through one epoch of the train split.

Reports images/s, batches/s, and per-batch timing. Run with --num-workers 0
first to isolate single-process fetch speed, then bump up to see
multiprocessing gains. Every item fetch is `len(channels)` separate GCS GET
requests (one PNG per channel, no batched read like the Census pipeline
uses) — expect this to be I/O-bound.

Example:
    # Quick smoke test — a few hundred images, no channel-stats fetch
    python scripts/rxrx1/test_dataloader.py --limit 200 --skip-normalize

    # Full train split for one cell line, real throughput numbers
    python scripts/rxrx1/test_dataloader.py --cell-types HEPG2 --num-workers 4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.rxrx1 import (
    compute_channel_stats,
    fetch_metadata,
    RxRx1CollateFn,
    RxRx1Dataset,
    worker_init_fn,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-types", nargs="+", default=None,
                   help="Restrict to these cell lines (e.g. HEPG2 HUVEC RPE U2OS). "
                        "Default: all.")
    p.add_argument("--sites", type=int, nargs="+", default=[1, 2])
    p.add_argument("--channels", type=int, nargs="+", default=None,
                   help="Default: all 6.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Only use the first N metadata rows (wells) instead of the "
                        "full split — for a quick smoke test.")
    p.add_argument("--skip-normalize", action="store_true",
                   help="Skip fetching pixel_stats.csv / channel standardization "
                        "(collate falls back to a plain /255 scale) — faster startup "
                        "for a quick smoke test.")
    return p.parse_args()


def main():
    args = parse_args()
    channels = tuple(args.channels) if args.channels else None

    print(f"Fetching metadata (cell_types={args.cell_types or 'all'})...")
    meta = fetch_metadata(split="train", cell_types=args.cell_types)
    if args.limit is not None:
        meta = meta.head(args.limit)
    print(f"{len(meta):,} wells | {meta['cell_type'].nunique()} cell type(s) | "
          f"{meta['experiment'].nunique()} experiment(s)")

    channel_stats = None
    if not args.skip_normalize:
        print("Computing channel stats from pixel_stats.csv (no image fetch)...")
        channel_stats = compute_channel_stats(cell_types=args.cell_types)

    ds_kwargs = {"sites": tuple(args.sites)}
    if channels is not None:
        ds_kwargs["channels"] = channels
    ds = RxRx1Dataset(meta, split="train", **ds_kwargs)
    print(f"Dataset: {len(ds):,} items ({len(meta):,} wells x {len(args.sites)} site(s))")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=RxRx1CollateFn(channel_stats),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    n_images, n_batches = 0, 0
    t0 = time.time()
    pbar = tqdm(loader, desc="train", unit="batch")
    for X, _ in pbar:
        n_images += len(X)
        n_batches += 1
        elapsed = time.time() - t0
        pbar.set_postfix(
            images=f"{n_images:,}",
            img_per_s=f"{n_images / elapsed:.1f}",
        )
    elapsed = time.time() - t0

    print(f"\n{n_images:,} images | {n_batches:,} batches | {elapsed:.1f}s total")
    if n_batches:
        print(f"{n_images / elapsed:.2f} images/s | {n_batches / elapsed:.3f} batches/s | "
              f"{elapsed / n_batches:.3f} s/batch  (num_workers={args.num_workers})")


if __name__ == "__main__":
    main()
