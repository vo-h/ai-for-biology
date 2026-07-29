#!/usr/bin/env python3
"""
Benchmark ExperimentDataset throughput.

Loops through one epoch of the train split and reports cells/s, batches/s,
and per-chunk fetch timing. Run with --num-workers 0 first to isolate
single-process fetch speed, then bump up to see multiprocessing gains.

Usage:
    python scripts/test.py
    python scripts/test.py --hvg-cache results/mlp/hvg_list.json --num-workers 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cellxgene_census
import torch
from tqdm import tqdm
from tiledbsoma_ml import experiment_dataloader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.census import build_census_dataset, CensusCollateFn, fetch_metadata, compute_hvg_list, SOMA_CTX, CENSUS_VERSION
from src.data.preprocessing import get_label_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hvg-cache", type=Path, default=None,
                   help="Path to cached HVG JSON; computed from scratch if not provided")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-sample-hvg", type=int, default=20_000)
    p.add_argument("--tissues", nargs="+", default=["blood"])
    p.add_argument("--n-donors", type=int, default=10)
    p.add_argument("--io-batch-size", type=int, default=65_536)
    p.add_argument("--shuffle-chunk-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--census-version", default=CENSUS_VERSION)
    return p.parse_args()


def main():
    args = parse_args()

    if args.hvg_cache and args.hvg_cache.exists():
        hvg_genes = json.loads(args.hvg_cache.read_text())
        print(f"Loaded {len(hvg_genes)} HVGs from {args.hvg_cache}")
    else:
        print("Computing HVG list from scratch...")
        hvg_genes = compute_hvg_list(
            tissues=args.tissues,
            n_sample_cells=args.n_sample_hvg,
            n_hvg=args.n_hvg,
            census_version=args.census_version,
        )
        if args.hvg_cache:
            args.hvg_cache.parent.mkdir(parents=True, exist_ok=True)
            args.hvg_cache.write_text(json.dumps(hvg_genes))
            print(f"Saved {len(hvg_genes)} HVGs to {args.hvg_cache}")
        print(f"Got {len(hvg_genes)} HVGs")

    print("Fetching metadata...")
    meta = fetch_metadata(tissues=args.tissues, census_version=args.census_version)
    top_donors = meta["donor_id"].value_counts().head(args.n_donors).index
    meta = meta[meta["donor_id"].isin(top_donors)].copy()
    type_counts = meta["cell_type"].value_counts()
    meta = meta[meta["cell_type"].isin(type_counts[type_counts >= 20].index)].copy()
    print(f"{len(meta):,} cells | {meta['donor_id'].nunique()} donors | {meta['cell_type'].nunique()} classes")

    label2int, _ = get_label_encoder(meta["cell_type"].tolist())

    print("Building dataset...")
    t0 = time.time()
    collate = CensusCollateFn(label2int)

    with cellxgene_census.open_soma(census_version=args.census_version, context=SOMA_CTX) as census:
        ds = build_census_dataset(
            census,
            soma_joinids=meta["soma_joinid"].tolist(),
            hvg_genes=hvg_genes,
            batch_size=args.batch_size,
            io_batch_size=args.io_batch_size,
            shuffle_chunk_size=args.shuffle_chunk_size,
            shuffle=False,
        )
        print(f"  dataset ready in {time.time()-t0:.1f}s")

        loader = experiment_dataloader(ds, num_workers=args.num_workers, collate_fn=collate)

        print(f"\nStreaming (num_workers={args.num_workers}, io_batch_size={args.io_batch_size:,}, batch_size={args.batch_size})...")
        t_start = time.time()
        n_batches = n_cells = 0
        t_first_batch = None

        for X, y in tqdm(loader, unit="batch"):
            if t_first_batch is None:
                t_first_batch = time.time() - t_start
            n_batches += 1
            n_cells += len(X)

        elapsed = time.time() - t_start
        print(f"\n--- Results ---")
        print(f"First batch latency : {t_first_batch:.1f}s")
        print(f"Total cells         : {n_cells:,}")
        print(f"Total batches       : {n_batches:,}")
        print(f"Total time          : {elapsed:.1f}s")
        print(f"Throughput          : {n_cells / elapsed:,.0f} cells/s  |  {n_batches / elapsed:.1f} batches/s")
        print(f"Tensor shape        : X={tuple(X.shape)}  y={tuple(y.shape)}")


if __name__ == "__main__":
    main()
