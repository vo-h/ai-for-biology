#!/usr/bin/env python3
"""
Train the MLP cell-type classifier with out-of-core, cross-donor evaluation.

Data never fully materialises in memory — streamed from Census S3 via
tiledbsoma_ml's ExperimentDataset (shuffle-chunked, IO-batched, then
mini-batched; see docs/cellxgene.md). DataLoader workers prefetch the next
IO batch while the GPU trains on the current one.

Example:
    python scripts/cellxgene/train_mlp.py \
        --tissues blood \
        --n-donors 30 \
        --n-epochs 10 \
        --io-batch-size 65536 \
        --num-workers 2 \
        --output-dir results/mlp/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.cellxgene import compute_hvg_list
from src.evaluation.cellxgene import group_k_fold_donors, random_k_fold
from src.training.cellxgene import TrainConfig, run_training

SPLIT_STRATEGIES = {"donor": group_k_fold_donors, "random": random_k_fold}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissues", nargs="+", default=["blood"])
    p.add_argument("--n-donors", type=int, default=30)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--io-batch-size", type=int, default=65_536)
    p.add_argument("--shuffle-chunk-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-sample-hvg", type=int, default=20_000,
                   help="Cells to sample for HVG computation")
    p.add_argument("--output-dir", type=Path, default=Path("results/mlp"))
    p.add_argument("--census-version", default="2025-11-08")
    p.add_argument("--hvg-cache", type=Path, default=None,
                   help="Path to a saved HVG list JSON (skips recomputation)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Train from local per-donor h5ad files in this directory "
                        "(see scripts/cellxgene/download_dataset.py) instead of streaming "
                        "live from Census. Skips HVG computation — the gene set "
                        "is whatever was downloaded.")
    p.add_argument("--split-strategy", choices=SPLIT_STRATEGIES, default="donor",
                   help="'donor' (default): cross-donor CV, no donor in both "
                        "train and test. 'random': random cell-level split, "
                        "ignoring donor identity — the accuracy-inflation "
                        "baseline (see src.evaluation.cellxgene).")
    p.add_argument("--patience", type=int, default=None,
                   help="Stop a fold's training early after this many epochs "
                        "without val macro-F1 improvement. Default: off, always "
                        "runs the full --n-epochs.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        hvg_genes = []
        print(f"Training from local files in {args.data_dir} — "
              "skipping HVG computation (gene set is fixed by the downloaded files).")
    else:
        # HVG list — compute once and cache so reruns are fast
        hvg_cache = args.hvg_cache or args.output_dir / "hvg_list.json"
        if hvg_cache.exists():
            hvg_genes = json.loads(hvg_cache.read_text())
            print(f"Loaded {len(hvg_genes)} HVGs from {hvg_cache}")
        else:
            print(f"Computing HVG list from {args.n_sample_hvg:,} cells...")
            hvg_genes = compute_hvg_list(
                tissues=args.tissues,
                n_sample_cells=args.n_sample_hvg,
                n_hvg=args.n_hvg,
                census_version=args.census_version,
            )
            hvg_cache.write_text(json.dumps(hvg_genes))
            print(f"Saved {len(hvg_genes)} HVGs to {hvg_cache}")

    cfg = TrainConfig(
        tissues=args.tissues,
        n_donors=args.n_donors,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        io_batch_size=args.io_batch_size,
        shuffle_chunk_size=args.shuffle_chunk_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        census_version=args.census_version,
        data_dir=args.data_dir,
        patience=args.patience,
    )

    results = run_training(hvg_genes=hvg_genes, cfg=cfg, split_fn=SPLIT_STRATEGIES[args.split_strategy])

    summary_path = args.output_dir / "training_results.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
