#!/usr/bin/env python3
"""
Evaluate a saved CellTypeMLP checkpoint against a local per-donor h5ad directory.

Reads model_config.json (n_genes, n_classes, architecture, label mapping,
gene list) from the checkpoint's directory — written automatically by
run_training — so architecture/label alignment never has to be re-specified
by hand. Validates the test data's genes match what the model was trained on
before running anything: CellTypeMLP's input is purely positional (a Linear
layer), so a silent gene mismatch would produce plausible-looking garbage
rather than an error.

Example:
    python scripts/test_mlp.py \
        --model-path results/mlp/best_model_fold0.pt \
        --data-dir results/donors-test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.census import CensusCollateFn
from src.data.local import LocalDonorDataset, list_available_donors
from src.evaluation.metrics import compute_metrics
from src.models.mlp import CellTypeMLP
from src.training.trainer import eval_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True,
                   help="Path to a saved checkpoint, e.g. results/mlp/best_model_fold0.pt")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory of per-donor h5ad files to evaluate on "
                        "(see scripts/download_dataset.py)")
    p.add_argument("--model-config", type=Path, default=None,
                   help="Path to model_config.json (default: alongside --model-path)")
    p.add_argument("--donors", nargs="+", default=None,
                   help="Restrict evaluation to these donor_ids (e.g. a fold's "
                        "held-out test_donors from training_results.json). "
                        "Default: every donor found in --data-dir.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    config_path = args.model_config or args.model_path.parent / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found — expected next to the checkpoint, written "
            "automatically by run_training. Pass --model-config to point elsewhere."
        )
    model_config = json.loads(config_path.read_text())
    int2label = {int(k): v for k, v in model_config["int2label"].items()}
    label2int = {v: k for k, v in int2label.items()}
    gene_list = model_config["gene_list"]

    donor_ids = list_available_donors(args.data_dir)
    if not donor_ids:
        raise FileNotFoundError(f"No *.h5ad files found in {args.data_dir}")

    if args.donors is not None:
        missing = set(args.donors) - set(donor_ids)
        if missing:
            raise FileNotFoundError(f"Requested donors not found in {args.data_dir}: {sorted(missing)}")
        donor_ids = list(args.donors)
    print(f"Evaluating on {len(donor_ids)} donor(s): {donor_ids}")

    # Gene identity check, not just a count check — same length with
    # different genes (or a different order) would still silently corrupt
    # every prediction.
    test_genes = ad.read_h5ad(args.data_dir / f"{donor_ids[0]}.h5ad", backed="r").var["feature_name"].tolist()
    if test_genes != gene_list:
        raise ValueError(
            f"Gene mismatch: {args.data_dir} was not downloaded with the same "
            "HVG list the model was trained on. Re-download it with the same "
            "--hvg-cache used for training."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = CellTypeMLP(
        n_genes=model_config["n_genes"],
        n_classes=model_config["n_classes"],
        hidden_dims=tuple(model_config["hidden_dims"]),
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded {args.model_path} ({model.n_params:,} params)")

    ds = LocalDonorDataset(args.data_dir, donor_ids=donor_ids, batch_size=args.batch_size, shuffle=False)
    collate = CensusCollateFn(label2int)
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers, collate_fn=collate)

    criterion = nn.CrossEntropyLoss()
    stats = eval_epoch(model, loader, criterion, device)
    metrics = compute_metrics(stats["y_true"], stats["y_pred"], int2label)

    print(f"\n{stats['n_cells']:,} cells evaluated "
          "(cells with a cell_type unseen during training are dropped by CensusCollateFn)")
    print(f"macro-F1: {metrics['macro_f1']:.4f}  "
          f"weighted-F1: {metrics['weighted_f1']:.4f}  "
          f"balanced-accuracy: {metrics['balanced_accuracy']:.4f}")

    out_path = args.data_dir / f"eval_{args.model_path.stem}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
