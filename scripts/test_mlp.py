#!/usr/bin/env python3
"""
Evaluate one or more saved CellTypeMLP checkpoints against a local per-donor
h5ad directory.

Reads model_config.json (n_genes, n_classes, architecture, label mapping,
gene list) from the checkpoint's directory — written automatically by
run_training — so architecture/label alignment never has to be re-specified
by hand. Validates the test data's genes match what the model was trained on
before running anything: CellTypeMLP's input is purely positional (a Linear
layer), so a silent gene mismatch would produce plausible-looking garbage
rather than an error.

Example:
    # Single checkpoint
    python scripts/test_mlp.py \
        --model-path results/mlp/best_model_fold0.pt \
        --data-dir results/donors-test

    # A whole directory of checkpoints (e.g. all folds) — evaluates each and
    # reports the mean/std across the group, not just one cherry-picked model
    python scripts/test_mlp.py \
        --model-dir results/mlp \
        --data-dir results/donors-test \
        --fname eval_all_folds.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anndata as ad
import numpy as np
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
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", type=Path,
                        help="Path to a single saved checkpoint, e.g. results/mlp/best_model_fold0.pt")
    group.add_argument("--model-dir", type=Path,
                        help="Directory of saved checkpoints (e.g. all folds from one run) — "
                             "evaluates every *.pt in it and reports the mean/std across the group.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory of per-donor h5ad files to evaluate on "
                        "(see scripts/download_dataset.py)")
    p.add_argument("--model-config", type=Path, default=None,
                   help="Path to model_config.json (default: alongside --model-path, "
                        "or inside --model-dir)")
    p.add_argument("--donors", nargs="+", default=None,
                   help="Restrict evaluation to these donor_ids (e.g. a fold's "
                        "held-out test_donors from training_results.json). "
                        "Default: every donor found in --data-dir.")
    p.add_argument("--fname", default=None,
                   help="Output filename, written into --data-dir. Default: "
                        "eval_{checkpoint stem}.json for --model-path, "
                        "eval_{model-dir name}_avg.json for --model-dir.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def evaluate_checkpoint(model_path: Path, model_config: dict, loader, int2label: dict, device) -> dict:
    """Load one checkpoint and return its compute_metrics() dict."""
    model = CellTypeMLP(
        n_genes=model_config["n_genes"],
        n_classes=model_config["n_classes"],
        hidden_dims=tuple(model_config["hidden_dims"]),
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  {model_path.name} ({model.n_params:,} params)")

    criterion = nn.CrossEntropyLoss()
    stats = eval_epoch(model, loader, criterion, device)
    return compute_metrics(stats["y_true"], stats["y_pred"], int2label)


def average_metrics(all_metrics: list[dict]) -> dict:
    """Mean + std of macro/weighted-F1, balanced accuracy, and per-class F1 across a group of models."""
    scalar_keys = ["macro_f1", "weighted_f1", "balanced_accuracy"]
    mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in scalar_keys}
    std = {k: float(np.std([m[k] for m in all_metrics])) for k in scalar_keys}

    per_class = {}
    for ct in all_metrics[0]["per_class"]:
        f1s = [m["per_class"][ct]["f1"] for m in all_metrics]
        per_class[ct] = {
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "support": all_metrics[0]["per_class"][ct]["support"],
        }
    return {"mean": mean, "std": std, "per_class": per_class}


def main():
    args = parse_args()

    if args.model_path is not None:
        checkpoints = [args.model_path]
        config_path = args.model_config or args.model_path.parent / "model_config.json"
    else:
        checkpoints = sorted(args.model_dir.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No *.pt checkpoints found in {args.model_dir}")
        config_path = args.model_config or args.model_dir / "model_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found — expected alongside the checkpoint(s), written "
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
            "HVG list the model(s) were trained on. Re-download it with the same "
            "--hvg-cache used for training."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # All checkpoints in the group share data/labels, so build the loader once
    # and reuse it — LocalDonorDataset yields a fresh iterator each pass.
    ds = LocalDonorDataset(args.data_dir, donor_ids=donor_ids, batch_size=args.batch_size, shuffle=False)
    collate = CensusCollateFn(label2int)
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers, collate_fn=collate)

    print(f"\nEvaluating {len(checkpoints)} checkpoint(s):")
    all_metrics = [evaluate_checkpoint(p, model_config, loader, int2label, device) for p in checkpoints]

    if len(checkpoints) == 1:
        result = all_metrics[0]
        print(f"\nmacro-F1: {result['macro_f1']:.4f}  "
              f"weighted-F1: {result['weighted_f1']:.4f}  "
              f"balanced-accuracy: {result['balanced_accuracy']:.4f}")
        default_fname = f"eval_{checkpoints[0].stem}.json"
    else:
        result = {
            "n_models": len(checkpoints),
            "checkpoints": [p.name for p in checkpoints],
            **average_metrics(all_metrics),
        }
        print(f"\nmacro-F1: {result['mean']['macro_f1']:.4f} ± {result['std']['macro_f1']:.4f}  "
              f"weighted-F1: {result['mean']['weighted_f1']:.4f} ± {result['std']['weighted_f1']:.4f}  "
              f"balanced-accuracy: {result['mean']['balanced_accuracy']:.4f} ± {result['std']['balanced_accuracy']:.4f}"
              f"  (n={len(checkpoints)} models)")
        default_fname = f"eval_{args.model_dir.name}_avg.json"

    out_path = args.data_dir / (args.fname or default_fname)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
