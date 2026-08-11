"""
Training and evaluation loops for the cell-type MLP.

Cross-donor evaluation is the core contribution: train on N-1 donor groups,
evaluate on the held-out group. Reports both random-split and cross-donor
macro-F1 so the accuracy gap is visible in the training output.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import cellxgene_census
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tiledbsoma_ml import experiment_dataloader

from src.data.cellxgene import (
    build_census_dataset, CensusCollateFn, fetch_metadata, get_label_encoder,
    LocalDonorDataset, list_available_donors, SOMA_CTX, CENSUS_VERSION,
)
from src.evaluation.cellxgene import group_k_fold_donors, macro_f1
from src.models.mlp import CellTypeMLP
from src.training.callbacks import EarlyStopping
from src.training.metadata import collect_hardware_info, save_run_metadata


@dataclass
class TrainConfig:
    tissues: list[str] = field(default_factory=lambda: ["blood"])
    n_donors: int = 30
    n_folds: int = 5
    n_epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    io_batch_size: int = 65_536
    shuffle_chunk_size: int = 64
    num_workers: int = 2
    hidden_dims: tuple = (512, 256)
    dropout: float = 0.3
    output_dir: Path = Path("results/mlp")
    census_version: str = CENSUS_VERSION
    data_dir: Path | None = None
    """If set, train from local per-donor h5ad files in this directory
    (see download_donor_h5ads) instead of streaming live from Census."""
    patience: int | None = None
    """If set, stop a fold's training early after this many epochs without
    val macro-F1 improvement (see src.training.callbacks.EarlyStopping).
    None (default) runs the full n_epochs every fold."""


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss, n_batches, n_cells = 0.0, 0, 0
    t0 = time.time()

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        n_cells += len(X)
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}", cells=f"{n_cells:,}")

    elapsed = time.time() - t0
    return {
        "loss": total_loss / max(n_batches, 1),
        "n_cells": n_cells,
        "n_batches": n_batches,
        "elapsed": elapsed,
        "time_per_batch_s": elapsed / max(n_batches, 1),
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for X, y in tqdm(loader, desc="   eval", leave=False, unit="batch"):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "loss": total_loss / max(n_batches, 1),
        "macro_f1": macro_f1(all_labels, all_preds),
        "n_cells": len(all_labels),
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def run_training(
    hvg_genes: list[str],
    cfg: TrainConfig,
    split_fn=group_k_fold_donors,
) -> list[dict]:
    """
    Full cross-donor training run.

    For each fold: train on N-1 donor groups → evaluate on held-out group.
    Returns a list of per-fold result dicts.

    split_fn yields DonorSplit objects (see src.evaluation.cellxgene) and
    defaults to group_k_fold_donors. Pass random_k_fold instead to get the
    random-split accuracy-inflation baseline from the same loop — donors can
    then overlap between train/test, which is exactly what's being measured.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    run_start = time.time()
    hardware = collect_hardware_info(device)
    fold_timings = []

    if cfg.data_dir is not None:
        print(f"Listing donors in {cfg.data_dir}...")
        donor_ids = list_available_donors(cfg.data_dir)
        if not donor_ids:
            raise FileNotFoundError(f"No *.h5ad files found in {cfg.data_dir}")
        dfs = []
        for d in donor_ids:
            obs = ad.read_h5ad(cfg.data_dir / f"{d}.h5ad", backed="r").obs[["cell_type", "donor_id"]].reset_index(drop=True)
            obs["_row_pos"] = np.arange(len(obs))  # position within this donor's own file
            dfs.append(obs)
        meta = pd.concat(dfs, ignore_index=True)
        # Gene set is fixed by what was downloaded, not by hvg_genes — read it
        # from disk rather than trust the caller passed a matching list (a
        # mismatch here silently breaks the model's input dim).
        gene_list = ad.read_h5ad(cfg.data_dir / f"{donor_ids[0]}.h5ad", backed="r").var["feature_name"].tolist()
        n_genes = len(gene_list)
    else:
        print("Fetching metadata...")
        meta = fetch_metadata(tissues=cfg.tissues, census_version=cfg.census_version)
        gene_list = hvg_genes
        n_genes = len(hvg_genes)

    if meta["donor_id"].nunique() > cfg.n_donors:
        top = meta["donor_id"].value_counts().head(cfg.n_donors).index
        meta = meta[meta["donor_id"].isin(top)].copy()

    # Filter to cell types with enough cells
    type_counts = meta["cell_type"].value_counts()
    meta = meta[meta["cell_type"].isin(type_counts[type_counts >= 20].index)].copy()
    meta = meta.reset_index(drop=True)

    label2int, int2label = get_label_encoder(meta["cell_type"].tolist())
    n_classes = len(label2int)
    print(f"{len(meta):,} cells | {meta['donor_id'].nunique()} donors | {n_classes} classes")

    # Everything needed to reconstruct + correctly run this run's checkpoints
    # later (scripts/cellxgene/test_mlp.py) — a .pt file is just a raw state_dict, with
    # no record of architecture, label mapping, or gene identity/order on its
    # own, and CellTypeMLP's input is purely positional (a Linear layer), so
    # a silent gene mismatch at eval time would produce plausible-looking
    # garbage rather than an error.
    (cfg.output_dir / "model_config.json").write_text(json.dumps({
        "n_genes": n_genes,
        "n_classes": n_classes,
        "hidden_dims": list(cfg.hidden_dims),
        "dropout": cfg.dropout,
        "int2label": int2label,
        "gene_list": gene_list,
    }, indent=2))

    fold_results = []
    collate = CensusCollateFn(label2int)

    # Local-file training never touches Census — no context needed there.
    census_ctx = (
        contextlib.nullcontext(None)
        if cfg.data_dir is not None
        else cellxgene_census.open_soma(census_version=cfg.census_version, context=SOMA_CTX)
    )

    with census_ctx as census:
        for split in split_fn(meta, n_folds=cfg.n_folds):
            print(f"\n{'='*50}")
            print(f"Fold {split.fold} — held-out donors: {split.test_donors}")

            train_meta = meta.iloc[split.train_idx]
            test_meta  = meta.iloc[split.test_idx]

            if cfg.data_dir is not None:
                # cell_indices, not just donor_ids: split_fn may put the same
                # donor's cells in both train and test (random_k_fold), so
                # whole-donor inclusion would be wrong there.
                train_cells = {d: g["_row_pos"].to_numpy() for d, g in train_meta.groupby("donor_id", observed=True)}
                test_cells  = {d: g["_row_pos"].to_numpy() for d, g in test_meta.groupby("donor_id", observed=True)}
                train_ds = LocalDonorDataset(
                    cfg.data_dir,
                    donor_ids=list(train_cells),
                    cell_indices=train_cells,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                )
                test_ds = LocalDonorDataset(
                    cfg.data_dir,
                    donor_ids=list(test_cells),
                    cell_indices=test_cells,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                )
                train_loader = DataLoader(train_ds, batch_size=None, num_workers=cfg.num_workers, collate_fn=collate)
                test_loader  = DataLoader(test_ds,  batch_size=None, num_workers=cfg.num_workers, collate_fn=collate)
            else:
                train_ds = build_census_dataset(
                    census,
                    soma_joinids=train_meta["soma_joinid"].tolist(),
                    hvg_genes=hvg_genes,
                    batch_size=cfg.batch_size,
                    io_batch_size=cfg.io_batch_size,
                    shuffle_chunk_size=cfg.shuffle_chunk_size,
                    shuffle=True,
                )
                test_ds = build_census_dataset(
                    census,
                    soma_joinids=test_meta["soma_joinid"].tolist(),
                    hvg_genes=hvg_genes,
                    batch_size=cfg.batch_size,
                    io_batch_size=cfg.io_batch_size,
                    shuffle_chunk_size=cfg.shuffle_chunk_size,
                    shuffle=False,
                )
                train_loader = experiment_dataloader(train_ds, num_workers=cfg.num_workers, collate_fn=collate)
                test_loader  = experiment_dataloader(test_ds,  num_workers=cfg.num_workers, collate_fn=collate)

            model = CellTypeMLP(
                n_genes=n_genes,
                n_classes=n_classes,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
            ).to(device)
            print(f"Model params: {model.n_params:,}")

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.n_epochs
            )
            criterion = nn.CrossEntropyLoss()

            best_f1, best_epoch = 0.0, 0
            epoch_logs = []
            stopper = EarlyStopping(patience=cfg.patience) if cfg.patience is not None else None

            epoch_pbar = tqdm(range(cfg.n_epochs), desc=f"Fold {split.fold}", unit="epoch")
            for epoch in epoch_pbar:
                train_ds.set_epoch(epoch)
                train_stats = train_epoch(model, train_loader, optimizer, criterion, device)
                val_stats   = eval_epoch(model, test_loader,  criterion, device)
                scheduler.step()

                log = {
                    "fold": split.fold,
                    "epoch": epoch,
                    "train_loss": round(train_stats["loss"], 4),
                    "val_loss": round(val_stats["loss"], 4),
                    "val_macro_f1": round(val_stats["macro_f1"], 4),
                    "train_cells": train_stats["n_cells"],
                    "val_cells": val_stats["n_cells"],
                    "elapsed_s": round(train_stats["elapsed"], 1),
                    "train_batches": train_stats["n_batches"],
                    "time_per_batch_s": round(train_stats["time_per_batch_s"], 4),
                }
                epoch_logs.append(log)
                epoch_pbar.set_postfix(
                    train_loss=f"{log['train_loss']:.4f}",
                    val_loss=f"{log['val_loss']:.4f}",
                    val_f1=f"{log['val_macro_f1']:.4f}",
                )

                if val_stats["macro_f1"] > best_f1:
                    best_f1 = val_stats["macro_f1"]
                    best_epoch = epoch
                    torch.save(
                        model.state_dict(),
                        cfg.output_dir / f"best_model_fold{split.fold}.pt",
                    )

                if stopper is not None and stopper.step(val_stats["macro_f1"], epoch):
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no val macro-F1 improvement for {cfg.patience} epochs)")
                    break

            print(f"  Best val macro-F1: {best_f1:.4f} at epoch {best_epoch}")
            fold_results.append({
                "fold": split.fold,
                "test_donors": split.test_donors,
                "best_macro_f1": best_f1,
                "best_epoch": best_epoch,
                "epochs": epoch_logs,
            })

            fold_train_time = sum(log["elapsed_s"] for log in epoch_logs)
            fold_train_cells = sum(log["train_cells"] for log in epoch_logs)
            fold_train_batches = sum(log["train_batches"] for log in epoch_logs)
            fold_timings.append({
                "fold": split.fold,
                "n_epochs": len(epoch_logs),  # actual epochs run — may be < cfg.n_epochs if stopped early
                "train_time_s": round(fold_train_time, 1),
                "total_train_cells": fold_train_cells,
                "avg_cells_per_s": round(fold_train_cells / fold_train_time, 1) if fold_train_time else None,
                "avg_time_per_batch_s": round(fold_train_time / fold_train_batches, 4) if fold_train_batches else None,
            })

    mean_f1 = np.mean([r["best_macro_f1"] for r in fold_results])
    std_f1 = np.std([r["best_macro_f1"] for r in fold_results])
    print(f"\nCross-donor macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    save_run_metadata(
        cfg.output_dir / "run_metadata.json",
        hardware=hardware,
        cfg=cfg,
        fold_timings=fold_timings,
        total_wall_time_s=time.time() - run_start,
    )
    print(f"Run metadata saved to {cfg.output_dir / 'run_metadata.json'}")

    return fold_results
