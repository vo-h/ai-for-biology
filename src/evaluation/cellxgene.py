"""
Cross-donor evaluation splits and metrics for the CELLxGENE Census pipeline.

Splitting enforces that no donor appears in both train and test — reflecting
real deployment where a model trained on existing samples must annotate a new
patient's biopsy. Random splits leak donor identity and inflate accuracy.
Metrics are implemented in numpy (no sklearn dependency) and accept integer
label arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Cross-donor splits
# ---------------------------------------------------------------------------

@dataclass
class DonorSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_donors: list[str]
    fold: int


def random_split(
    n: int,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random train/test split — included only to show the inflated accuracy
    that results from ignoring donor identity. Not used for real evaluation.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_fraction))
    return idx[:split], idx[split:]


def leave_one_donor_out(
    metadata: pd.DataFrame,
    donor_col: str = "donor_id",
    max_folds: int | None = None,
) -> Generator[DonorSplit, None, None]:
    """Yield one DonorSplit per donor, holding out that donor each time."""
    donors = sorted(metadata[donor_col].unique())
    if max_folds is not None:
        donors = donors[:max_folds]

    for fold, held_out in enumerate(donors):
        test_mask = metadata[donor_col] == held_out
        yield DonorSplit(
            train_idx=np.where(~test_mask)[0],
            test_idx=np.where(test_mask)[0],
            test_donors=[held_out],
            fold=fold,
        )


def group_k_fold_donors(
    metadata: pd.DataFrame,
    n_folds: int = 5,
    donor_col: str = "donor_id",
    seed: int = 42,
) -> Generator[DonorSplit, None, None]:
    """
    K-fold where each fold holds out a disjoint set of donors.

    Donors are shuffled then assigned to folds round-robin, so each fold
    gets a similar number of donors. More stable than LODO when there are
    many donors — fewer model fits, more test cells per fold.
    """
    donors = np.array(sorted(metadata[donor_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(donors)

    donor_to_fold = {d: i % n_folds for i, d in enumerate(donors)}

    for fold in range(n_folds):
        test_donors = [d for d, f in donor_to_fold.items() if f == fold]
        test_mask = metadata[donor_col].isin(test_donors)
        yield DonorSplit(
            train_idx=np.where(~test_mask)[0],
            test_idx=np.where(test_mask)[0],
            test_donors=test_donors,
            fold=fold,
        )


def random_k_fold(
    metadata: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> Generator[DonorSplit, None, None]:
    """
    K-fold where each fold's test set is a random slice of cells, ignoring
    donor identity — the same donor's cells can land in both train and test.
    Drop-in replacement for group_k_fold_donors, to show the resulting
    accuracy inflation (see random_split).
    """
    n = len(metadata)
    for fold in range(n_folds):
        train_idx, test_idx = random_split(n, test_fraction=1 / n_folds, seed=seed + fold)
        test_donors = sorted(metadata.iloc[test_idx]["donor_id"].unique().tolist())
        yield DonorSplit(train_idx=train_idx, test_idx=test_idx, test_donors=test_donors, fold=fold)


def summarize_split(split: DonorSplit, metadata: pd.DataFrame) -> dict:
    train_meta = metadata.iloc[split.train_idx]
    test_meta = metadata.iloc[split.test_idx]
    return {
        "fold": split.fold,
        "n_train": len(split.train_idx),
        "n_test": len(split.test_idx),
        "n_train_donors": train_meta["donor_id"].nunique(),
        "n_test_donors": len(split.test_donors),
        "cell_types_only_in_test": sorted(
            set(test_meta["cell_type"].unique())
            - set(train_meta["cell_type"].unique())
        ),
    }


# ---------------------------------------------------------------------------
# Core metric primitives
# ---------------------------------------------------------------------------

def _per_class_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, dict]:
    """Return TP, FP, FN, support per class."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    stats = {}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        stats[c] = {"tp": tp, "fp": fp, "fn": fn, "support": int((y_true == c).sum())}
    return stats


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    stats = _per_class_stats(y_true, y_pred)
    return float(np.mean([_f1(**{k: v[k] for k in ("tp", "fp", "fn")})
                          for v in stats.values()]))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    stats = _per_class_stats(y_true, y_pred)
    total = len(y_true)
    if total == 0:
        return 0.0
    return float(sum(
        _f1(v["tp"], v["fp"], v["fn"]) * v["support"]
        for v in stats.values()
    ) / total)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall."""
    stats = _per_class_stats(y_true, y_pred)
    recalls = []
    for v in stats.values():
        denom = v["tp"] + v["fn"]
        recalls.append(v["tp"] / denom if denom > 0 else 0.0)
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# Aggregate reports
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    int2label: dict[int, str] | None = None,
) -> dict:
    stats = _per_class_stats(y_true, y_pred)
    per_class = {
        (int2label[c] if int2label else str(c)): {
            "f1": round(_f1(v["tp"], v["fp"], v["fn"]), 4),
            "support": v["support"],
        }
        for c, v in stats.items()
    }
    return {
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "weighted_f1": round(weighted_f1(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 4),
        "per_class": per_class,
    }


def accuracy_gap_report(
    random_metrics: dict,
    cross_donor_metrics: dict,
) -> pd.DataFrame:
    """Gap between random-split and cross-donor accuracy — the central finding."""
    rows = []
    for metric in ["macro_f1", "weighted_f1", "balanced_accuracy"]:
        r, c = random_metrics[metric], cross_donor_metrics[metric]
        rows.append({
            "metric": metric,
            "random_split": r,
            "cross_donor": c,
            "gap_pp": round((r - c) * 100, 2),
        })
    return pd.DataFrame(rows)


def per_class_f1_comparison(
    random_metrics: dict,
    cross_donor_metrics: dict,
) -> pd.DataFrame:
    """Per-cell-type F1 under both splits, sorted by cross-donor F1."""
    rand = pd.DataFrame(random_metrics["per_class"]).T.rename(
        columns={"f1": "f1_random"}
    )
    cd = pd.DataFrame(cross_donor_metrics["per_class"]).T.rename(
        columns={"f1": "f1_cross_donor"}
    )
    merged = rand[["f1_random", "support"]].join(cd[["f1_cross_donor"]], how="outer")
    merged["f1_drop"] = merged["f1_random"] - merged["f1_cross_donor"]
    return (
        merged.sort_values("f1_cross_donor")
        .reset_index()
        .rename(columns={"index": "cell_type"})
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    normalize: bool = True,
) -> np.ndarray:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums > 0)
    return cm


def confusion_matrix_df(
    y_true, y_pred, labels: list, normalize: bool = True,
) -> pd.DataFrame:
    cm = confusion_matrix_np(y_true, y_pred, labels, normalize=normalize)
    return pd.DataFrame(cm, index=labels, columns=labels)


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str = "Confusion Matrix",
    figsize: tuple = (14, 12),
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_df, ax=ax, cmap="Blues", vmin=0, vmax=1,
        linewidths=0.3, square=True,
        xticklabels=[l[:25] for l in cm_df.columns],
        yticklabels=[l[:25] for l in cm_df.index],
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def rare_cell_type_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rare_threshold: int = 100,
) -> pd.DataFrame:
    """Compare macro-F1 on rare vs. common cell types."""
    unique, counts = np.unique(y_true, return_counts=True)
    rare = set(unique[counts < rare_threshold].tolist())
    common = set(unique[counts >= rare_threshold].tolist())

    rows = []
    for label, group in [("rare", rare), ("common", common)]:
        mask = np.isin(y_true, list(group))
        if mask.sum() == 0:
            continue
        rows.append({
            "group": label,
            "n_cell_types": len(group),
            "n_cells": int(mask.sum()),
            "macro_f1": round(macro_f1(y_true[mask], y_pred[mask]), 4),
        })
    return pd.DataFrame(rows)


def donor_accuracy_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    donor_ids: np.ndarray,
) -> pd.DataFrame:
    """Per-donor macro-F1. High variance signals batch effects."""
    rows = []
    for donor in np.unique(donor_ids):
        mask = donor_ids == donor
        rows.append({
            "donor_id": donor,
            "macro_f1": round(macro_f1(y_true[mask], y_pred[mask]), 4),
            "n_cells": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("macro_f1").reset_index(drop=True)
