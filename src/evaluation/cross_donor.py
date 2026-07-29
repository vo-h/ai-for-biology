"""
Cross-donor evaluation splits.

Enforces that no donor appears in both train and test — reflecting real
deployment where a model trained on existing samples must annotate a new
patient's biopsy. Random splits leak donor identity and inflate accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd


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
