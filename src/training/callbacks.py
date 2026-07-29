"""
Training callbacks — currently just early stopping.

Kept separate from trainer.py's loop so the stopping policy can be reasoned
about (and tested) independently of the training mechanics that call it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """
    Stops a training run when a monitored score hasn't improved for
    `patience` consecutive epochs.

    One instance per fold — cross-donor CV trains a fresh model per fold, so
    "best epoch" / "epochs since improvement" must be fold-local, not shared
    across folds. Call `reset()` (or construct a new instance) before each
    fold.

    Usage:
        stopper = EarlyStopping(patience=5)
        for epoch in range(n_epochs):
            val_macro_f1 = ...
            if stopper.step(val_macro_f1, epoch):
                break  # no improvement for `patience` epochs
        stopper.best_score, stopper.best_epoch
    """

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"  # "max" for metrics like macro-F1, "min" for loss

    best_score: float | None = field(init=False, default=None)
    best_epoch: int = field(init=False, default=-1)
    num_bad_epochs: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, epoch: int) -> bool:
        """
        Record this epoch's score.

        Returns True if training should stop now (no improvement for
        `patience` consecutive epochs).
        """
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

    def reset(self) -> None:
        """Clear state — call before starting a new fold."""
        self.best_score = None
        self.best_epoch = -1
        self.num_bad_epochs = 0
