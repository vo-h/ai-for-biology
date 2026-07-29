"""
MLP cell-type classifier.

Input:  log-normalized HVG expression (2,000-dim by default).
Output: logits over cell-type classes.

Architecture: linear blocks with BatchNorm + ReLU + Dropout. BatchNorm is
important here — expression values vary by orders of magnitude across genes,
and per-layer normalization stabilizes training without needing careful
per-gene scaling upfront.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CellTypeMLP(nn.Module):
    """
    Args:
        n_genes:     Input dimensionality (number of HVGs).
        n_classes:   Number of cell-type classes.
        hidden_dims: Sizes of hidden layers. Default (512, 256) works well for
                     2k-gene input; add a layer for larger gene sets.
        dropout:     Dropout probability. 0.3 is a reasonable default for
                     scRNA-seq — expression is noisy and dropout regularization
                     meaningfully reduces overfitting to donor-specific patterns.
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        dims = [n_genes, *hidden_dims]
        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
