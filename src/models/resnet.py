"""
ResNet-18 adapted for RxRx1's 6-channel fluorescence microscopy images.

Input:  6-channel standardized fluorescence image, 512x512
        (Hoechst/ConA/Phalloidin/Syto14/MitoTracker/WGA — see docs/rxrx1.md).
Output: logits over 1,108 siRNA classes.

torchvision's resnet18 expects 3-channel RGB input; conv1 is rebuilt for 6
input channels, the rest of the architecture is unchanged. No ImageNet
pretrained weights: conv1's weight shape wouldn't match, and naively
repeating pretrained RGB weights across 6 fluorescence channels would carry
over a color-statistics prior that has nothing to do with these stains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.data.rxrx1 import N_CHANNELS, N_SIRNA_CLASSES


class RxRx1ResNet18(nn.Module):
    """
    Args:
        n_channels: Input channel count. Default is the full RxRx1 stain
                    panel (src.data.rxrx1.N_CHANNELS); pass a smaller count
                    to match a `channels` subset used when building
                    RxRx1Dataset.
        n_classes:  Output classes. Default is all 1,108 siRNA reagents
                    (src.data.rxrx1.N_SIRNA_CLASSES / docs/rxrx1.md).
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_SIRNA_CLASSES):
        super().__init__()
        self.net = resnet18(weights=None, num_classes=n_classes)
        self.net.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
