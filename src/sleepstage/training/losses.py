"""Loss functions for imbalanced classification."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


def compute_class_weights(labels: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency class weights from training labels.

    Args:
        labels: 1D label tensor.
        n_classes: Number of classes.

    Returns:
        Weight tensor of shape (n_classes,).
    """
    counts = torch.bincount(labels, minlength=n_classes).float()
    return compute_class_weights_from_counts(counts, n_classes)


def compute_class_weights_from_counts(counts: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency weights from per-class epoch counts."""
    counts = counts.float()
    if counts.numel() < n_classes:
        padded = torch.zeros(n_classes)
        padded[: counts.numel()] = counts
        counts = padded
    counts = torch.clamp(counts[:n_classes], min=1.0)
    weights = 1.0 / counts
    return weights / weights.sum() * n_classes


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=cast(torch.Tensor | None, self.weight), reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(
    name: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory for training loss.

    Args:
        name: 'weighted_ce' or 'focal'.
        class_weights: Optional class weight tensor.
        focal_gamma: Focal loss gamma.

    Returns:
        Loss module.
    """
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    weight_tensor = class_weights if isinstance(class_weights, torch.Tensor) else None
    return nn.CrossEntropyLoss(weight=weight_tensor)
