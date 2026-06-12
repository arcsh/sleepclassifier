"""Base model interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

MODEL_REGISTRY: dict[str, type["SleepStageModel"]] = {}


def register_model(name: str):
    """Decorator to register a model class.

    Args:
        name: Registry key.

    Returns:
        Decorator function.
    """

    def decorator(cls: type[SleepStageModel]) -> type[SleepStageModel]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


class SleepStageModel(nn.Module, ABC):
    """Abstract sleep stage classifier."""

    n_classes: int = 5
    is_sequence_model: bool = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits.

        Args:
            x: Input tensor. Single-epoch: (B, C, T). Sequence: (B, L, C, T).

        Returns:
            Logits shaped (B, n_classes) for center-epoch prediction.
        """
        ...

    def count_parameters(self) -> int:
        """Return trainable parameter count.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate representations for visualization.

        Args:
            x: Model input.

        Returns:
            Embedding tensor.
        """
        logits = self.forward(x)
        return logits
