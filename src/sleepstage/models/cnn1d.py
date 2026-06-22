"""1D CNN single-epoch classifier."""

from __future__ import annotations

import torch
from torch import nn

from sleepstage.models.base import SleepStageModel, register_model


@register_model("cnn1d")
class CNN1D(SleepStageModel):
    """Multi-layer 1D CNN operating on raw epoch waveforms."""

    is_sequence_model = False

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        sample_rate: int = 100,
        channels: list[int] | None = None,
        kernel_size: int = 50,
        dropout: float = 0.3,
    ) -> None:
        """Initialize CNN1D.

        Args:
            n_channels: Input EEG channels.
            n_classes: Output classes.
            sample_rate: Sampling rate (stored for reference).
            channels: Conv channel sizes.
            kernel_size: Conv kernel size.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_classes = n_classes
        self.sample_rate = sample_rate
        channels = channels or [64, 128, 128]

        layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels[-1], n_classes)
        self._embed_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # Sequence input: use center epoch only for standalone CNN
            center = x.shape[1] // 2
            x = x[:, center]
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        self._last_embed = h
        return self.head(h)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.forward(x)
        return self._last_embed
