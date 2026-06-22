"""CNN + BiLSTM sequence model (DeepSleepNet-style)."""

from __future__ import annotations

import torch
from torch import nn

from sleepstage.models.base import SleepStageModel, register_model


class EpochCNNEncoder(nn.Module):
    """Shared CNN encoder for per-epoch embeddings."""

    def __init__(
        self,
        n_channels: int,
        cnn_channels: list[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in cnn_channels:
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
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.net(x)
        return self.pool(h).squeeze(-1)


@register_model("cnn_bilstm")
class CNNBiLSTM(SleepStageModel):
    """Two-stage CNN feature extractor + BiLSTM with residual skip."""

    is_sequence_model = True

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        sample_rate: int = 100,
        sequence_length: int = 20,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 50,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        cnn_channels = cnn_channels or [64, 128, 128]

        self.encoder = EpochCNNEncoder(n_channels, cnn_channels, cnn_kernel_size, dropout)
        embed_dim = self.encoder.out_dim
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden_size * 2
        self.skip_proj = nn.Linear(embed_dim, lstm_out)
        self.head = nn.Linear(lstm_out + embed_dim, n_classes)
        self._embed_dim = lstm_out + embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, T)
        b, seq_len, c, t = x.shape
        flat = x.reshape(b * seq_len, c, t)
        epoch_embed = self.encoder(flat).reshape(b, seq_len, -1)

        lstm_out, _ = self.lstm(epoch_embed)
        center = seq_len // 2
        lstm_center = lstm_out[:, center]
        cnn_center = epoch_embed[:, center]
        skip = self.skip_proj(cnn_center)
        combined = torch.cat([lstm_center + skip, cnn_center], dim=-1)
        self._last_embed = combined
        return self.head(combined)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.forward(x)
        return self._last_embed
