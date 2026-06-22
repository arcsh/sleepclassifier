"""AttnSleep-style multi-resolution CNN + self-attention."""

from __future__ import annotations

import torch
from torch import nn

from sleepstage.models.base import SleepStageModel, register_model


class MultiResolutionCNN(nn.Module):
    """Parallel CNN branches with different kernel sizes."""

    def __init__(
        self,
        n_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_channels, out_channels, k, padding=k // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    nn.Dropout(dropout),
                )
                for k in kernel_sizes
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.pool(branch(x)).squeeze(-1) for branch in self.branches]
        return torch.cat(feats, dim=-1)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention over sequence of epoch embeddings."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(x, x, x, need_weights=True)
        return self.norm(x + attn_out), weights


@register_model("attnsleep")
class AttnSleep(SleepStageModel):
    """Multi-resolution CNN with multi-head self-attention."""

    is_sequence_model = True

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 5,
        sample_rate: int = 100,
        sequence_length: int = 20,
        cnn_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        n_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        cnn_channels = cnn_channels or [64, 128]
        kernel_sizes = kernel_sizes or [3, 15, 63]

        self.encoder = MultiResolutionCNN(n_channels, cnn_channels[0], kernel_sizes, dropout)
        embed_dim = self.encoder.out_dim
        self.proj = nn.Linear(embed_dim, cnn_channels[-1])
        self.attention = MultiHeadSelfAttention(cnn_channels[-1], n_heads, dropout)
        self.head = nn.Linear(cnn_channels[-1], n_classes)
        self._embed_dim = cnn_channels[-1]
        self._last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, c, t = x.shape
        flat = x.reshape(b * seq_len, c, t)
        embed = self.encoder(flat).reshape(b, seq_len, -1)
        embed = self.proj(embed)
        attended, weights = self.attention(embed)
        center = seq_len // 2
        self._last_embed = attended[:, center]
        self._last_attn = weights
        return self.head(self._last_embed)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.forward(x)
        return self._last_embed

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return last forward pass attention weights."""
        return self._last_attn

    def count_parameters(self) -> int:
        return super().count_parameters()
