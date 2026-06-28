"""Embedding visualization (t-SNE/UMAP)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from sleepstage.constants import STAGE_NAMES


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    perplexity: float = 30.0,
) -> None:
    """Project embeddings to 2D via t-SNE.

    Args:
        embeddings: Shape (N, D).
        labels: Integer stage labels.
        output_path: Save path.
        perplexity: t-SNE perplexity.
    """
    n = len(embeddings)
    perp = min(perplexity, max(5, n // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, stage in enumerate(STAGE_NAMES):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], label=stage, alpha=0.5, s=10)
    ax.legend()
    ax.set_title("t-SNE of Learned Epoch Embeddings")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
