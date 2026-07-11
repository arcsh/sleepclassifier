"""Embedding visualization (t-SNE/UMAP)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from sleepstage.constants import STAGE_NAMES
from sleepstage.visualization.style import apply_style, stage_color, style_axes


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    perplexity: float = 30.0,
) -> None:
    """Project embeddings to 2D via t-SNE."""
    apply_style()
    n = len(embeddings)
    perp = min(perplexity, max(5, n // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for idx, stage in enumerate(STAGE_NAMES):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=stage,
            alpha=0.55,
            s=14,
            color=stage_color(stage),
            edgecolors="none",
        )
    ax.legend(title="Stage", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("CNN1D epoch embeddings (2D projection)")
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
