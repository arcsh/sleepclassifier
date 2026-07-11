"""Confusion matrix plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sleepstage.constants import STAGE_NAMES
from sleepstage.visualization.style import ACCENT, apply_style, prettify_model_name, style_axes


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap."""
    apply_style()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display = cm / row_sums
    else:
        display = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        display,
        annot=True,
        fmt=".0%" if normalize else "d",
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        cmap=sns.light_palette(ACCENT, as_cmap=True),
        vmin=0,
        vmax=1 if normalize else None,
        linewidths=0.5,
        linecolor="#ffffff",
        cbar_kws={"label": "Recall share" if normalize else "Count", "shrink": 0.82},
        ax=ax,
    )
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(prettify_model_name(title) if title else "Confusion matrix")
    style_axes(ax, hide_top_right=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
