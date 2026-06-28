"""Confusion matrix plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sleepstage.constants import STAGE_NAMES


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix counts.
        output_path: Save path.
        normalize: Row-normalize if True.
        title: Figure title.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display = cm / row_sums
    else:
        display = cm.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        display,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
