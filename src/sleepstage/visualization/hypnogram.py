"""Hypnogram visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleepstage.constants import STAGE_NAMES
from sleepstage.visualization.style import ACCENT, ACCENT_2, apply_style, style_axes


def plot_hypnogram_overlay(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    output_path: Path,
    epoch_length_sec: float = 30.0,
) -> None:
    """Plot predicted vs ground-truth hypnogram."""
    apply_style()
    time_hours = np.arange(len(true_labels)) * epoch_length_sec / 3600.0
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.step(
        time_hours,
        true_labels,
        where="post",
        label="Ground truth",
        linewidth=1.6,
        color=ACCENT,
    )
    ax.step(
        time_hours,
        pred_labels,
        where="post",
        label="Predicted",
        linewidth=1.4,
        alpha=0.85,
        color=ACCENT_2,
    )
    ax.set_yticks(range(len(STAGE_NAMES)))
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Sleep stage")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Hypnogram overlay — one held-out night")
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
