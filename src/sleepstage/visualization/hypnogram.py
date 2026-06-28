"""Hypnogram visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleepstage.constants import STAGE_NAMES


def plot_hypnogram_overlay(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    output_path: Path,
    epoch_length_sec: float = 30.0,
) -> None:
    """Plot predicted vs ground-truth hypnogram.

    Args:
        true_labels: Ground truth stage indices.
        pred_labels: Predicted stage indices.
        output_path: Save path.
        epoch_length_sec: Epoch duration for x-axis.
    """
    time_hours = np.arange(len(true_labels)) * epoch_length_sec / 3600.0
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(time_hours, true_labels, where="post", label="Ground truth", linewidth=1)
    ax.step(time_hours, pred_labels, where="post", label="Predicted", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(STAGE_NAMES)))
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Sleep stage")
    ax.legend()
    ax.set_title("Hypnogram Overlay")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
