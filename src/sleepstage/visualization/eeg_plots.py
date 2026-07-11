"""EEG waveform plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleepstage.constants import STAGE_NAMES
from sleepstage.visualization.style import apply_style, stage_color, style_axes


def plot_example_epochs_per_class(
    epochs_by_class: dict[str, np.ndarray],
    sfreq: float,
    output_path: Path,
) -> None:
    """Plot one representative epoch per sleep stage."""
    apply_style()
    fig, axes_raw = plt.subplots(len(STAGE_NAMES), 1, figsize=(12, 9), sharex=True)
    axes_list = axes_raw if isinstance(axes_raw, np.ndarray) else [axes_raw]
    for ax, stage in zip(axes_list, STAGE_NAMES, strict=True):
        if stage in epochs_by_class:
            wave = epochs_by_class[stage]
            t = np.arange(len(wave)) / sfreq
            ax.plot(t, wave, linewidth=0.65, color=stage_color(stage))
        ax.set_ylabel(stage, fontweight="600", color=stage_color(stage))
        ax.set_xlim(0, 30)
        style_axes(ax)
    axes_list[-1].set_xlabel("Time (s)")
    fig.suptitle("Example 30-second EEG epochs by sleep stage", fontsize=14, fontweight="600", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_psd_per_class(
    psd_by_class: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot averaged PSD per class."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for stage, (freqs, psd) in psd_by_class.items():
        ax.semilogy(freqs, psd, label=stage, color=stage_color(stage), linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_xlim(0, 35)
    ax.legend(title="Stage", ncol=5, loc="upper right")
    ax.set_title("Average spectral profile by sleep stage")
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
