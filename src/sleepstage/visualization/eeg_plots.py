"""EEG waveform plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleepstage.constants import STAGE_NAMES


def plot_example_epochs_per_class(
    epochs_by_class: dict[str, np.ndarray],
    sfreq: float,
    output_path: Path,
) -> None:
    """Plot one representative epoch per sleep stage.

    Args:
        epochs_by_class: Mapping stage name → 1D waveform.
        sfreq: Sampling rate.
        output_path: Save path for figure.
    """
    fig, axes_raw = plt.subplots(len(STAGE_NAMES), 1, figsize=(12, 10), sharex=True)
    axes_list = axes_raw if isinstance(axes_raw, np.ndarray) else [axes_raw]
    for ax, stage in zip(axes_list, STAGE_NAMES, strict=True):
        if stage in epochs_by_class:
            wave = epochs_by_class[stage]
            t = np.arange(len(wave)) / sfreq
            ax.plot(t, wave, linewidth=0.5)
        ax.set_ylabel(stage)
    axes_list[-1].set_xlabel("Time (s)")
    fig.suptitle("Example EEG Epochs per Sleep Stage")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_psd_per_class(
    psd_by_class: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot averaged PSD per class.

    Args:
        psd_by_class: Stage → (freqs, psd) tuples.
        output_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage, (freqs, psd) in psd_by_class.items():
        ax.semilogy(freqs, psd, label=stage)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend()
    ax.set_title("Average Power Spectral Density by Sleep Stage")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
