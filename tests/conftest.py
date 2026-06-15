"""Synthetic EDF fixtures for testing."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np


def make_synthetic_hypnogram_edf(path: Path, duration_sec: float = 600.0) -> None:
    """Create a minimal hypnogram EDF+ with sleep stage annotations."""
    info = mne.create_info(["Hypnogram"], sfreq=1.0, ch_types=["stim"])
    data = np.zeros((1, int(duration_sec)))
    raw = mne.io.RawArray(data, info)
    onsets = [0, 120, 240, 360, 480]
    durations = [120, 120, 120, 120, 120]
    descriptions = [
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage R",
    ]
    annotations = mne.Annotations(onsets, durations, descriptions)
    raw.set_annotations(annotations)
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True)


def make_synthetic_psg_edf(path: Path, duration_sec: float = 600.0, sfreq: float = 100.0) -> None:
    """Create a minimal PSG EDF with standard Sleep-EDF channel names."""
    ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=["eeg", "eeg", "eog"])
    n_samples = int(duration_sec * sfreq)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 10e-6, (len(ch_names), n_samples))
    raw = mne.io.RawArray(data, info)
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True)
