"""Preprocessing: filtering, epoching, wake trimming, normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy import signal

from sleepstage.constants import EPOCH_LENGTH_SEC, N_CLASSES, STAGE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    bandpass_low: float = 0.3
    bandpass_high: float = 35.0
    epoch_length_sec: float = EPOCH_LENGTH_SEC
    wake_padding_min: float = 30.0
    apply_notch: bool = False
    notch_freq: float = 50.0


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Apply zero-phase bandpass filter along the time axis.

    Args:
        data: Array shaped (n_channels, n_samples).
        sfreq: Sampling frequency in Hz.
        low: Low cutoff Hz.
        high: High cutoff Hz.

    Returns:
        Filtered data with same shape.
    """
    nyq = sfreq / 2.0
    b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def apply_notch(data: np.ndarray, sfreq: float, freq: float = 50.0) -> np.ndarray:
    """Apply notch filter at line frequency.

    Args:
        data: Array shaped (n_channels, n_samples).
        sfreq: Sampling frequency.
        freq: Notch center frequency.

    Returns:
        Filtered data.
    """
    nyq = sfreq / 2.0
    b, a = signal.iirnotch(freq / nyq, Q=30.0)
    return signal.filtfilt(b, a, data, axis=-1)


def epoch_signal(
    data: np.ndarray,
    sfreq: float,
    epoch_length_sec: float,
) -> np.ndarray:
    """Segment continuous data into non-overlapping epochs.

    Args:
        data: Shape (n_channels, n_samples).
        sfreq: Sampling rate.
        epoch_length_sec: Epoch duration.

    Returns:
        Array shaped (n_epochs, n_channels, n_samples_per_epoch).
    """
    samples_per_epoch = int(sfreq * epoch_length_sec)
    n_epochs = data.shape[1] // samples_per_epoch
    trimmed = data[:, : n_epochs * samples_per_epoch]
    return trimmed.reshape(data.shape[0], n_epochs, samples_per_epoch).transpose(1, 0, 2)


def map_and_filter_epochs(
    epochs: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only epochs with valid sleep-stage labels.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples).
        labels: Per-epoch label indices (-1 for invalid).
        valid_mask: Boolean mask of labeled epochs.

    Returns:
        Filtered epochs and labels.
    """
    keep = valid_mask & (labels >= 0)
    return epochs[keep], labels[keep]


def find_sleep_onset_offset(labels: np.ndarray) -> tuple[int, int]:
    """Find first and last non-wake epoch indices (W=0).

    Args:
        labels: Integer stage labels.

    Returns:
        Tuple of (onset_idx, offset_idx) inclusive.
    """
    non_wake = np.where(labels != 0)[0]
    if len(non_wake) == 0:
        return 0, len(labels) - 1
    return int(non_wake[0]), int(non_wake[-1])


def trim_wake_padding(
    epochs: np.ndarray,
    labels: np.ndarray,
    padding_epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim excess wake before sleep onset and after final sleep.

    Args:
        epochs: Epoch array.
        labels: Stage labels.
        padding_epochs: Wake epochs to keep on each side.

    Returns:
        Trimmed epochs and labels.
    """
    onset, offset = find_sleep_onset_offset(labels)
    start = max(0, onset - padding_epochs)
    end = min(len(labels), offset + padding_epochs + 1)
    return epochs[start:end], labels[start:end]


def zscore_per_recording(epochs: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel using recording statistics.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples).

    Returns:
        Normalized epochs.
    """
    mean = epochs.mean(axis=(0, 2), keepdims=True)
    std = epochs.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (epochs - mean) / std


def class_counts(labels: np.ndarray) -> dict[str, int]:
    """Count epochs per stage.

    Args:
        labels: Integer stage indices.

    Returns:
        Mapping from stage name to count.
    """
    counts = {name: 0 for name in STAGE_NAMES}
    for idx in range(N_CLASSES):
        counts[STAGE_NAMES[idx]] = int(np.sum(labels == idx))
    return counts


def preprocess_recording(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline for one recording.

    Args:
        raw: MNE raw object (channels already picked).
        annotations: Hypnogram annotations.
        config: Preprocessing parameters.

    Returns:
        Tuple of (epochs, labels) after filtering, epoching, trimming, normalization.
    """
    from sleepstage.data.edf_io import annotations_to_epoch_labels

    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()
    data = bandpass_filter(data, sfreq, config.bandpass_low, config.bandpass_high)
    if config.apply_notch:
        data = apply_notch(data, sfreq, config.notch_freq)

    epochs = epoch_signal(data, sfreq, config.epoch_length_sec)
    duration = data.shape[1] / sfreq
    labels, valid = annotations_to_epoch_labels(annotations, config.epoch_length_sec, duration)
    # Align label array length with epoch count
    if len(labels) > len(epochs):
        labels = labels[: len(epochs)]
        valid = valid[: len(epochs)]
    elif len(labels) < len(epochs):
        pad = len(epochs) - len(labels)
        labels = np.concatenate([labels, np.full(pad, -1)])
        valid = np.concatenate([valid, np.zeros(pad, dtype=bool)])

    epochs, labels = map_and_filter_epochs(epochs, labels, valid)
    padding_epochs = int((config.wake_padding_min * 60) / config.epoch_length_sec)
    epochs, labels = trim_wake_padding(epochs, labels, padding_epochs)
    epochs = zscore_per_recording(epochs)
    return epochs, labels


def save_processed_npz(
    path: Path,
    epochs: np.ndarray,
    labels: np.ndarray,
    subject_id: str,
    night_id: str,
    sample_rate: float,
) -> None:
    """Persist processed epochs to NPZ.

    Args:
        path: Output file path.
        epochs: Processed epoch array.
        labels: Stage labels.
        subject_id: Subject identifier.
        night_id: Night identifier.
        sample_rate: Sampling rate Hz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        epochs=epochs,
        labels=labels,
        subject_id=subject_id,
        night_id=night_id,
        sample_rate=sample_rate,
    )


@dataclass
class ProcessedRecord:
    """Cached preprocessed recording."""

    epochs: np.ndarray
    labels: np.ndarray
    subject_id: str
    night_id: str
    sample_rate: float


def load_processed_npz(path: Path) -> ProcessedRecord:
    """Load processed NPZ file.

    Args:
        path: NPZ file path.

    Returns:
        Processed epochs, labels, and metadata.
    """
    data = np.load(path, allow_pickle=True)
    return ProcessedRecord(
        epochs=data["epochs"],
        labels=data["labels"],
        subject_id=str(data["subject_id"]),
        night_id=str(data["night_id"]),
        sample_rate=float(data["sample_rate"]),
    )
