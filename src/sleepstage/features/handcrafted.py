"""Handcrafted feature extraction for classical baseline."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.stats import entropy

# Frequency bands for sleep EEG (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def bandpower(epoch: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    """Compute relative band power for one epoch.

    Args:
        epoch: 1D signal array.
        sfreq: Sampling rate.
        band: (low, high) Hz.

    Returns:
        Relative power in the band.
    """
    freqs, psd = signal.welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    band_power = float(trapezoid(psd[idx], freqs[idx]))
    total_power = float(trapezoid(psd, freqs))
    return float(band_power / (total_power + 1e-12))


def spectral_entropy(epoch: np.ndarray, sfreq: float) -> float:
    """Compute spectral entropy of an epoch.

    Args:
        epoch: 1D signal.
        sfreq: Sampling rate.

    Returns:
        Normalized spectral entropy.
    """
    _, psd = signal.welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(entropy(psd_norm))


def hjorth_mobility(epoch: np.ndarray) -> float:
    """Hjorth mobility parameter.

    Args:
        epoch: 1D signal.

    Returns:
        Mobility value.
    """
    d1 = np.diff(epoch)
    var0 = np.var(epoch)
    var1 = np.var(d1)
    return float(np.sqrt(var1 / (var0 + 1e-12)))


def hjorth_complexity(epoch: np.ndarray) -> float:
    """Hjorth complexity parameter.

    Args:
        epoch: 1D signal.

    Returns:
        Complexity value.
    """
    mobility = hjorth_mobility(epoch)
    d1 = np.diff(epoch)
    d2 = np.diff(d1)
    mobility_d1 = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)))
    return mobility_d1 / (mobility + 1e-12)


def zero_crossing_rate(epoch: np.ndarray) -> float:
    """Fraction of sign changes in the signal.

    Args:
        epoch: 1D signal.

    Returns:
        Zero-crossing rate.
    """
    signs = np.sign(epoch)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / max(len(epoch) - 1, 1))


def extract_epoch_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract handcrafted features for one epoch (first channel if multi).

    Args:
        epoch: Shape (n_channels, n_samples) or (n_samples,).
        sfreq: Sampling rate.

    Returns:
        1D feature vector.
    """
    if epoch.ndim == 2:
        signal_1d = epoch[0]
    else:
        signal_1d = epoch

    features: list[float] = []
    for band in BANDS.values():
        features.append(bandpower(signal_1d, sfreq, band))
    features.append(spectral_entropy(signal_1d, sfreq))
    features.append(hjorth_mobility(signal_1d))
    features.append(hjorth_complexity(signal_1d))
    features.append(zero_crossing_rate(signal_1d))
    return np.array(features, dtype=np.float32)


def extract_features_batch(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract features for a batch of epochs.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples).
        sfreq: Sampling rate.

    Returns:
        Shape (n_epochs, n_features).
    """
    return np.stack([extract_epoch_features(ep, sfreq) for ep in epochs])
