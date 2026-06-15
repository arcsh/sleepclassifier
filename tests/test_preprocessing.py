"""Tests for preprocessing correctness."""

from __future__ import annotations

import numpy as np

from sleepstage.constants import RAW_LABEL_MAP, STAGE_TO_IDX
from sleepstage.features.preprocessing import (
    find_sleep_onset_offset,
    trim_wake_padding,
    zscore_per_recording,
)


def test_label_mapping() -> None:
    assert RAW_LABEL_MAP["Sleep stage 3"] == "N3"
    assert RAW_LABEL_MAP["Sleep stage 4"] == "N3"
    assert RAW_LABEL_MAP["Sleep stage ?"] is None
    assert RAW_LABEL_MAP["Movement time"] is None
    assert STAGE_TO_IDX["REM"] == 4


def test_wake_trimming() -> None:
    labels = np.array([0, 0, 0, 0, 1, 2, 2, 3, 2, 0, 0, 0, 0, 0])
    epochs = np.random.randn(len(labels), 1, 100).astype(np.float32)
    trimmed_e, trimmed_l = trim_wake_padding(epochs, labels, padding_epochs=1)
    assert len(trimmed_l) < len(labels)
    assert trimmed_l[0] == 0  # one wake epoch before onset
    assert 3 in trimmed_l


def test_zscore_normalization() -> None:
    epochs = np.random.randn(20, 2, 100).astype(np.float64) * 5 + 10
    normed = zscore_per_recording(epochs)
    for ch in range(epochs.shape[1]):
        channel_data = normed[:, ch, :]
        np.testing.assert_allclose(channel_data.mean(), 0, atol=1e-5)
        np.testing.assert_allclose(channel_data.std(), 1, atol=1e-4)


def test_sleep_onset_offset() -> None:
    labels = np.array([0, 0, 1, 2, 2, 0])
    onset, offset = find_sleep_onset_offset(labels)
    assert onset == 2
    assert offset == 4
