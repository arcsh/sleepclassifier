"""Tests for dataset and subject-wise splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sleepstage.data.dataset import EpochDataset, SequenceEpochDataset
from sleepstage.features.preprocessing import save_processed_npz
from sleepstage.features.splits import (
    compute_group_kfold_splits,
    persist_splits,
    verify_no_subject_leakage,
)


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    subjects = ["SC400", "SC401", "SC402", "SC403", "SC404", "SC405"]
    for i, subj in enumerate(subjects):
        for night in ["1", "2"]:
            epochs = np.random.randn(50, 1, 3000).astype(np.float32)
            labels = np.random.randint(0, 5, size=50)
            path = tmp_path / f"{subj}_{night}.npz"
            save_processed_npz(path, epochs, labels, subj, night, 100.0)
            rows.append({"subject_id": subj, "night_id": night, "path": str(path), "n_epochs": 50})
    return pd.DataFrame(rows)


def test_group_kfold_no_subject_leakage(synthetic_manifest: pd.DataFrame) -> None:
    subject_ids = synthetic_manifest["subject_id"].tolist()
    folds = compute_group_kfold_splits(subject_ids, n_folds=3)
    for fold in folds:
        verify_no_subject_leakage(fold)
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])
        test = set(fold["test_subjects"])
        assert len(train | val | test) > 0
        # No subject in multiple splits
        all_subjects_in_fold = train | val | test
        for sid in all_subjects_in_fold:
            appearances = sum(sid in s for s in [train, val, test])
            assert appearances == 1


def test_epoch_dataset_shapes(synthetic_manifest: pd.DataFrame) -> None:
    ds = EpochDataset(synthetic_manifest, list(range(len(synthetic_manifest))))
    x, y, subj, night = ds[0]
    assert x.shape == (1, 3000)
    assert y.dtype == torch.long
    assert isinstance(subj, str)


def test_sequence_windows_same_night(synthetic_manifest: pd.DataFrame) -> None:
    seq_len = 10
    ds = SequenceEpochDataset(synthetic_manifest, [0], seq_len)
    for i in range(min(5, len(ds))):
        x, y, subj, night = ds[i]
        assert x.shape == (seq_len, 1, 3000)
        assert subj == synthetic_manifest.iloc[0]["subject_id"]


def test_split_persistence(tmp_path: Path, synthetic_manifest: pd.DataFrame) -> None:
    folds = compute_group_kfold_splits(synthetic_manifest["subject_id"].tolist(), n_folds=3)
    out = tmp_path / "splits"
    persist_splits(folds, out)
    assert (out / "fold_0.json").exists()
