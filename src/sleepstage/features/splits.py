"""Subject-wise cross-validation splits."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from sleepstage.utils.io import load_json, save_json

logger = logging.getLogger(__name__)


class FoldSplit(TypedDict):
    """One fold of subject-wise train/val/test assignments."""

    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    fold_index: int


def compute_group_kfold_splits(
    subject_ids: list[str],
    n_folds: int,
    seed: int = 42,
) -> list[FoldSplit]:
    """Compute subject-wise k-fold splits.

    Args:
        subject_ids: One subject ID per recording/night row.
        n_folds: Number of folds.
        seed: Unused for GroupKFold but kept for API consistency.

    Returns:
        List of fold dicts with train_subjects, val_subjects, test_subjects keys.
    """
    _ = seed
    unique_subjects = sorted(set(subject_ids))
    subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subject_to_idx[s] for s in subject_ids])

    gkf = GroupKFold(n_splits=n_folds)
    folds: list[FoldSplit] = []
    indices = np.arange(len(subject_ids))

    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(indices, groups=groups)):
        train_val_subjects = set(subject_ids[i] for i in train_val_idx)
        test_subjects = set(subject_ids[i] for i in test_idx)
        # Split train_val into train/val by subject (80/20 of subjects)
        tv_list = sorted(train_val_subjects)
        n_val = max(1, len(tv_list) // 5)
        val_subjects = set(tv_list[-n_val:])
        train_subjects = train_val_subjects - val_subjects

        folds.append(
            {
                "train_subjects": sorted(train_subjects),
                "val_subjects": sorted(val_subjects),
                "test_subjects": sorted(test_subjects),
                "fold_index": fold_idx,
            }
        )
    return folds


def persist_splits(folds: list[dict[str, list[str]]], output_dir: Path) -> None:
    """Write fold assignments to JSON files.

    Args:
        folds: Fold definitions from compute_group_kfold_splits.
        output_dir: Directory for fold_{k}.json files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        path = output_dir / f"fold_{fold['fold_index']}.json"
        save_json(path, fold)
        logger.info("Wrote split %s", path)


def load_split(path: Path) -> dict[str, list[str]]:
    """Load a persisted fold split.

    Args:
        path: Path to fold JSON.

    Returns:
        Fold dictionary.
    """
    result = load_json(path)
    assert isinstance(result, dict)
    return result


def assign_recordings_to_split(
    manifest: pd.DataFrame,
    fold: dict[str, list[str]],
) -> dict[str, list[int]]:
    """Map manifest row indices to train/val/test by subject.

    Args:
        manifest: DataFrame with subject_id column.
        fold: Fold dict with train/val/test subject lists.

    Returns:
        Dict mapping split name to list of manifest indices.
    """
    train_set = set(fold["train_subjects"])
    val_set = set(fold["val_subjects"])
    test_set = set(fold["test_subjects"])

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, row in manifest.iterrows():
        sid = row["subject_id"]
        if sid in train_set:
            splits["train"].append(int(idx))
        elif sid in val_set:
            splits["val"].append(int(idx))
        elif sid in test_set:
            splits["test"].append(int(idx))
        else:
            raise ValueError(f"Subject {sid} not in any split for this fold")
    return splits


def verify_no_subject_leakage(fold: dict[str, list[str]]) -> bool:
    """Check that no subject appears in more than one split.

    Args:
        fold: Fold with train/val/test subject lists.

    Returns:
        True if no leakage detected.

    Raises:
        AssertionError: If any subject appears in multiple splits.
    """
    train = set(fold["train_subjects"])
    val = set(fold["val_subjects"])
    test = set(fold["test_subjects"])
    assert train.isdisjoint(val), f"Leak train/val: {train & val}"
    assert train.isdisjoint(test), f"Leak train/test: {train & test}"
    assert val.isdisjoint(test), f"Leak val/test: {val & test}"
    return True
