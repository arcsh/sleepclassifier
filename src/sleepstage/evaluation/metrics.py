"""Evaluation metrics for sleep staging."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from sleepstage.constants import N_CLASSES, STAGE_NAMES


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    subjects: list[str] | None = None,
) -> dict[str, Any]:
    """Compute standard sleep-staging metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        subjects: Optional per-epoch subject IDs for per-subject breakdown.

    Returns:
        Dictionary with accuracy, macro_f1, kappa, per-class metrics, confusion matrix.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(N_CLASSES)),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(N_CLASSES)))

    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true_arr, y_pred_arr)),
        "per_class": {name: report[name] for name in STAGE_NAMES},
        "confusion_matrix": cm.tolist(),
    }

    if subjects is not None:
        per_subject: dict[str, float] = {}
        for sid in set(subjects):
            mask = np.array([s == sid for s in subjects])
            if mask.sum() == 0:
                continue
            per_subject[sid] = float(
                f1_score(y_true_arr[mask], y_pred_arr[mask], average="macro", zero_division=0)
            )
        result["per_subject_macro_f1"] = per_subject

    return result


def transition_matrix(labels: list[int] | np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Compute stage transition probability matrix P(s_{t+1} | s_t).

    Args:
        labels: Sequence of stage labels.
        n_classes: Number of classes.

    Returns:
        Row-normalized transition matrix.
    """
    labels_arr = np.asarray(labels)
    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(len(labels_arr) - 1):
        counts[labels_arr[i], labels_arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def aggregate_confusion_matrices(matrices: list[list[list[int]]]) -> np.ndarray:
    """Sum confusion matrices across folds.

    Args:
        matrices: List of confusion matrix lists.

    Returns:
        Aggregated confusion matrix.
    """
    return np.sum(np.array(matrices), axis=0)
