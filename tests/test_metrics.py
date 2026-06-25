"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from sleepstage.evaluation.metrics import compute_metrics, transition_matrix
from sleepstage.training.losses import compute_class_weights, compute_class_weights_from_counts


def test_compute_metrics_matches_sklearn() -> None:
    y_true = [0, 1, 2, 2, 3, 4, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 3, 4, 0, 2, 2, 3]
    m = compute_metrics(y_true, y_pred)
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"), rel=1e-5)
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y_true, y_pred), rel=1e-5)


def test_transition_matrix_rows_sum_to_one() -> None:
    labels = [0, 1, 2, 2, 3, 4, 0, 1, 2, 3]
    tm = transition_matrix(labels)
    row_sums = tm.sum(axis=1)
    for rs in row_sums:
        if rs > 0:
            assert rs == pytest.approx(1.0, abs=1e-6)


def test_class_weights_from_counts_matches_labels() -> None:
    import torch

    labels = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4])
    from_counts = compute_class_weights_from_counts(torch.bincount(labels, minlength=5).float())
    from_labels = compute_class_weights(labels)
    assert from_counts == pytest.approx(from_labels, rel=1e-5)
