"""Statistical comparison between models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test on per-fold scores.

    Args:
        scores_a: Model A per-fold macro-F1 scores.
        scores_b: Model B per-fold macro-F1 scores.

    Returns:
        Dict with statistic, p-value, and mean difference.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    if len(a) != len(b):
        raise ValueError("Score lists must have equal length")
    stat, pval = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "mean_diff": float(a.mean() - b.mean()),
    }
