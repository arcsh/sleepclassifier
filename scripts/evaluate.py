#!/usr/bin/env python3
"""Evaluate trained models and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sleepstage.evaluation.compare import paired_wilcoxon
from sleepstage.evaluation.metrics import aggregate_confusion_matrices
from sleepstage.utils.logging import setup_logging
from sleepstage.visualization.confusion import plot_confusion_matrix

logger = logging.getLogger(__name__)


def aggregate_experiment(exp_dir: Path) -> dict:
    """Aggregate per-fold metrics from an experiment directory."""
    rows = []
    cms = []
    for fold_dir in sorted(exp_dir.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        m["fold"] = int(fold_dir.name.split("_")[1])
        rows.append(m)
        cms.append(m["confusion_matrix"])

    if not rows:
        raise FileNotFoundError(f"No fold metrics in {exp_dir}")

    df = pd.DataFrame(rows)
    agg_cm = aggregate_confusion_matrices(cms)
    return {
        "summary": df,
        "macro_f1_mean": df["macro_f1"].mean(),
        "macro_f1_std": df["macro_f1"].std(),
        "kappa_mean": df["kappa"].mean(),
        "kappa_std": df["kappa"].std(),
        "confusion_matrix": agg_cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument("--experiments", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    setup_logging()

    results = {}
    for exp_dir in args.experiments:
        name = exp_dir.name
        agg = aggregate_experiment(exp_dir)
        results[name] = agg
        plot_confusion_matrix(
            np.array(agg["confusion_matrix"]),
            args.output_dir / "figures" / f"confusion_{name}.png",
            title=f"Confusion Matrix: {name}",
        )
        logger.info(
            "%s: macro-F1=%.3f±%.3f, κ=%.3f±%.3f",
            name,
            agg["macro_f1_mean"],
            agg["macro_f1_std"],
            agg["kappa_mean"],
            agg["kappa_std"],
        )

    # Paired comparison if exactly two experiments
    names = list(results.keys())
    if len(names) == 2:
        a, b = names
        test = paired_wilcoxon(
            results[a]["summary"]["macro_f1"].tolist(),
            results[b]["summary"]["macro_f1"].tolist(),
        )
        logger.info("Wilcoxon %s vs %s: p=%.4f", a, b, test["p_value"])

    out_path = args.output_dir / "evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: {
            "macro_f1_mean": v["macro_f1_mean"],
            "macro_f1_std": v["macro_f1_std"],
            "kappa_mean": v["kappa_mean"],
            "kappa_std": v["kappa_std"],
        }
        for k, v in results.items()
    }
    out_path.write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
