"""Generate RESULTS.md from experiment outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleepstage.constants import STAGE_NAMES

MODELS = ["rf_baseline", "cnn1d", "cnn_bilstm", "attnsleep"]
DATA_NAME = "sleep_edf_sc"
SANITY_F1_LO = 0.75
SANITY_F1_HI = 0.82
CHANCE_F1 = 0.2

# ponytail: portfolio scope labels — honest CV coverage per model
PORTFOLIO_MODELS: list[tuple[str, str, str, list[int], int]] = [
    ("rf_baseline", "rf_baseline_sleep_edf_sc", "5-fold × 3 seeds", [42, 43, 44], 5),
    ("cnn1d", "poc_cnn1d_sleep_edf_sc", "2-fold, seed 42", [42], 2),
    ("attnsleep", "poc_attnsleep_sleep_edf_sc", "1-fold, seed 42", [42], 1),
]


def experiment_dir(experiments_root: Path, model: str) -> Path:
    """Return experiment directory for a model on Sleep-EDF SC."""
    return experiments_root / f"{model}_{DATA_NAME}"


def collect_combo_metrics(
    experiments_root: Path,
    models: list[str] | None = None,
    n_folds: int = 5,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect per-combo metrics and failures from disk."""
    models = models or MODELS
    seeds = seeds or [42, 43, 44]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_status_path = experiments_root / "run_status.json"
    if run_status_path.exists():
        status = json.loads(run_status_path.read_text())
        for entry in status.get("combos", []):
            if entry.get("status") == "failed":
                failures.append(entry)

    for model in models:
        exp_dir = experiment_dir(experiments_root, model)
        for seed in seeds:
            for fold in range(n_folds):
                metrics_path = exp_dir / f"seed_{seed}" / f"fold_{fold}" / "metrics.json"
                if not metrics_path.exists():
                    continue
                m = json.loads(metrics_path.read_text())
                per_class_f1 = [m["per_class"][s]["f1-score"] for s in STAGE_NAMES]
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "macro_f1": m["macro_f1"],
                        "kappa": m["kappa"],
                        "accuracy": m["accuracy"],
                        "n1_f1": m["per_class"]["N1"]["f1-score"],
                        "weakest_class": STAGE_NAMES[int(np.argmin(per_class_f1))],
                    }
                )
    return pd.DataFrame(rows), failures, []


def aggregate_by_model(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-combo metrics to per-model mean ± std."""
    if summary.empty:
        return summary
    agg = (
        summary.groupby("model")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            n_combos=("macro_f1", "count"),
            n1_f1_mean=("n1_f1", "mean"),
        )
        .reset_index()
    )
    agg["macro_f1_std"] = agg["macro_f1_std"].fillna(0.0)
    agg["kappa_std"] = agg["kappa_std"].fillna(0.0)
    return agg


def determine_status(
    summary: pd.DataFrame,
    failures: list[dict[str, Any]],
    expected_combos: int,
    circuit_breaker: str | None = None,
) -> str:
    """Map completion counts to SUCCESS / PARTIAL / FAILED."""
    if circuit_breaker:
        return "FAILED"
    completed = len(summary)
    if completed == 0 and failures:
        return "FAILED"
    if completed >= expected_combos and not failures:
        return "SUCCESS"
    if completed > 0:
        return "PARTIAL"
    return "FAILED"


def _load_auto_fixes(experiments_root: Path) -> list[str]:
    fixes_path = experiments_root.parent / "logs" / "watchdog_fixes.json"
    if fixes_path.exists():
        return json.loads(fixes_path.read_text())
    return []


def format_status_line(status: str, auto_fixes: list[str] | None) -> str:
    """Format RESULTS.md status header."""
    if auto_fixes:
        return f"STATUS: {status} ({'; '.join(auto_fixes)})"
    return f"STATUS: {status}"


def write_results_md(
    path: Path,
    experiments_root: Path,
    *,
    models: list[str] | None = None,
    n_folds: int = 5,
    seeds: list[int] | None = None,
    circuit_breaker: str | None = None,
    extra_failures: list[dict[str, Any]] | None = None,
    auto_fixes: list[str] | None = None,
    watchdog_diagnosis: str | None = None,
) -> str:
    """Write RESULTS.md and return STATUS string."""
    models = models or MODELS
    seeds = seeds or [42, 43, 44]
    expected = len(models) * n_folds * len(seeds)
    summary, status_failures, _ = collect_combo_metrics(experiments_root, models, n_folds, seeds)
    failures = status_failures + (extra_failures or [])
    agg = aggregate_by_model(summary)
    status = determine_status(summary, failures, expected, circuit_breaker)
    fixes = auto_fixes if auto_fixes is not None else _load_auto_fixes(experiments_root)

    lines: list[str] = []
    if watchdog_diagnosis:
        lines.extend(
            [
                "## Watchdog diagnosis",
                "",
                watchdog_diagnosis,
                "",
            ]
        )
    lines.extend(
        [
            format_status_line(status, fixes),
            "",
            f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
        ]
    )
    if circuit_breaker:
        lines.extend(
            [
                "## Circuit breaker",
                "",
                circuit_breaker,
                "",
            ]
        )

    lines.extend(
        [
            "## Results (mean ± std macro-F1 / κ)",
            "",
            f"Completed combos: {len(summary)} / {expected}",
            "",
            "| Model | macro-F1 | κ | N1 F1 (mean) | combos |",
            "|-------|----------|---|--------------|--------|",
        ]
    )
    if agg.empty:
        lines.append("| _none yet_ | — | — | — | 0 |")
    else:
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['model']} | "
                f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | "
                f"{row['kappa_mean']:.3f} ± {row['kappa_std']:.3f} | "
                f"{row['n1_f1_mean']:.3f} | {int(row['n_combos'])} |"
            )

    lines.extend(
        [
            "",
            "## Sanity band",
            "",
            f"Expected macro-F1 on Sleep-EDF SC: **{SANITY_F1_LO:.2f}–{SANITY_F1_HI:.2f}** "
            f"(chance floor ≈ {CHANCE_F1:.2f}).",
            "",
        ]
    )
    if not agg.empty:
        in_band = agg[
            (agg["macro_f1_mean"] >= SANITY_F1_LO) & (agg["macro_f1_mean"] <= SANITY_F1_HI)
        ]
        if len(in_band):
            lines.append(f"Models in sanity band: {', '.join(in_band['model'].tolist())}.")
        else:
            best = agg.loc[agg["macro_f1_mean"].idxmax()]
            lines.append(
                f"No model in band yet. Best so far: **{best['model']}** "
                f"({best['macro_f1_mean']:.3f} macro-F1)."
            )
        n1_weakest = (agg["n1_f1_mean"] == agg[["n1_f1_mean"]].min().iloc[0]).any()
        lines.append(
            "N1 weakest class (expected): "
            + (
                "yes (lowest mean per-class F1 among headline classes)"
                if n1_weakest
                else "not confirmed yet"
            )
        )
    else:
        lines.append("_No completed runs yet._")

    if failures:
        lines.extend(["", "## Failures", ""])
        seen = set()
        for f in failures:
            key = (f.get("model"), f.get("seed"), f.get("fold"), f.get("reason", "")[:80])
            if key in seen:
                continue
            seen.add(key)
            combo = f"{f.get('model', '?')} seed={f.get('seed', '?')} fold={f.get('fold', '?')}"
            lines.append(f"- **{combo}**: {f.get('reason', 'unknown')}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return status


def _cnn1d_scope_label(n_folds: int) -> str:
    if n_folds >= 5:
        return "5-fold, seed 42"
    return f"{n_folds}-fold, seed 42"


def collect_portfolio_metrics(
    experiments_root: Path,
    combos: list[tuple[str, str, list[int]]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    """Collect metrics for portfolio experiments with per-model scope labels."""
    if combos is None:
        combos = [
            ("cnn1d", "poc_cnn1d_sleep_edf_sc", [0, 1]),
            ("attnsleep", "poc_attnsleep_sleep_edf_sc", [0]),
        ]

    rows: list[dict[str, Any]] = []
    scope_labels: dict[str, str] = {"rf_baseline": "5-fold × 3 seeds"}
    failures: list[dict[str, Any]] = []

    status_path = experiments_root / "portfolio_run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        for entry in status.get("combos", []):
            if entry.get("status") in ("failed", "skipped"):
                failures.append(entry)

    # RF: full 3-seed CV
    rf_dir = experiments_root / "rf_baseline_sleep_edf_sc"
    for seed in [42, 43, 44]:
        for fold in range(5):
            metrics_path = rf_dir / f"seed_{seed}" / f"fold_{fold}" / "metrics.json"
            if not metrics_path.exists():
                continue
            m = json.loads(metrics_path.read_text())
            per_class_f1 = [m["per_class"][s]["f1-score"] for s in STAGE_NAMES]
            rows.append(
                {
                    "model": "rf_baseline",
                    "seed": seed,
                    "fold": fold,
                    "macro_f1": m["macro_f1"],
                    "kappa": m["kappa"],
                    "accuracy": m["accuracy"],
                    "n1_f1": m["per_class"]["N1"]["f1-score"],
                    "weakest_class": STAGE_NAMES[int(np.argmin(per_class_f1))],
                }
            )

    for model, exp_name, folds in combos:
        if model == "rf_baseline":
            continue
        n_done = 0
        for fold in folds:
            metrics_path = experiments_root / exp_name / "seed_42" / f"fold_{fold}" / "metrics.json"
            if not metrics_path.exists():
                continue
            n_done += 1
            m = json.loads(metrics_path.read_text())
            per_class_f1 = [m["per_class"][s]["f1-score"] for s in STAGE_NAMES]
            rows.append(
                {
                    "model": model,
                    "seed": 42,
                    "fold": fold,
                    "macro_f1": m["macro_f1"],
                    "kappa": m["kappa"],
                    "accuracy": m["accuracy"],
                    "n1_f1": m["per_class"]["N1"]["f1-score"],
                    "weakest_class": STAGE_NAMES[int(np.argmin(per_class_f1))],
                }
            )
        scope_labels[model] = (
            _cnn1d_scope_label(n_done) if model == "cnn1d" else f"{len(folds)}-fold, seed 42"
        )

    return pd.DataFrame(rows), failures, scope_labels


def portfolio_expected_combos(combos: list[tuple[str, str, list[int]]]) -> int:
    """Count expected portfolio combos (RF pre-done + DL folds)."""
    expected = 15  # RF 5-fold × 3 seeds
    for model, _, folds in combos:
        if model != "rf_baseline":
            expected += len(folds)
    return expected


def portfolio_complete(
    experiments_root: Path,
    combos: list[tuple[str, str, list[int]]],
) -> bool:
    """True when all planned DL folds have metrics.json."""
    rf_ok = (experiments_root / "rf_baseline_sleep_edf_sc" / "aggregate.json").exists()
    if not rf_ok:
        return False
    for model, exp_name, folds in combos:
        if model == "rf_baseline":
            continue
        for fold in folds:
            p = experiments_root / exp_name / "seed_42" / f"fold_{fold}" / "metrics.json"
            if not p.exists():
                return False
    return True


def write_portfolio_results_md(
    path: Path,
    experiments_root: Path,
    *,
    combos: list[tuple[str, str, list[int]]] | None = None,
    extra_failures: list[dict[str, Any]] | None = None,
) -> str:
    """Write portfolio RESULTS.md with scope column and honest completion status."""
    if combos is None:
        combos = [
            ("cnn1d", "poc_cnn1d_sleep_edf_sc", [0, 1]),
            ("attnsleep", "poc_attnsleep_sleep_edf_sc", [0]),
        ]

    summary, status_failures, scope_labels = collect_portfolio_metrics(experiments_root, combos)
    failures = status_failures + (extra_failures or [])
    expected = portfolio_expected_combos(combos)
    completed = len(summary)

    dl_folds_expected = sum(len(f) for m, _, f in combos if m != "rf_baseline")
    dl_folds_done = len(summary[summary["model"] != "rf_baseline"]) if not summary.empty else 0

    if portfolio_complete(experiments_root, combos):
        status = "COMPLETE"
    elif completed > 0:
        status = "PARTIAL"
    else:
        status = "IN PROGRESS — portfolio run pending"

    lines: list[str] = [
        f"STATUS: {status}",
        "",
        f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Results (mean ± std macro-F1 / κ)",
        "",
        f"Completed combos: {completed} / {expected} "
        f"(DL folds: {dl_folds_done}/{dl_folds_expected})",
        "",
        "| Model | Scope | macro-F1 | κ | N1 F1 (mean) | combos |",
        "|-------|-------|----------|---|--------------|--------|",
    ]

    if summary.empty:
        lines.append("| _none yet_ | — | — | — | — | 0 |")
    else:
        for model in ["rf_baseline", "cnn1d", "attnsleep"]:
            sub = summary[summary["model"] == model]
            if sub.empty:
                scope = scope_labels.get(model, "not started")
                lines.append(f"| {model} | {scope} | — | — | — | 0 |")
                continue
            scope = scope_labels.get(model, "—")
            lines.append(
                f"| {model} | {scope} | "
                f"{sub['macro_f1'].mean():.3f} ± {sub['macro_f1'].std() or 0.0:.3f} | "
                f"{sub['kappa'].mean():.3f} ± {sub['kappa'].std() or 0.0:.3f} | "
                f"{sub['n1_f1'].mean():.3f} | {len(sub)} |"
            )

    lines.extend(
        [
            "",
            "## Sanity band (reference)",
            "",
            f"Published Sleep-EDF SC macro-F1 is often **{SANITY_F1_LO:.2f}–{SANITY_F1_HI:.2f}** "
            f"(chance floor ≈ {CHANCE_F1:.2f}). Portfolio RF ~0.61; DL should beat RF clearly.",
            "",
            "Run: `python scripts/run_portfolio_sweep.py`",
            "",
        ]
    )

    if failures:
        lines.extend(["## Failures / skipped", ""])
        seen: set[tuple[Any, ...]] = set()
        for f in failures:
            key = (f.get("model"), f.get("seed"), f.get("fold"), str(f.get("reason", ""))[:80])
            if key in seen:
                continue
            seen.add(key)
            combo = (
                f"{f.get('experiment', f.get('model', '?'))} "
                f"seed={f.get('seed', '?')} fold={f.get('fold', '?')}"
            )
            lines.append(f"- **{combo}**: {f.get('reason', 'unknown')[:200]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return status
