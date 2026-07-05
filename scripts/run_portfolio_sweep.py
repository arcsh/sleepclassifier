#!/usr/bin/env python3
"""Portfolio training sweep — CNN1D + AttnSleep on laptop-scale compute budget."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleepstage.evaluation.results_report import (  # noqa: E402
    write_portfolio_results_md,
)
from sleepstage.utils.logging import setup_logging

logger = logging.getLogger(__name__)

SEED = 42
ERRORS_LOG = ROOT / "logs" / "portfolio_errors.log"
STATUS_PATH = ROOT / "experiments" / "portfolio_run_status.json"
RESULTS_PATH = ROOT / "RESULTS.md"
MAX_ATTEMPTS = 3

PORTFOLIO_COMBOS: list[tuple[str, str, list[int]]] = [
    ("cnn1d", "poc_cnn1d_sleep_edf_sc", [0, 1]),
    ("attnsleep", "poc_attnsleep_sleep_edf_sc", [0]),
]


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {
        "started": datetime.now(timezone.utc).isoformat(),
        "combos": [],
    }


def save_status(status: dict) -> None:
    status["updated"] = datetime.now(timezone.utc).isoformat()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def update_combo_status(status: dict, entry: dict) -> None:
    key = (entry["model"], entry["seed"], entry["fold"])
    status["combos"] = [c for c in status["combos"] if (c["model"], c["seed"], c["fold"]) != key]
    status["combos"].append(entry)
    save_status(status)


def log_error(model: str, seed: int, fold: int, tb: str, retry_note: str = "") -> None:
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_LOG.open("a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"{datetime.now(timezone.utc).isoformat()} {model} seed={seed} fold={fold}")
        if retry_note:
            f.write(f" [{retry_note}]")
        f.write("\n")
        f.write(tb)
        f.write("\n")


def classify_failure(output: str, returncode: int) -> str:
    if returncode < 0 or returncode in (139, -11):
        return "segfault"
    low = output.lower()
    if "mps" in low or "not implemented for" in low or "mps backend" in low:
        return "mps"
    if "out of memory" in low or "oom" in low or ("allocat" in low and "memory" in low):
        return "oom"
    return "other"


def build_train_cmd(
    model: str,
    exp_name: str,
    seed: int,
    fold: int,
    *,
    accelerator: str | None = None,
    batch_size: int | None = None,
) -> list[str]:
    accel = accelerator or "auto"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        f"model={model}",
        "training=portfolio",
        f"seed={seed}",
        f"fold_only={fold}",
        "training.n_seeds=1",
        "force_seed_subdir=true",
        f"experiment_name={exp_name}",
        f"output_dir=experiments/{exp_name}",
        f"training.accelerator={accel}",
        "training.precision=32-true",
        "training.num_workers=0",
    ]
    if batch_size is not None:
        cmd.append(f"training.batch_size={batch_size}")
    return cmd


def run_subprocess(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def run_combo(
    model: str,
    exp_name: str,
    seed: int,
    fold: int,
    default_batch: int = 64,
) -> tuple[bool, str, str]:
    """Run one combo with up to MAX_ATTEMPTS; MPS/OOM get cpu / half-batch retries."""
    strategies: list[tuple[str | None, int | None]] = [
        (None, None),
        ("cpu", None),
        ("cpu", max(8, default_batch // 2)),
    ]
    last_cause, last_reason = "other", ""

    for attempt, (accel, batch) in enumerate(strategies[:MAX_ATTEMPTS]):
        cmd = build_train_cmd(model, exp_name, seed, fold, accelerator=accel, batch_size=batch)
        returncode, out = run_subprocess(cmd)
        if returncode == 0:
            note = ""
            if attempt == 1:
                note = "recovered via cpu fallback"
            elif attempt == 2:
                note = f"recovered via cpu + batch_size={batch}"
            return True, "ok" if attempt == 0 else classify_failure(out, returncode), note

        cause = classify_failure(out, returncode)
        last_cause, last_reason = cause, out[-500:]
        note = f"attempt_{attempt + 1}"
        if attempt == 0 and cause == "oom":
            strategies[1] = (None, max(8, default_batch // 2))
        log_error(model, seed, fold, out, retry_note=note)
        logger.warning(
            "%s seed=%d fold=%d attempt %d failed (%s)",
            model,
            seed,
            fold,
            attempt + 1,
            cause,
        )

    return False, last_cause, last_reason


def combo_done(exp_name: str, seed: int, fold: int) -> bool:
    p = ROOT / "experiments" / exp_name / f"seed_{seed}" / f"fold_{fold}" / "metrics.json"
    return p.exists()


def refresh_aggregates(exp_name: str) -> None:
    """Rebuild summary.csv + aggregate.json from completed folds."""
    import pandas as pd

    exp_dir = ROOT / "experiments" / exp_name
    rows = []
    for mp in sorted(exp_dir.glob("seed_*/fold_*/metrics.json")):
        m = json.loads(mp.read_text())
        seed_m = re.search(r"seed_(\d+)", str(mp))
        fold_m = re.search(r"fold_(\d+)", str(mp))
        if seed_m and fold_m:
            m["seed"] = int(seed_m.group(1))
            m["fold"] = int(fold_m.group(1))
        rows.append(m)
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(exp_dir / "summary.csv", index=False)

    def _safe_std(series: pd.Series) -> float:
        val = series.std()
        return 0.0 if pd.isna(val) else float(val)

    agg = {
        "macro_f1_mean": float(summary["macro_f1"].mean()),
        "macro_f1_std": _safe_std(summary["macro_f1"]),
        "kappa_mean": float(summary["kappa"].mean()),
        "kappa_std": _safe_std(summary["kappa"]),
        "accuracy_mean": float(summary["accuracy"].mean()),
        "accuracy_std": _safe_std(summary["accuracy"]),
        "n_seeds": len(summary["seed"].unique()),
        "n_folds": len(summary["fold"].unique()),
    }
    (exp_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))


def portfolio_combos(overnight: bool) -> list[tuple[str, str, list[int]]]:
    combos = []
    for model, exp_name, folds in PORTFOLIO_COMBOS:
        if model == "cnn1d" and overnight:
            folds = list(range(5))
        combos.append((model, exp_name, folds))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio training sweep")
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Extend CNN1D to all 5 folds (adds ~12-16 h)",
    )
    args = parser.parse_args()

    setup_logging()
    ROOT.joinpath("logs").mkdir(exist_ok=True)
    status = load_status()
    combos = portfolio_combos(args.overnight)

    rf_agg = ROOT / "experiments" / "rf_baseline_sleep_edf_sc" / "aggregate.json"
    if rf_agg.exists():
        logger.info("Reusing RF baseline at %s", rf_agg)
    else:
        logger.warning("RF baseline aggregate missing — expected pre-computed")

    write_portfolio_results_md(RESULTS_PATH, ROOT / "experiments", combos=combos)

    for model, exp_name, folds in combos:
        for fold in folds:
            if combo_done(exp_name, SEED, fold):
                logger.info("Skip done: %s seed=%d fold=%d", exp_name, SEED, fold)
                continue

            logger.info("Running %s seed=%d fold=%d", exp_name, SEED, fold)
            try:
                ok, cause, reason = run_combo(model, exp_name, SEED, fold)
            except Exception:
                tb = traceback.format_exc()
                log_error(model, SEED, fold, tb)
                ok, cause, reason = False, "other", tb[-500:]

            entry = {
                "model": model,
                "experiment": exp_name,
                "seed": SEED,
                "fold": fold,
                "status": "success" if ok else "skipped",
                "root_cause": cause,
                "reason": reason,
                "finished": datetime.now(timezone.utc).isoformat(),
            }
            update_combo_status(status, entry)
            if ok:
                refresh_aggregates(exp_name)
            else:
                logger.error(
                    "Skipped %s seed=%d fold=%d after %d attempts: %s",
                    exp_name,
                    SEED,
                    fold,
                    MAX_ATTEMPTS,
                    reason[:200],
                )

    write_portfolio_results_md(RESULTS_PATH, ROOT / "experiments", combos=combos)
    _run_figures()
    logger.info("Portfolio sweep complete.")


def _run_figures() -> None:
    logger.info("Running make_figures.py on available checkpoints...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py")],
        cwd=ROOT,
        check=False,
    )


if __name__ == "__main__":
    main()
