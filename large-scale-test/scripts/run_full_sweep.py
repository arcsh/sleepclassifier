#!/usr/bin/env python3
"""Fault-tolerant full model sweep — one subprocess per (model, fold, seed)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))

from sleepstage.evaluation.results_report import (  # noqa: E402
    MODELS,
    write_results_md,
)
from sleepstage.utils.logging import setup_logging

logger = logging.getLogger(__name__)

N_FOLDS = 5
SEEDS = [42, 43, 44]
ERRORS_LOG = ROOT / "logs" / "errors.log"
STATUS_PATH = ROOT / "experiments" / "run_status.json"
RESULTS_PATH = ROOT / "RESULTS.md"


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {
        "started": datetime.now(timezone.utc).isoformat(),
        "combos": [],
        "circuit_breaker": None,
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
    write_results_md(
        RESULTS_PATH,
        ROOT / "experiments",
        circuit_breaker=status.get("circuit_breaker"),
    )


def log_error(model: str, seed: int, fold: int, tb: str, retry_note: str = "") -> None:
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_LOG.open("a") as f:
        f.write(f"\n{'='*60}\n")
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
    if "out of memory" in low or "oom" in low or "allocat" in low and "memory" in low:
        return "oom"
    return "other"


def build_train_cmd(
    model: str,
    seed: int,
    fold: int,
    *,
    accelerator: str | None = None,
    batch_size: int | None = None,
) -> list[str]:
    exp_name = f"{model}_sleep_edf_sc"
    # ponytail: RF=sklearn CPU; DL auto→MPS on Apple Silicon (num_workers=0)
    accel = accelerator or ("cpu" if model == "rf_baseline" else "auto")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        f"model={model}",
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
    # ponytail: random_forest — lightgbm segfaults on M1 (see large-scale-test/mac-logs/error.md)
    if model == "rf_baseline":
        cmd.append("model.classifier=random_forest")
    if batch_size is not None:
        cmd.append(f"training.batch_size={batch_size}")
    return cmd


def run_combo(model: str, seed: int, fold: int, default_batch: int = 64) -> tuple[bool, str, str]:
    """Run one combo; retry once on MPS/OOM. Returns (ok, root_cause, reason)."""
    cmd = build_train_cmd(model, seed, fold)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return True, "ok", ""

    cause = classify_failure(out, proc.returncode)
    log_error(model, seed, fold, out)

    if cause == "mps":
        log_error(model, seed, fold, "retrying with accelerator=cpu\n", retry_note="mps_retry")
        cmd = build_train_cmd(model, seed, fold, accelerator="cpu")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out2 = proc.stdout + proc.stderr
        if proc.returncode == 0:
            return True, "mps", "recovered via cpu fallback"
        log_error(model, seed, fold, out2, retry_note="mps_retry_failed")
        return False, "mps", out2[-500:]

    if cause == "oom":
        half = max(8, default_batch // 2)
        log_error(model, seed, fold, f"retrying with batch_size={half}\n", retry_note="oom_retry")
        cmd = build_train_cmd(model, seed, fold, batch_size=half)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out2 = proc.stdout + proc.stderr
        if proc.returncode == 0:
            return True, "oom", f"recovered with batch_size={half}"
        log_error(model, seed, fold, out2, retry_note="oom_retry_failed")
        return False, "oom", out2[-500:]

    return False, cause, out[-500:]


def combo_done(model: str, seed: int, fold: int) -> bool:
    p = (
        ROOT
        / "experiments"
        / f"{model}_sleep_edf_sc"
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "metrics.json"
    )
    return p.exists()


def refresh_aggregates(model: str) -> None:
    """Rebuild summary.csv + aggregate.json for one model from completed folds."""
    import pandas as pd

    exp_dir = ROOT / "experiments" / f"{model}_sleep_edf_sc"
    rows = []
    for mp in sorted(exp_dir.glob("seed_*/fold_*/metrics.json")):
        m = json.loads(mp.read_text())
        seed = int(re.search(r"seed_(\d+)", str(mp)).group(1))  # type: ignore[union-attr]
        fold = int(re.search(r"fold_(\d+)", str(mp)).group(1))  # type: ignore[union-attr]
        m["seed"] = seed
        m["fold"] = fold
        rows.append(m)
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(exp_dir / "summary.csv", index=False)
    agg = {
        "macro_f1_mean": float(summary["macro_f1"].mean()),
        "macro_f1_std": float(summary["macro_f1"].std() or 0.0),
        "kappa_mean": float(summary["kappa"].mean()),
        "kappa_std": float(summary["kappa"].std() or 0.0),
        "accuracy_mean": float(summary["accuracy"].mean()),
        "accuracy_std": float(summary["accuracy"].std() or 0.0),
        "n_seeds": len(summary["seed"].unique()),
        "n_folds": len(summary["fold"].unique()),
    }
    (exp_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))


def main() -> None:
    setup_logging()
    ROOT.joinpath("logs").mkdir(exist_ok=True)
    status = load_status()
    write_results_md(
        RESULTS_PATH, ROOT / "experiments", circuit_breaker=status.get("circuit_breaker")
    )

    consecutive: list[str] = []
    for model in MODELS:
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                if combo_done(model, seed, fold):
                    logger.info("Skip done: %s seed=%d fold=%d", model, seed, fold)
                    continue

                logger.info("Running %s seed=%d fold=%d", model, seed, fold)
                try:
                    ok, cause, reason = run_combo(model, seed, fold)
                except Exception:
                    tb = traceback.format_exc()
                    log_error(model, seed, fold, tb)
                    ok, cause, reason = False, "other", tb[-500:]

                entry = {
                    "model": model,
                    "seed": seed,
                    "fold": fold,
                    "status": "success" if ok else "failed",
                    "root_cause": cause,
                    "reason": reason,
                    "finished": datetime.now(timezone.utc).isoformat(),
                }
                if not ok and cause in ("mps", "oom"):
                    entry["status"] = "retried_failed"
                update_combo_status(status, entry)
                if ok:
                    refresh_aggregates(model)
                    consecutive.clear()
                else:
                    consecutive.append(cause)
                    if len(consecutive) >= 3 and len(set(consecutive[-3:])) == 1:
                        msg = (
                            f"Stopped after 3 consecutive `{consecutive[-1]}` failures "
                            f"(last: {model} seed={seed} fold={fold})."
                        )
                        status["circuit_breaker"] = msg
                        save_status(status)
                        write_results_md(
                            RESULTS_PATH,
                            ROOT / "experiments",
                            circuit_breaker=msg,
                        )
                        logger.error(msg)
                        _run_figures()
                        raise SystemExit(1)

    write_results_md(RESULTS_PATH, ROOT / "experiments")
    _run_figures()
    logger.info("Full sweep complete.")


def _run_figures() -> None:
    logger.info("Running make_figures.py on available checkpoints...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py")],
        cwd=ROOT,
        check=False,
    )


if __name__ == "__main__":
    main()
