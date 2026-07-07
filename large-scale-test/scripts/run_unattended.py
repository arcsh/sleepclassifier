#!/usr/bin/env python3
"""Unattended Sleep-EDF pipeline: preflight → data → sanity gate → background sweep."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))

from sleepstage.evaluation.results_report import CHANCE_F1, write_results_md
from sleepstage.utils.logging import setup_logging

logger = logging.getLogger(__name__)

MIN_DISK_GB = 10
SANITY_F1_MIN = 0.30  # meaningfully above ~0.2 chance floor
N1_PCT_LO, N1_PCT_HI = 3.0, 15.0
N2_PCT_LO, N2_PCT_HI = 30.0, 55.0


def disk_ok() -> tuple[bool, float]:
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / (1024**3)
    return free_gb >= MIN_DISK_GB, free_gb


def check_class_distribution(manifest_path: Path) -> tuple[bool, str]:
    df = pd.read_csv(manifest_path)
    count_cols = [c for c in df.columns if c.startswith("count_")]
    if not count_cols:
        return False, "manifest missing count_* columns — re-run preprocess on real data"
    totals = {c.replace("count_", ""): df[c].sum() for c in count_cols}
    total_epochs = sum(totals.values())
    if total_epochs < 1000:
        return (
            False,
            f"only {total_epochs} epochs — looks like synthetic demo data, not Sleep-EDF SC",
        )
    pcts = {k: 100.0 * v / total_epochs for k, v in totals.items()}
    lines = [f"total_epochs={total_epochs}"] + [f"{k}={pcts[k]:.2f}%" for k in sorted(pcts)]
    report = "\n".join(lines)
    ok = True
    reasons = []
    if not (N1_PCT_LO <= pcts.get("N1", 0) <= N1_PCT_HI):
        ok = False
        reasons.append(f"N1 {pcts.get('N1', 0):.2f}% outside {N1_PCT_LO}-{N1_PCT_HI}%")
    if not (N2_PCT_LO <= pcts.get("N2", 0) <= N2_PCT_HI):
        ok = False
        reasons.append(f"N2 {pcts.get('N2', 0):.2f}% outside {N2_PCT_LO}-{N2_PCT_HI}%")
    if reasons:
        report += "\nFAIL: " + "; ".join(reasons)
    else:
        report += "\nPASS"
    return ok, report


def run_sanity_gate() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "model=cnn1d",
        "n_folds=5",
        "training.n_seeds=1",
        "fold_only=0",
        "force_seed_subdir=true",
        "seed=42",
        "experiment_name=sanity_cnn1d_sleep_edf_sc",
        "output_dir=experiments/sanity_cnn1d_sleep_edf_sc",
        "training.accelerator=cpu",
        "training.num_workers=0",
        "training.precision=32-true",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    metrics_path = (
        ROOT / "experiments" / "sanity_cnn1d_sleep_edf_sc" / "seed_42" / "fold_0" / "metrics.json"
    )
    if proc.returncode != 0 or not metrics_path.exists():
        return False, f"sanity train failed (rc={proc.returncode})\n{out[-2000:]}"
    m = json.loads(metrics_path.read_text())
    f1 = m["macro_f1"]
    report = f"macro_f1={f1:.4f} (floor={CHANCE_F1}, min_pass={SANITY_F1_MIN})"
    if f1 < SANITY_F1_MIN:
        report += f"\nFAIL: macro-F1 {f1:.4f} < {SANITY_F1_MIN}"
        return False, report
    report += "\nPASS"
    return True, report


def main() -> None:
    setup_logging()
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    ok, free_gb = disk_ok()
    logger.info("Disk free: %.1f GB (need %d GB)", free_gb, MIN_DISK_GB)
    if not ok:
        (logs / "preflight.log").write_text(f"FAIL: only {free_gb:.1f} GB free\n")
        write_results_md(ROOT / "RESULTS.md", ROOT / "experiments")
        raise SystemExit(1)

    write_results_md(ROOT / "RESULTS.md", ROOT / "experiments")

    logger.info("Stage 1: download SC subset")
    for attempt in range(1, 4):
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "download_data.py"), "--subset", "sc"],
            cwd=ROOT,
        ).returncode
        if rc == 0:
            break
        msg = f"download attempt {attempt} failed rc={rc}\n"
        with (logs / "download.log").open("a" if attempt > 1 else "w") as f:
            f.write(msg)
        if attempt == 3:
            write_results_md(
                ROOT / "RESULTS.md",
                ROOT / "experiments",
                extra_failures=[
                    {"model": "pipeline", "reason": "download failed after 3 attempts"}
                ],
            )
            raise SystemExit(rc)
        time.sleep(60 * attempt)

    logger.info("Stage 1: preprocess")
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preprocess.py")], cwd=ROOT
    ).returncode
    if rc != 0:
        (logs / "preprocess.log").write_text(f"preprocess failed rc={rc}\n")
        raise SystemExit(rc)

    manifest = ROOT / "data" / "processed" / "manifest.csv"
    dist_ok, dist_report = check_class_distribution(manifest)
    (logs / "preprocess_check.log").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n{dist_report}\n"
    )
    if not dist_ok:
        write_results_md(
            ROOT / "RESULTS.md",
            ROOT / "experiments",
            extra_failures=[
                {"model": "pipeline", "reason": f"preprocess check failed: {dist_report}"}
            ],
        )
        logger.error("Preprocess check failed — stopping.\n%s", dist_report)
        raise SystemExit(1)

    logger.info("Stage 2: sanity gate (cnn1d, 1 fold)")
    gate_ok, gate_report = run_sanity_gate()
    (logs / "sanity_gate.log").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n{gate_report}\n"
    )
    if not gate_ok:
        write_results_md(
            ROOT / "RESULTS.md",
            ROOT / "experiments",
            extra_failures=[{"model": "sanity_gate", "reason": gate_report}],
        )
        logger.error("Sanity gate failed — stopping.\n%s", gate_report)
        raise SystemExit(1)

    logger.info("Stage 3: launch background full sweep")
    sweep_log = logs / "full_run.log"
    with sweep_log.open("w") as logf:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "large-scale-test" / "scripts" / "run_full_sweep.py")],
            cwd=ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        )
    (logs / "sweep_pid.txt").write_text(str(proc.pid))
    logger.info("Sweep PID %d — log: %s", proc.pid, sweep_log)
    write_results_md(ROOT / "RESULTS.md", ROOT / "experiments")


if __name__ == "__main__":
    main()
