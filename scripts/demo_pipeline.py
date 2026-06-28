#!/usr/bin/env python3
"""CI-only synthetic smoke demo — does not touch real data/processed."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sleepstage.features.preprocessing import save_processed_npz
from sleepstage.features.splits import compute_group_kfold_splits, persist_splits
from sleepstage.utils.logging import setup_logging

logger = logging.getLogger(__name__)

MODELS = ["rf_baseline", "cnn1d", "cnn_bilstm", "attnsleep"]


def make_synthetic_dataset(processed_dir: Path, n_subjects: int = 12) -> Path:
    """Create synthetic processed data for smoke/demo runs."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_subjects):
        subj = f"SC4{i:02d}"
        for night in ("1", "2"):
            epochs = np.random.randn(40, 1, 3000).astype(np.float32)
            labels = np.random.randint(0, 5, size=40)
            path = processed_dir / f"{subj}_{night}.npz"
            save_processed_npz(path, epochs, labels, subj, night, 100.0)
            rows.append(
                {
                    "subject_id": subj,
                    "night_id": night,
                    "path": str(path),
                    "n_epochs": 40,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest_path = processed_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    folds = compute_group_kfold_splits(manifest["subject_id"].tolist(), n_folds=3)
    persist_splits(folds, processed_dir / "splits")
    return manifest_path


def main() -> None:
    setup_logging()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "demo_processed"
    make_synthetic_dataset(processed)
    logger.info("Synthetic manifest at %s", processed / "manifest.csv")

    for model in MODELS:
        cmd = [
            sys.executable,
            "scripts/train.py",
            f"model={model}",
            "n_folds=3",
            "data.processed_dir=data/demo_processed",
            "training.max_epochs=2",
            "training.batch_size=16",
            "training.patience=2",
            "training.n_seeds=1",
            f"experiment_name=demo_{model}",
            f"output_dir=experiments/demo_{model}",
        ]
        if model == "rf_baseline":
            cmd.append("model.classifier=random_forest")
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=root)

    logger.info("Demo pipeline complete (synthetic only).")


if __name__ == "__main__":
    main()
