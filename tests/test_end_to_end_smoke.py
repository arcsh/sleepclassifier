"""Fast end-to-end smoke test (no real EDF download)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sleepstage.data.dataset import EpochDataset, make_dataloader
from sleepstage.evaluation.metrics import compute_metrics
from sleepstage.features.preprocessing import save_processed_npz
from sleepstage.features.splits import compute_group_kfold_splits, persist_splits
from sleepstage.models.cnn1d import CNN1D
from sleepstage.training.lightning_module import SleepStageLightningModule


@pytest.fixture
def mini_pipeline(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    subjects = [f"SC40{i}" for i in range(10)]
    rows = []
    for subj in subjects:
        epochs = np.random.randn(30, 1, 3000).astype(np.float32)
        labels = np.random.randint(0, 5, size=30)
        path = processed / f"{subj}_1.npz"
        save_processed_npz(path, epochs, labels, subj, "1", 100.0)
        rows.append({"subject_id": subj, "night_id": "1", "path": str(path), "n_epochs": 30})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(processed / "manifest.csv", index=False)
    folds = compute_group_kfold_splits(manifest["subject_id"].tolist(), n_folds=2)
    persist_splits(folds, processed / "splits")
    return processed


def test_end_to_end_smoke(mini_pipeline: Path, tmp_path: Path) -> None:
    manifest = pd.read_csv(mini_pipeline / "manifest.csv")
    fold = json.loads((mini_pipeline / "splits" / "fold_0.json").read_text())
    train_subjects = set(fold["train_subjects"])
    train_idx = [i for i, r in manifest.iterrows() if r["subject_id"] in train_subjects]
    val_idx = [i for i, r in manifest.iterrows() if r["subject_id"] in fold["val_subjects"]]

    train_ds = EpochDataset(manifest, train_idx)
    val_ds = EpochDataset(manifest, val_idx)
    train_loader = make_dataloader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = make_dataloader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = CNN1D()
    lit = SleepStageLightningModule(model=model, max_epochs=1)
    trainer = __import__("pytorch_lightning", fromlist=["Trainer"]).Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(lit, train_loader, val_loader)

    # Evaluate
    lit.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            preds.extend(lit(x).argmax(-1).tolist())
            targets.extend(y.tolist())
    metrics = compute_metrics(targets, preds)
    assert "macro_f1" in metrics
    assert "kappa" in metrics

    out = tmp_path / "metrics.json"
    out.write_text(json.dumps(metrics))
    assert out.exists()
