"""Cross-validation training orchestration."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sleepstage.data.dataset import EpochDataset, SequenceEpochDataset, make_dataloader
from sleepstage.evaluation.metrics import compute_metrics, transition_matrix
from sleepstage.features.preprocessing import load_processed_npz
from sleepstage.features.splits import assign_recordings_to_split, load_split
from sleepstage.models.base import SleepStageModel
from sleepstage.models.baseline_rf import RandomForestBaseline
from sleepstage.training.lightning_module import SleepStageLightningModule
from sleepstage.utils.io import get_git_hash, save_json, save_yaml, snapshot_environment
from sleepstage.utils.seed import set_seed

logger = logging.getLogger(__name__)

PrecisionSetting = Literal[
    "64", "32", "16", "bf16", "32-true", "16-mixed", "bf16-mixed", "transformer-engine"
]


def build_model(cfg: DictConfig) -> SleepStageModel:
    """Instantiate model from Hydra config.

    Args:
        cfg: Full config with model group.

    Returns:
        Initialized model.
    """
    model_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True))
    target = str(model_cfg.pop("_target_"))
    module_path, class_name = target.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    params = {str(k): v for k, v in model_cfg.items() if k != "name"}
    return cls(**params)


def _collect_epochs(
    manifest: pd.DataFrame, indices: list[int]
) -> tuple[np.ndarray, np.ndarray, float]:
    xs, ys = [], []
    sfreq = 100.0
    for idx in indices:
        row = manifest.iloc[idx]
        data = load_processed_npz(Path(row["path"]))
        xs.append(data.epochs)
        ys.append(data.labels)
        sfreq = data.sample_rate
    return np.concatenate(xs), np.concatenate(ys), sfreq


def _resolve_precision(cfg: DictConfig) -> PrecisionSetting:
    """Pick Lightning precision string with CPU/MPS-safe fallback."""
    precision = str(cfg.training.get("precision", "auto"))
    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    elif precision in ("32", "32-true"):
        precision = "32-true"
    elif precision == "16-mixed" and not torch.cuda.is_available():
        precision = "32-true"
    return cast(PrecisionSetting, precision)


def _predict_night(
    lit: SleepStageLightningModule,
    epochs: np.ndarray,
    is_seq: bool,
    seq_len: int,
) -> list[int]:
    """Predict labels for every epoch in one night."""
    n_epochs = epochs.shape[0]
    preds = [-1] * n_epochs
    lit.eval()
    with torch.no_grad():
        if not is_seq:
            for i in range(n_epochs):
                x = torch.from_numpy(epochs[i : i + 1].astype(np.float32))
                preds[i] = int(lit(x).argmax(-1).item())
            return preds
        half = seq_len // 2
        for center in range(half, n_epochs - (seq_len - half - 1)):
            start = center - half
            window = epochs[start : start + seq_len]
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            logits = lit(x)
            preds[center] = int(logits.argmax(-1).item())
    return [p if p >= 0 else 0 for p in preds]


def _save_fold_artifacts(
    lit: SleepStageLightningModule,
    manifest: pd.DataFrame,
    test_indices: list[int],
    output_dir: Path,
    is_seq: bool,
    seq_len: int,
    model_name: str,
) -> None:
    """Save training history, hypnogram sample, embeddings, and attention weights."""
    save_json(output_dir / "training_history.json", lit.epoch_history)

    if not test_indices:
        return
    row = manifest.iloc[test_indices[0]]
    data = load_processed_npz(Path(row["path"]))
    preds = _predict_night(lit, data.epochs, is_seq, seq_len)
    save_json(
        output_dir / "hypnogram_sample.json",
        {
            "subject_id": row["subject_id"],
            "night_id": row["night_id"],
            "y_true": data.labels.tolist(),
            "y_pred": preds,
        },
    )
    save_json(
        output_dir / "transition_matrix_pred.json",
        {"matrix": transition_matrix(preds).tolist()},
    )

    n_embed = min(500, data.epochs.shape[0])
    embeds: list[list[float]] = []
    labels: list[int] = []
    with torch.no_grad():
        for i in range(n_embed):
            x = torch.from_numpy(data.epochs[i : i + 1].astype(np.float32))
            if is_seq:
                half = seq_len // 2
                if i < half or i >= data.epochs.shape[0] - (seq_len - half - 1):
                    continue
                start = i - half
                window = data.epochs[start : start + seq_len]
                x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            emb = lit.model.get_embeddings(x)
            embeds.append(emb[0].cpu().tolist())
            labels.append(int(data.labels[i]))
    if embeds:
        np.savez(
            output_dir / "embeddings_sample.npz",
            embeddings=np.array(embeds),
            labels=np.array(labels),
        )

    if model_name == "attnsleep" and is_seq:
        from sleepstage.models.attnsleep import AttnSleep

        if isinstance(lit.model, AttnSleep):
            half = seq_len // 2
            center = half
            if data.epochs.shape[0] > seq_len:
                window = data.epochs[:seq_len]
                x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                lit.model(x)
                weights = lit.model.get_attention_weights()
                if weights is not None:
                    w = weights[0, center].cpu().numpy()
                    save_json(output_dir / "attention_weights.json", {"weights": w.tolist()})


def train_baseline_fold(
    cfg: DictConfig,
    manifest: pd.DataFrame,
    fold_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train classical baseline for one fold.

    Args:
        cfg: Hydra config.
        manifest: Preprocessing manifest.
        fold_path: Path to fold JSON.
        output_dir: Experiment output directory.

    Returns:
        Test metrics dict.
    """
    fold = load_split(fold_path)
    splits = assign_recordings_to_split(manifest, fold)
    train_x, train_y, sfreq = _collect_epochs(manifest, splits["train"])
    test_x, test_y, _ = _collect_epochs(manifest, splits["test"])

    model = RandomForestBaseline(
        n_estimators=cfg.model.get("n_estimators", 200),
        classifier=cfg.model.get("classifier", "random_forest"),
    )
    model.fit(train_x, train_y, sfreq)
    preds = model.predict(test_x, sfreq)
    metrics = compute_metrics(test_y.tolist(), preds.tolist())
    save_json(output_dir / "metrics.json", metrics)
    shutil.copy2(fold_path, output_dir / f"split_{fold_path.stem}.json")
    if splits["test"]:
        row = manifest.iloc[splits["test"][0]]
        data = load_processed_npz(Path(row["path"]))
        night_preds = model.predict(data.epochs, sfreq).tolist()
        save_json(
            output_dir / "hypnogram_sample.json",
            {
                "subject_id": row["subject_id"],
                "night_id": row["night_id"],
                "y_true": data.labels.tolist(),
                "y_pred": night_preds,
            },
        )
    return metrics


def train_dl_fold(
    cfg: DictConfig,
    manifest: pd.DataFrame,
    fold_path: Path,
    output_dir: Path,
    fold_idx: int,
) -> dict[str, Any]:
    """Train deep learning model for one fold.

    Args:
        cfg: Hydra config.
        manifest: Manifest DataFrame.
        fold_path: Fold split JSON.
        output_dir: Output directory for this fold.
        fold_idx: Fold index.

    Returns:
        Test metrics dictionary.
    """
    set_seed(cfg.seed + fold_idx)
    fold = load_split(fold_path)
    splits = assign_recordings_to_split(manifest, fold)

    seq_len = int(cfg.training.get("sequence_length", 20))
    model = build_model(cfg)
    is_seq = getattr(model, "is_sequence_model", False)

    train_ds: EpochDataset | SequenceEpochDataset
    val_ds: EpochDataset | SequenceEpochDataset
    test_ds: EpochDataset | SequenceEpochDataset
    if is_seq:
        train_ds = SequenceEpochDataset(manifest, splits["train"], seq_len)
        val_ds = SequenceEpochDataset(manifest, splits["val"], seq_len)
        test_ds = SequenceEpochDataset(manifest, splits["test"], seq_len)
    else:
        train_ds = EpochDataset(manifest, splits["train"])
        val_ds = EpochDataset(manifest, splits["val"])
        test_ds = EpochDataset(manifest, splits["test"])

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.get("num_workers", 0))
    train_loader = make_dataloader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_dataloader(val_ds, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_dataloader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    class_weights = SleepStageLightningModule.class_weights_from_manifest(manifest, splits["train"])
    lit = SleepStageLightningModule(
        model=model,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        loss_name=str(cfg.training.loss),
        focal_gamma=float(cfg.training.get("focal_gamma", 2.0)),
        scheduler_name=str(cfg.training.scheduler),
        max_epochs=int(cfg.training.max_epochs),
        class_weights=class_weights,
    )

    precision = _resolve_precision(cfg)

    accelerator = str(cfg.training.get("accelerator", "auto"))
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    elif accelerator == "gpu":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="best",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_macro_f1", patience=int(cfg.training.patience), mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb")

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=float(cfg.training.get("gradient_clip_val", 5.0)),
        callbacks=[checkpoint_cb, early_stop, lr_monitor],
        enable_progress_bar=True,
        logger=tb_logger,
        default_root_dir=str(output_dir),
    )
    trainer.fit(lit, train_loader, val_loader)
    shutil.copy2(fold_path, output_dir / f"split_{fold_path.stem}.json")

    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])

    # Evaluate on test
    lit.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_subjects: list[str] = []
    with torch.no_grad():
        for x, y, subjects, _ in test_loader:
            logits = lit(x)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            all_subjects.extend(subjects)

    metrics = compute_metrics(all_targets, all_preds, subjects=all_subjects)
    save_json(output_dir / "metrics.json", metrics)
    if fold_idx == 0:
        _save_fold_artifacts(
            lit,
            manifest,
            splits["test"],
            output_dir,
            is_seq,
            seq_len,
            str(cfg.model.name),
        )
    return metrics


def run_cross_validation(cfg: DictConfig, manifest_path: Path) -> pd.DataFrame:
    """Run full k-fold CV and aggregate results.

    Args:
        cfg: Hydra config.
        manifest_path: Path to manifest CSV.

    Returns:
        Summary DataFrame with per-fold metrics.
    """
    manifest = pd.read_csv(manifest_path)
    splits_dir = manifest_path.parent / "splits"
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(
        output_dir / "config.yaml", cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    )
    save_json(output_dir / "git_hash.json", {"hash": get_git_hash()})
    snapshot_environment(output_dir / "environment.txt")

    model_name = str(cfg.model.name)
    is_baseline = model_name == "rf_baseline"
    n_folds = int(cfg.n_folds)
    n_seeds = int(cfg.training.get("n_seeds", 1))
    fold_only = cfg.get("fold_only")
    fold_indices = [int(fold_only)] if fold_only is not None else list(range(n_folds))
    use_seed_dir = n_seeds > 1 or bool(cfg.get("force_seed_subdir", False))
    rows: list[dict[str, Any]] = []

    for seed_i in range(n_seeds):
        run_seed = int(cfg.seed) + seed_i
        set_seed(run_seed)
        for fold_idx in fold_indices:
            fold_path = splits_dir / f"fold_{fold_idx}.json"
            if use_seed_dir:
                fold_dir = output_dir / f"seed_{run_seed}" / f"fold_{fold_idx}"
            else:
                fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if is_baseline:
                metrics = train_baseline_fold(cfg, manifest, fold_path, fold_dir)
            else:
                metrics = train_dl_fold(cfg, manifest, fold_path, fold_dir, fold_idx)
            metrics["fold"] = fold_idx
            metrics["seed"] = run_seed
            rows.append(metrics)
            logger.info("Seed %d fold %d metrics: %s", run_seed, fold_idx, metrics)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    agg = {
        "macro_f1_mean": float(summary["macro_f1"].mean()),
        "macro_f1_std": float(summary["macro_f1"].std()),
        "kappa_mean": float(summary["kappa"].mean()),
        "kappa_std": float(summary["kappa"].std()),
        "accuracy_mean": float(summary["accuracy"].mean()),
        "accuracy_std": float(summary["accuracy"].std()),
        "n_seeds": n_seeds,
        "n_folds": n_folds,
    }
    save_json(output_dir / "aggregate.json", agg)
    return summary
