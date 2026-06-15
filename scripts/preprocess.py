#!/usr/bin/env python3
"""Preprocess raw EDF files into cached NPZ epochs."""

from __future__ import annotations

import argparse
import logging
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from sleepstage.data.edf_io import find_psg_pairs, load_psg_hypnogram, parse_subject_night
from sleepstage.features.preprocessing import (
    PreprocessConfig,
    class_counts,
    preprocess_recording,
    save_processed_npz,
)
from sleepstage.features.splits import compute_group_kfold_splits, persist_splits
from sleepstage.utils.logging import setup_logging
from sleepstage.visualization.training_curves import plot_class_distribution

logger = logging.getLogger(__name__)

SYSTEMIC_SKIP_LIMIT = 5
SKIP_EXIT_CODE = 2


def classify_preprocess_error(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "missing_hypnogram"
    if isinstance(exc, ValueError):
        return "validation"
    name = type(exc).__name__.lower()
    if "edf" in name or exc.__class__.__module__.startswith("mne"):
        return "bad_edf_header"
    return type(exc).__name__


def is_skippable_record_error(exc: BaseException) -> bool:
    return classify_preprocess_error(exc) in {
        "missing_hypnogram",
        "validation",
        "bad_edf_header",
    }


def load_compose_config(config_path: Path) -> DictConfig:
    """Load fully composed Hydra config (resolves config group defaults)."""
    config_dir = config_path.parent.resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-log",
        type=Path,
        default=None,
        help="Log skipped corrupt/malformed recordings (watchdog)",
    )
    args = parser.parse_args()
    setup_logging()

    cfg = load_compose_config(args.config)
    data_cfg = cfg.data
    raw_dir = args.raw_dir or Path(data_cfg.raw_dir) / "sc"
    processed_dir = args.processed_dir or Path(data_cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    channel_variant = data_cfg.get("channel_variant", "single")
    channels = list(
        data_cfg.channels_single if channel_variant == "single" else data_cfg.channels_multi
    )

    pp_cfg = PreprocessConfig(
        bandpass_low=float(data_cfg.bandpass_low),
        bandpass_high=float(data_cfg.bandpass_high),
        epoch_length_sec=float(data_cfg.epoch_length_sec),
        wake_padding_min=float(data_cfg.wake_padding_min),
    )

    pairs = find_psg_pairs(raw_dir)
    if not pairs:
        logger.error("No PSG pairs found in %s. Run scripts/download_data.py first.", raw_dir)
        raise SystemExit(1)

    manifest_rows: list[dict] = []
    manifest_path = processed_dir / "manifest.csv"
    if manifest_path.exists():
        manifest_rows = pd.read_csv(manifest_path).to_dict("records")

    skip_log = args.skip_log
    if skip_log:
        skip_log.parent.mkdir(parents=True, exist_ok=True)
    skip_errors: Counter[str] = Counter()
    all_before: dict[str, int] = {s: 0 for s in ["W", "N1", "N2", "N3", "REM"]}
    all_after: dict[str, int] = {s: 0 for s in ["W", "N1", "N2", "N3", "REM"]}

    def flush_manifest() -> None:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    def already_done(subject_id: str, night_id: str) -> bool:
        return any(
            r.get("subject_id") == subject_id and r.get("night_id") == night_id
            for r in manifest_rows
        )

    for psg_path, hypno_path in pairs:
        try:
            subject_id, night_id = parse_subject_night(psg_path)
            out_path = processed_dir / f"{subject_id}_{night_id}.npz"
            if out_path.exists() and already_done(subject_id, night_id):
                logger.info("Skip existing %s_%s", subject_id, night_id)
                continue

            if not hypno_path.exists():
                raise FileNotFoundError(f"missing hypnogram for {psg_path.name}")

            raw, annotations, meta = load_psg_hypnogram(psg_path, hypno_path, channels)

            # Before trim counts (without wake trim, after label filter only)
            from sleepstage.data.edf_io import annotations_to_epoch_labels as ann_labels
            from sleepstage.features.preprocessing import (
                bandpass_filter,
                epoch_signal,
                map_and_filter_epochs,
            )

            sfreq = float(raw.info["sfreq"])
            data = bandpass_filter(raw.get_data(), sfreq, pp_cfg.bandpass_low, pp_cfg.bandpass_high)
            epochs_raw = epoch_signal(data, sfreq, pp_cfg.epoch_length_sec)
            duration = data.shape[1] / sfreq
            labels_raw, valid = ann_labels(annotations, pp_cfg.epoch_length_sec, duration)
            if len(labels_raw) > len(epochs_raw):
                labels_raw = labels_raw[: len(epochs_raw)]
                valid = valid[: len(epochs_raw)]
            ep_filt, lab_filt = map_and_filter_epochs(epochs_raw, labels_raw, valid)
            for k, v in class_counts(lab_filt).items():
                all_before[k] += v

            epochs, labels = preprocess_recording(raw, annotations, pp_cfg)
            for k, v in class_counts(labels).items():
                all_after[k] += v

            save_processed_npz(out_path, epochs, labels, subject_id, night_id, sfreq)
            counts = class_counts(labels)
            manifest_rows.append(
                {
                    "subject_id": subject_id,
                    "night_id": night_id,
                    "path": str(out_path),
                    "n_epochs": len(labels),
                    **{f"count_{k}": v for k, v in counts.items()},
                }
            )
            flush_manifest()
            logger.info("Processed %s_%s: %d epochs", subject_id, night_id, len(labels))
        except Exception as exc:
            if not is_skippable_record_error(exc):
                logger.exception("Preprocess failed on %s", psg_path)
                raise
            err_class = classify_preprocess_error(exc)
            skip_errors[err_class] += 1
            record_id = f"{psg_path.name}"
            try:
                record_id = f"{parse_subject_night(psg_path)[0]}_{parse_subject_night(psg_path)[1]}"
            except ValueError:
                pass
            msg = (
                f"{datetime.now(timezone.utc).isoformat()} "
                f"id={record_id} class={err_class} error={exc}\n"
                f"{traceback.format_exc()}\n"
            )
            logger.warning("Skipping bad record %s: %s", record_id, exc)
            if skip_log:
                with skip_log.open("a") as f:
                    f.write(msg)
            if skip_errors[err_class] > SYSTEMIC_SKIP_LIMIT:
                logger.error(
                    "Systemic preprocess failure: %s hit %d records (limit %d)",
                    err_class,
                    skip_errors[err_class],
                    SYSTEMIC_SKIP_LIMIT,
                )
                raise SystemExit(SKIP_EXIT_CODE) from exc

    if not manifest_rows:
        logger.error("No recordings processed successfully")
        raise SystemExit(1)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    logger.info("Wrote manifest: %s (%d recordings)", manifest_path, len(manifest))

    # Class distribution figure
    fig_dir = Path("reports/figures")
    plot_class_distribution(all_before, all_after, fig_dir / "03_class_distribution.png")

    # Persist splits
    n_subjects = int(manifest["subject_id"].nunique())
    n_folds = min(int(cfg.n_folds), n_subjects)
    if n_folds < int(cfg.n_folds):
        logger.warning(
            "Only %d subjects available; persisting %d folds (config requests %d)",
            n_subjects,
            n_folds,
            int(cfg.n_folds),
        )
    folds = compute_group_kfold_splits(manifest["subject_id"].tolist(), n_folds=n_folds)
    persist_splits(folds, processed_dir / "splits")


if __name__ == "__main__":
    main()
