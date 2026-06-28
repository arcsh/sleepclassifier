#!/usr/bin/env python3
"""Generate all required figures from processed data and experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from sleepstage.constants import STAGE_NAMES
from sleepstage.evaluation.metrics import transition_matrix
from sleepstage.features.preprocessing import load_processed_npz
from sleepstage.utils.logging import setup_logging
from sleepstage.visualization.confusion import plot_confusion_matrix
from sleepstage.visualization.eeg_plots import plot_example_epochs_per_class, plot_psd_per_class
from sleepstage.visualization.embeddings import plot_embedding_tsne
from sleepstage.visualization.hypnogram import plot_hypnogram_overlay
from sleepstage.visualization.training_curves import (
    plot_attention_weights,
    plot_class_distribution,
    plot_model_comparison,
    plot_per_subject_f1,
    plot_training_curves,
)

logger = logging.getLogger(__name__)

# ponytail: portfolio dirs use poc_* prefix alongside rf_baseline_sleep_edf_sc
PORTFOLIO_EXP_GLOB = ["*_sleep_edf_sc", "poc_*"]


def _find_fold_dirs(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("fold_*")) + sorted(exp_dir.glob("seed_*/fold_*"))


def _discover_experiment_dirs(experiments_dir: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in PORTFOLIO_EXP_GLOB:
        for p in experiments_dir.glob(pattern):
            if p.is_dir():
                found[p.name] = p
    return sorted(found.values(), key=lambda p: p.name)


def _display_name(exp_name: str) -> str:
    if exp_name.startswith("poc_"):
        return exp_name.replace("_sleep_edf_sc", "").replace("poc_", "")
    return exp_name.replace("_sleep_edf_sc", "")


def _best_experiment(exp_dirs: list[Path]) -> Path | None:
    best_dir, best_f1 = None, -1.0
    for exp_dir in exp_dirs:
        agg_path = exp_dir / "aggregate.json"
        if not agg_path.exists():
            continue
        agg = json.loads(agg_path.read_text())
        f1 = float(agg.get("macro_f1_mean", 0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_dir = exp_dir
    return best_dir


def _best_dl_experiment(exp_dirs: list[Path]) -> Path | None:
    dl_dirs = [
        d for d in exp_dirs if "rf_baseline" not in d.name and (d / "aggregate.json").exists()
    ]
    return _best_experiment(dl_dirs) if dl_dirs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_written = 0

    manifest_path = args.processed_dir / "manifest.csv"
    if not manifest_path.exists():
        logger.error("No manifest at %s", manifest_path)
        raise SystemExit(1)

    manifest = pd.read_csv(manifest_path)

    # Fig 1: example epochs per class
    epochs_by_class: dict[str, np.ndarray] = {}
    sfreq = 100.0
    for _, row in manifest.iterrows():
        data = load_processed_npz(Path(row["path"]))
        epochs = data.epochs
        labels = data.labels
        sfreq = data.sample_rate
        for idx, stage in enumerate(STAGE_NAMES):
            if stage not in epochs_by_class:
                mask = labels == idx
                if mask.any():
                    epochs_by_class[stage] = epochs[mask][0, 0]
        if len(epochs_by_class) == len(STAGE_NAMES):
            break
    plot_example_epochs_per_class(epochs_by_class, sfreq, args.output_dir / "01_example_epochs.png")
    figures_written += 1

    # Fig 2: PSD per class
    psd_accum: dict[str, list[np.ndarray]] = {s: [] for s in STAGE_NAMES}
    freqs_ref = None
    for _, row in manifest.head(10).iterrows():
        data = load_processed_npz(Path(row["path"]))
        epochs = data.epochs
        labels = data.labels
        sfreq = data.sample_rate
        for idx, stage in enumerate(STAGE_NAMES):
            mask = labels == idx
            if not mask.any():
                continue
            for ep in epochs[mask][:20]:
                f, p = signal.welch(ep[0], fs=sfreq, nperseg=256)
                psd_accum[stage].append(p)
                freqs_ref = f
    psd_by_class = {
        s: (freqs_ref, np.mean(psd_accum[s], axis=0))
        for s in STAGE_NAMES
        if psd_accum[s] and freqs_ref is not None
    }
    if psd_by_class:
        plot_psd_per_class(psd_by_class, args.output_dir / "02_psd_per_class.png")
        figures_written += 1

    # Fig 3: class distribution (from preprocess or rebuild from manifest)
    fig3 = args.output_dir / "03_class_distribution.png"
    if not fig3.exists():
        counts_before: dict[str, int] = {s: 0 for s in STAGE_NAMES}
        counts_after: dict[str, int] = {s: 0 for s in STAGE_NAMES}
        for _, row in manifest.iterrows():
            data = load_processed_npz(Path(row["path"]))
            for idx, stage in enumerate(STAGE_NAMES):
                n = int((data.labels == idx).sum())
                counts_after[stage] += n
                counts_before[stage] += n  # ponytail: after-trim only in manifest
        plot_class_distribution(counts_before, counts_after, fig3)
    figures_written += 1

    exp_dirs = _discover_experiment_dirs(args.experiments_dir)
    exp_dirs_with_agg = [d for d in exp_dirs if (d / "aggregate.json").exists()]

    model_names, f1_means, f1_stds, k_means, k_stds = [], [], [], [], []
    all_per_subject: dict[str, float] = {}

    for exp_dir in exp_dirs_with_agg:
        agg = json.loads((exp_dir / "aggregate.json").read_text())
        model_names.append(_display_name(exp_dir.name))
        f1_means.append(agg["macro_f1_mean"])
        f1_stds.append(agg["macro_f1_std"])
        k_means.append(agg["kappa_mean"])
        k_stds.append(agg["kappa_std"])

        cms = []
        fold_dirs = _find_fold_dirs(exp_dir)
        for fold_dir in fold_dirs:
            mp = fold_dir / "metrics.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                cms.append(m["confusion_matrix"])
                for sid, score in m.get("per_subject_macro_f1", {}).items():
                    all_per_subject[sid] = score
        if cms:
            from sleepstage.evaluation.metrics import aggregate_confusion_matrices

            plot_confusion_matrix(
                aggregate_confusion_matrices(cms),
                args.output_dir / f"05_confusion_{exp_dir.name}.png",
                title=_display_name(exp_dir.name),
            )
            figures_written += 1

    if model_names:
        plot_model_comparison(
            model_names,
            f1_means,
            f1_stds,
            k_means,
            k_stds,
            args.output_dir / "10_model_comparison.png",
        )
        figures_written += 1
    else:
        logger.warning("No aggregate.json found — skipping model comparison figure")

    if all_per_subject:
        plot_per_subject_f1(all_per_subject, args.output_dir / "08_per_subject_f1.png")
        figures_written += 1

    best_dl = _best_dl_experiment(exp_dirs_with_agg)

    # Fig 6: training curves from best DL fold
    if best_dl is not None:
        best_fold_dirs = _find_fold_dirs(best_dl)
        best_fold = best_fold_dirs[0] if best_fold_dirs else None
        if best_fold is not None:
            hist_path = best_fold / "training_history.json"
            if hist_path.exists():
                history = json.loads(hist_path.read_text())
                plot_training_curves(
                    [h["train_loss"] for h in history],
                    [h["val_loss"] for h in history],
                    [h["val_macro_f1"] for h in history],
                    args.output_dir / "06_training_curves.png",
                )
                figures_written += 1
            else:
                logger.warning("No training_history.json in %s — skipping fig 6", best_fold)
    else:
        logger.warning("No DL experiment with aggregate — skipping training curves")

    # Fig 4: hypnogram (stretch — skip if missing)
    hypno_path = None
    for exp_dir in exp_dirs_with_agg:
        for fold_dir in _find_fold_dirs(exp_dir):
            candidate = fold_dir / "hypnogram_sample.json"
            if candidate.exists():
                hypno_path = candidate
                break
        if hypno_path:
            break
    if hypno_path is None:
        logger.warning("No hypnogram_sample.json — skipping fig 4")
    else:
        hypno = json.loads(hypno_path.read_text())
        plot_hypnogram_overlay(
            np.array(hypno["y_true"]),
            np.array(hypno["y_pred"]),
            args.output_dir / "04_hypnogram_overlay.png",
        )
        figures_written += 1

    # Fig 7: t-SNE (stretch — skip if missing)
    embed_path = None
    for exp_dir in exp_dirs_with_agg:
        for fold_dir in _find_fold_dirs(exp_dir):
            candidate = fold_dir / "embeddings_sample.npz"
            if candidate.exists():
                embed_path = candidate
                break
        if embed_path:
            break
    if embed_path is None:
        logger.warning("No embeddings_sample.npz — skipping fig 7")
    else:
        embed_data = np.load(embed_path)
        plot_embedding_tsne(
            embed_data["embeddings"],
            embed_data["labels"],
            args.output_dir / "07_tsne_embeddings.png",
        )
        figures_written += 1

    # Fig 9: attention weights — skip if missing (portfolio partial runs)
    attn_path = None
    for exp_dir in exp_dirs:
        if "attn" not in exp_dir.name:
            continue
        for fold_dir in _find_fold_dirs(exp_dir):
            candidate = fold_dir / "attention_weights.json"
            if candidate.exists():
                attn_path = candidate
                break
    if attn_path is None:
        logger.warning("No attention_weights.json for AttnSleep — skipping fig 9")
    else:
        attn = json.loads(attn_path.read_text())
        out_path = args.output_dir / "09_attention_weights.png"
        plot_attention_weights(np.array(attn["weights"]), out_path)
        figures_written += 1

    # Transition matrices
    row = manifest.iloc[0]
    data = load_processed_npz(Path(row["path"]))
    labels = data.labels
    tm_gt = transition_matrix(labels.tolist())
    np.savetxt(args.output_dir / "transition_matrix_gt.csv", tm_gt, delimiter=",")

    for exp_dir in exp_dirs_with_agg:
        for fold_dir in _find_fold_dirs(exp_dir):
            pred_tm_path = fold_dir / "transition_matrix_pred.json"
            if pred_tm_path.exists():
                pred_tm = json.loads(pred_tm_path.read_text())["matrix"]
                np.savetxt(
                    args.output_dir / f"transition_matrix_pred_{exp_dir.name}.csv",
                    np.array(pred_tm),
                    delimiter=",",
                )
                break

    logger.info("Figures written to %s (%d figures)", args.output_dir, figures_written)
    if figures_written < 5:
        logger.warning("Fewer than 5 figures produced — training may still be in progress")


if __name__ == "__main__":
    main()
