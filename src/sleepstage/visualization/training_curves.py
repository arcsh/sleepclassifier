"""Training curve plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sleepstage.constants import STAGE_NAMES
from sleepstage.visualization.style import (
    ACCENT,
    ACCENT_2,
    apply_style,
    bar_colors_for_models,
    prettify_model_name,
    stage_color,
    stage_colors_list,
    style_axes,
)


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    output_path: Path,
) -> None:
    """Plot loss and macro-F1 curves."""
    apply_style()
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_losses, label="Train", color=ACCENT, linewidth=2, marker="o", markersize=4)
    ax1.plot(epochs, val_losses, label="Val", color=ACCENT_2, linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training loss")
    style_axes(ax1)

    best_idx = int(np.argmax(val_f1s))
    ax2.plot(epochs, val_f1s, color=ACCENT, linewidth=2.2, marker="o", markersize=5)
    ax2.axvline(best_idx + 1, color=ACCENT_2, linestyle="--", linewidth=1, alpha=0.7)
    ax2.scatter([best_idx + 1], [val_f1s[best_idx]], color=ACCENT_2, s=48, zorder=5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation macro-F1")
    style_axes(ax2)

    fig.suptitle("CNN1D training — early stopping on macro-F1", fontsize=13, fontweight="600", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_class_distribution(
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    output_path: Path,
) -> None:
    """Bar chart of class distribution before/after wake trimming."""
    apply_style()
    stages = list(counts_before.keys())
    before = [counts_before[s] for s in stages]
    after = [counts_after[s] for s in stages]
    after_pct = [100 * v / max(sum(after), 1) for v in after]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(stages))
    w = 0.36

    ax1.bar(x - w / 2, before, w, label="Before trim", color="#8a96a3", alpha=0.55, edgecolor="white")
    ax1.bar(
        x + w / 2,
        after,
        w,
        label="After trim",
        color=stage_colors_list(stages),
        edgecolor="white",
    )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(stages)
    ax1.set_ylabel("Epoch count")
    ax1.set_title("Wake trimming rebalances the dataset")
    ax1.legend(loc="upper right")
    style_axes(ax1)

    y = np.arange(len(stages))
    ax2.barh(y, after_pct, color=stage_colors_list(stages), height=0.62, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(stages)
    ax2.invert_yaxis()
    ax2.set_xlabel("Share of epochs (%)")
    ax2.set_title("Final class mix (after trim)")
    for i, pct in enumerate(after_pct):
        ax2.text(pct + 0.6, i, f"{pct:.1f}%", va="center", fontsize=9.5)
    style_axes(ax2)
    ax2.grid(axis="x")

    fig.suptitle("Class distribution — Sleep-EDF Expanded", fontsize=14, fontweight="600", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_comparison(
    model_names: list[str],
    macro_f1_means: list[float],
    macro_f1_stds: list[float],
    kappa_means: list[float],
    kappa_stds: list[float],
    output_path: Path,
) -> None:
    """Summary bar chart comparing models."""
    apply_style()
    labels = [prettify_model_name(n) for n in model_names]
    colors = bar_colors_for_models(model_names)
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    for ax, means, stds, ylabel, title in [
        (ax1, macro_f1_means, macro_f1_stds, "Macro-F1", "Macro-F1 (higher is better)"),
        (ax2, kappa_means, kappa_stds, "Cohen's κ", "Cohen's κ (agreement vs chance)"),
    ]:
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.52, edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0.48, max(means) + 0.12)
        style_axes(ax)
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="600",
            )

    if len(macro_f1_means) >= 2:
        delta = macro_f1_means[0] - macro_f1_means[1]
        ax1.annotate(
            f"{delta:+.3f}",
            xy=(0, macro_f1_means[0]),
            xytext=(0.45, macro_f1_means[0] + 0.055),
            fontsize=10,
            fontweight="600",
            color=ACCENT,
            ha="center",
        )

    fig.suptitle("Model comparison on held-out subjects", fontsize=14, fontweight="600", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_subject_f1(
    per_subject_scores: dict[str, float],
    output_path: Path,
) -> None:
    """Per-subject macro-F1 distribution."""
    import seaborn as sns

    apply_style()
    scores = np.array(list(per_subject_scores.values()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(scores, positions=[0], widths=0.55, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor(ACCENT)
        body.set_alpha(0.35)
    ax.scatter(np.zeros(len(scores)), scores, color=ACCENT, alpha=0.45, s=22, zorder=3)
    ax.axhline(np.mean(scores), color=ACCENT_2, linestyle="--", linewidth=1, label=f"Mean {np.mean(scores):.3f}")
    ax.set_xticks([])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Per-subject macro-F1 on held-out nights")
    ax.legend(loc="lower left")
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_attention_weights(
    weights: np.ndarray,
    output_path: Path,
) -> None:
    """Visualize attention weights over sequence window."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 3))
    if weights.ndim == 2:
        for i in range(weights.shape[0]):
            ax.plot(weights[i], alpha=0.65, linewidth=1.8, label=f"Head {i + 1}")
        ax.legend(ncol=min(4, weights.shape[0]), loc="upper right")
    else:
        ax.bar(range(len(weights)), weights, color=ACCENT, edgecolor="white")
    ax.set_xlabel("Epoch position in window")
    ax.set_ylabel("Attention weight")
    ax.set_title("Attention weights (center epoch prediction)")
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
