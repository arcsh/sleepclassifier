"""Training curve plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    output_path: Path,
) -> None:
    """Plot loss and macro-F1 curves.

    Args:
        train_losses: Per-epoch training loss.
        val_losses: Per-epoch validation loss.
        val_f1s: Per-epoch validation macro-F1.
        output_path: Save path.
    """
    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training Loss")

    ax2.plot(epochs, val_f1s, label="Val macro-F1", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.legend()
    ax2.set_title("Validation Macro-F1")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    output_path: Path,
) -> None:
    """Bar chart of class distribution before/after wake trimming.

    Args:
        counts_before: Stage counts before trimming.
        counts_after: Stage counts after trimming.
        output_path: Save path.
    """
    stages = list(counts_before.keys())
    x = range(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [i - width / 2 for i in x], [counts_before[s] for s in stages], width, label="Before trim"
    )
    ax.bar([i + width / 2 for i in x], [counts_after[s] for s in stages], width, label="After trim")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Epoch count")
    ax.legend()
    ax.set_title("Class Distribution Before/After Wake Trimming")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    model_names: list[str],
    macro_f1_means: list[float],
    macro_f1_stds: list[float],
    kappa_means: list[float],
    kappa_stds: list[float],
    output_path: Path,
) -> None:
    """Summary bar chart comparing models.

    Args:
        model_names: Model labels.
        macro_f1_means: Mean macro-F1 per model.
        macro_f1_stds: Std macro-F1 per model.
        kappa_means: Mean kappa per model.
        kappa_stds: Std kappa per model.
        output_path: Save path.
    """
    x = range(len(model_names))
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.bar(x, macro_f1_means, yerr=macro_f1_stds, capsize=4)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Model Comparison: Macro-F1")

    ax2.bar(x, kappa_means, yerr=kappa_stds, capsize=4, color="orange")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.set_ylabel("Cohen's κ")
    ax2.set_title("Model Comparison: κ")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_subject_f1(
    per_subject_scores: dict[str, float],
    output_path: Path,
) -> None:
    """Box/strip plot of per-subject macro-F1.

    Args:
        per_subject_scores: Subject ID → macro-F1.
        output_path: Save path.
    """
    import seaborn as sns

    scores = list(per_subject_scores.values())
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(y=scores, ax=ax)
    sns.stripplot(y=scores, ax=ax, color="black", alpha=0.5, size=4)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Per-Subject Macro-F1 Distribution")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_attention_weights(
    weights: np.ndarray,
    output_path: Path,
) -> None:
    """Visualize attention weights over sequence window.

    Args:
        weights: Attention weight array (seq_len,) or (heads, seq_len).
        output_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    if weights.ndim == 2:
        for i in range(weights.shape[0]):
            ax.plot(weights[i], alpha=0.6, label=f"Head {i}")
        ax.legend()
    else:
        ax.bar(range(len(weights)), weights)
    ax.set_xlabel("Epoch position in window")
    ax.set_ylabel("Attention weight")
    ax.set_title("Attention Weights (center epoch prediction)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
