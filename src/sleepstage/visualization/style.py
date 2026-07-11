"""Shared matplotlib styling for sleep-staging figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from sleepstage.constants import STAGE_NAMES

# Hypnogram-inspired palette (colorblind-friendly, works on white figure bg)
STAGE_COLORS: dict[str, str] = {
    "W": "#f0b429",
    "N1": "#e07b54",
    "N2": "#5383ae",
    "N3": "#1e3a5f",
    "REM": "#9b6bb8",
}

MODEL_COLORS = {
    "cnn1d": "#5383ae",
    "rf_baseline": "#8a96a3",
    "rf": "#8a96a3",
    "attnsleep": "#9b6bb8",
    "lightgbm": "#6b9bc3",
}

ACCENT = "#5383ae"
ACCENT_2 = "#e07b54"
MUTED = "#8a96a3"
GRID = "#e8ecf0"
TEXT = "#1a1f26"
TEXT_MUTED = "#5c6670"


def apply_style() -> None:
    """Apply blog-friendly rcParams for all sleepclassifier figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "#fafbfc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "600",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.35,
            "savefig.facecolor": "#fafbfc",
        }
    )


def stage_color(stage: str) -> str:
    return STAGE_COLORS.get(stage, ACCENT)


def stage_colors_list(stages: list[str] | None = None) -> list[str]:
    names = stages or STAGE_NAMES
    return [stage_color(s) for s in names]


def prettify_model_name(name: str) -> str:
    mapping = {
        "cnn1d": "CNN1D",
        "rf_baseline": "Random Forest",
        "rf": "Random Forest",
        "attnsleep": "AttnSleep",
        "lightgbm": "LightGBM",
        "cnn_bilstm": "CNN-BiLSTM",
    }
    key = name.lower().replace("poc_", "").replace("_sleep_edf_sc", "")
    return mapping.get(key, name.replace("_", " ").title())


def bar_colors_for_models(names: list[str]) -> list[str]:
    out = []
    for n in names:
        key = n.lower().replace("poc_", "").replace("_sleep_edf_sc", "")
        out.append(MODEL_COLORS.get(key, ACCENT))
    return out


def style_axes(ax: plt.Axes, *, hide_top_right: bool = True) -> None:
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
