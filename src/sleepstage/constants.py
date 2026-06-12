"""Sleep stage label constants and mappings."""

from __future__ import annotations

# Five-class AASM labels after R&K 3/4 collapse.
STAGE_NAMES: list[str] = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = len(STAGE_NAMES)
STAGE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(STAGE_NAMES)}

# Raw hypnogram annotation strings → target class or None (drop).
RAW_LABEL_MAP: dict[str, str | None] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "Movement time": None,
}

EPOCH_LENGTH_SEC = 30
