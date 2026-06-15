"""EDF and hypnogram loading with channel validation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingMeta:
    """Metadata for one PSG/hypnogram pair."""

    subject_id: str
    night_id: str
    psg_path: Path
    hypno_path: Path
    sample_rate: float
    channel_names: tuple[str, ...]


def parse_subject_night(psg_path: Path) -> tuple[str, str]:
    """Extract subject and night IDs from a Sleep-EDF filename.

    Args:
        psg_path: Path to *-PSG.edf file.

    Returns:
        Tuple of (subject_id, night_id).

    Raises:
        ValueError: If filename does not match expected pattern.
    """
    match = re.match(r"^(SC\d+)([A-Z]\d+)-PSG\.edf$", psg_path.name, re.IGNORECASE)
    if not match:
        match = re.match(r"^(ST\d+)([A-Z]\d+)-PSG\.edf$", psg_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse subject/night from {psg_path.name}")
    return match.group(1).upper(), match.group(2).upper()


def hypnogram_path_for_psg(psg_path: Path, raw_dir: Path | None = None) -> Path | None:
    """Find matching hypnogram for a Sleep-EDF PSG file.

    Sleep-EDF hypnograms share a prefix with the PSG stem but use a letter
    suffix (e.g. SC4001E0-PSG.edf pairs with SC4001EC-Hypnogram.edf).

    Args:
        psg_path: Path to *-PSG.edf.
        raw_dir: Optional root to search when hypnogram is not beside the PSG.

    Returns:
        Path to hypnogram if exactly one match exists, else None.
    """
    stem = psg_path.name.replace("-PSG.edf", "")
    if len(stem) < 2:
        return None
    prefix = stem[:-1]
    search_roots = [psg_path.parent]
    if raw_dir is not None:
        search_roots.append(raw_dir)
    matches: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        for hypno in root.rglob(f"{prefix}*-Hypnogram.edf"):
            if hypno not in seen:
                seen.add(hypno)
                matches.append(hypno)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact = [m for m in matches if m.name == f"{prefix}C-Hypnogram.edf"]
        if len(exact) == 1:
            return exact[0]
    return None


def find_psg_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Find PSG/hypnogram pairs under a raw data directory.

    Args:
        raw_dir: Root directory containing downloaded EDF files.

    Returns:
        Sorted list of (psg_path, hypnogram_path) tuples.
    """
    pairs: list[tuple[Path, Path]] = []
    for psg in sorted(raw_dir.rglob("*-PSG.edf")):
        hypno = hypnogram_path_for_psg(psg, raw_dir)
        if hypno is not None:
            pairs.append((psg, hypno))
    return pairs


def load_psg_hypnogram(
    psg_path: Path,
    hypno_path: Path,
    channels: list[str],
) -> tuple[mne.io.BaseRaw, mne.Annotations, RecordingMeta]:
    """Load PSG and hypnogram with channel validation.

    Args:
        psg_path: Path to PSG EDF.
        hypno_path: Path to hypnogram EDF+.
        channels: Required channel names to pick.

    Returns:
        Tuple of raw object, hypnogram annotations, and metadata.

    Raises:
        ValueError: If required channels are missing.
    """
    subject_id, night_id = parse_subject_night(psg_path)
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Missing channels {missing} in {psg_path.name}. " f"Available: {raw.ch_names}"
        )
    raw.pick(channels)
    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        raise ValueError(f"Invalid sample rate {sfreq} in {psg_path.name}")

    hypno_raw = mne.io.read_raw_edf(hypno_path, preload=False, verbose=False)
    annotations = mne.read_annotations(hypno_path)
    if len(annotations) == 0:
        raise ValueError(f"No annotations found in {hypno_path}")

    meta = RecordingMeta(
        subject_id=subject_id,
        night_id=night_id,
        psg_path=psg_path,
        hypno_path=hypno_path,
        sample_rate=sfreq,
        channel_names=tuple(channels),
    )
    _ = hypno_raw  # loaded only to validate file readability
    return raw, annotations, meta


def annotations_to_epoch_labels(
    annotations: mne.Annotations,
    epoch_length_sec: float,
    total_duration_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert hypnogram annotations to per-epoch integer labels.

    Args:
        annotations: MNE annotations from hypnogram.
        epoch_length_sec: Epoch duration in seconds.
        total_duration_sec: Total recording duration to epoch.

    Returns:
        Tuple of (label_indices, valid_mask). Invalid epochs have label -1.
    """
    from sleepstage.constants import RAW_LABEL_MAP, STAGE_TO_IDX

    n_epochs = int(total_duration_sec // epoch_length_sec)
    labels = np.full(n_epochs, -1, dtype=np.int64)
    valid = np.zeros(n_epochs, dtype=bool)

    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description, strict=True
    ):
        mapped = RAW_LABEL_MAP.get(desc)
        if mapped is None:
            continue
        stage_idx = STAGE_TO_IDX[mapped]
        start_epoch = int(onset // epoch_length_sec)
        end_epoch = int((onset + duration) // epoch_length_sec)
        for ep in range(start_epoch, min(end_epoch, n_epochs)):
            labels[ep] = stage_idx
            valid[ep] = True

    return labels, valid
