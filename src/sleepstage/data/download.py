"""Dataset download wrappers for Sleep-EDF."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mne

logger = logging.getLogger(__name__)

_FETCH_RETRIES = 5
_FETCH_BACKOFF_SEC = 30
# ponytail: MNE default PhysioNet URLs 404; archive host still serves sleep-edfx.
SC_BASE_URL = "https://archive.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/"
ST_BASE_URL = "https://archive.physionet.org/physiobank/database/sleep-edfx/sleep-telemetry/"


def _fetch_with_retries(fetch_fn, label: str):
    """Retry MNE/pooch fetch on transient network errors."""
    last_err: Exception | None = None
    for attempt in range(1, _FETCH_RETRIES + 1):
        try:
            return fetch_fn()
        except Exception as exc:
            exc_name = type(exc).__name__
            if exc_name not in (
                "ConnectionError",
                "ReadTimeout",
                "Timeout",
                "ChunkedEncodingError",
            ):
                raise
            last_err = exc
            if attempt == _FETCH_RETRIES:
                break
            wait = _FETCH_BACKOFF_SEC * attempt
            logger.warning(
                "%s fetch attempt %d failed: %s — retry in %ds", label, attempt, exc, wait
            )
            time.sleep(wait)
    raise RuntimeError(f"{label} download failed after {_FETCH_RETRIES} attempts") from last_err


def download_sleep_edf_sc(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Download Sleep Cassette (SC) subset via MNE.

    Args:
        raw_dir: Directory to cache raw EDF files.

    Returns:
        List of (psg_path, hypnogram_path) tuples.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    # ponytail: MNE requires explicit subject list; None is rejected
    subjects = [s for s in range(83) if s not in (39, 68, 69, 78, 79)]
    paths = _fetch_with_retries(
        lambda: mne.datasets.sleep_physionet.age.fetch_data(
            subjects,
            recording=[1, 2],
            path=raw_dir,
            base_url=SC_BASE_URL,
            on_missing="ignore",
        ),
        "Sleep-EDF SC",
    )
    pairs: list[tuple[Path, Path]] = []
    for psg, hypno in paths:
        pairs.append((Path(psg), Path(hypno)))
    logger.info("Downloaded %d SC recording pairs to %s", len(pairs), raw_dir)
    return pairs


def download_sleep_edf_st(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Download Sleep Telemetry (ST) subset via MNE.

    Args:
        raw_dir: Directory to cache raw EDF files.

    Returns:
        List of (psg_path, hypnogram_path) tuples.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(range(22))
    paths = _fetch_with_retries(
        lambda: mne.datasets.sleep_physionet.temazepam.fetch_data(
            subjects, path=raw_dir, base_url=ST_BASE_URL
        ),
        "Sleep-EDF ST",
    )
    pairs: list[tuple[Path, Path]] = []
    for psg, hypno in paths:
        pairs.append((Path(psg), Path(hypno)))
    logger.info("Downloaded %d ST recording pairs to %s", len(pairs), raw_dir)
    return pairs
