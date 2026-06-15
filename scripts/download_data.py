#!/usr/bin/env python3
"""Download Sleep-EDF SC (and optionally ST) data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sleepstage.data.download import download_sleep_edf_sc, download_sleep_edf_st
from sleepstage.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Sleep-EDF data")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--subset", choices=["sc", "st", "both"], default="sc")
    args = parser.parse_args()
    setup_logging()

    if args.subset in ("sc", "both"):
        pairs = download_sleep_edf_sc(args.raw_dir / "sc")
        logger.info("SC: %d pairs", len(pairs))
    if args.subset in ("st", "both"):
        pairs = download_sleep_edf_st(args.raw_dir / "st")
        logger.info("ST: %d pairs", len(pairs))


if __name__ == "__main__":
    main()
