#!/usr/bin/env python3
"""Train a model with subject-wise cross-validation."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from sleepstage.training.cv_runner import run_cross_validation
from sleepstage.utils.logging import setup_logging
from sleepstage.utils.seed import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(int(cfg.seed))
    manifest_path = Path(cfg.data.processed_dir) / "manifest.csv"
    if not manifest_path.exists():
        logger.error("Manifest not found at %s. Run preprocess first.", manifest_path)
        raise SystemExit(1)
    logger.info("Training %s with config:\n%s", cfg.model.name, OmegaConf.to_yaml(cfg))
    summary = run_cross_validation(cfg, manifest_path)
    logger.info("CV complete:\n%s", summary)


if __name__ == "__main__":
    main()
