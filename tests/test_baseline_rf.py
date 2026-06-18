"""Regression tests for classical baseline defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

from sleepstage.models.baseline_rf import RandomForestBaseline


def test_rf_baseline_config_default_is_random_forest() -> None:
    cfg = yaml.safe_load((Path("configs/model/rf_baseline.yaml")).read_text())
    assert cfg["classifier"] == "random_forest"


def test_random_forest_baseline_constructor_default() -> None:
    model = RandomForestBaseline()
    assert model.classifier == "random_forest"
