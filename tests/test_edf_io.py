"""Tests for EDF pairing and preprocess config loading."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from sleepstage.data.edf_io import find_psg_pairs, hypnogram_path_for_psg


def test_hypnogram_path_for_sleep_edf_naming(tmp_path: Path) -> None:
    psg = tmp_path / "SC4001E0-PSG.edf"
    hyp = tmp_path / "SC4001EC-Hypnogram.edf"
    psg.write_bytes(b"x")
    hyp.write_bytes(b"y")
    assert hypnogram_path_for_psg(psg) == hyp


def test_hypnogram_path_eh_suffix(tmp_path: Path) -> None:
    psg = tmp_path / "SC4011E0-PSG.edf"
    hyp = tmp_path / "SC4011EH-Hypnogram.edf"
    psg.write_bytes(b"x")
    hyp.write_bytes(b"y")
    assert hypnogram_path_for_psg(psg) == hyp


def test_find_psg_pairs(tmp_path: Path) -> None:
    psg = tmp_path / "SC4022E0-PSG.edf"
    hyp = tmp_path / "SC4022EJ-Hypnogram.edf"
    psg.write_bytes(b"x")
    hyp.write_bytes(b"y")
    pairs = find_psg_pairs(tmp_path)
    assert pairs == [(psg, hyp)]


def test_compose_config_has_data_group() -> None:
    config_dir = Path("configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg: DictConfig = compose(config_name="config")
    assert "data" in cfg
    assert cfg.data.processed_dir == "data/processed"
