"""Portfolio sweep command-building tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "run_portfolio_sweep.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("run_portfolio_sweep", SWEEP_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_portfolio_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sweep():
    return _load_sweep_module()


def test_build_train_cmd_cnn1d(sweep) -> None:
    cmd = sweep.build_train_cmd("cnn1d", "poc_cnn1d_sleep_edf_sc", 42, 0)
    assert "model=cnn1d" in cmd
    assert "training=portfolio" in cmd
    assert "seed=42" in cmd
    assert "fold_only=0" in cmd
    assert "experiment_name=poc_cnn1d_sleep_edf_sc" in cmd
    assert "output_dir=experiments/poc_cnn1d_sleep_edf_sc" in cmd
    assert "training.num_workers=0" in cmd
    assert "force_seed_subdir=true" in cmd


def test_build_train_cmd_attnsleep_cpu_fallback(sweep) -> None:
    cmd = sweep.build_train_cmd(
        "attnsleep", "poc_attnsleep_sleep_edf_sc", 42, 0, accelerator="cpu", batch_size=32
    )
    assert "model=attnsleep" in cmd
    assert "training.accelerator=cpu" in cmd
    assert "training.batch_size=32" in cmd


def test_portfolio_combos_default(sweep) -> None:
    combos = sweep.portfolio_combos(overnight=False)
    models = [c[0] for c in combos]
    assert "cnn1d" in models
    assert "attnsleep" in models
    cnn_folds = next(f for m, _, f in combos if m == "cnn1d")
    assert cnn_folds == [0, 1]
    attn_folds = next(f for m, _, f in combos if m == "attnsleep")
    assert attn_folds == [0]


def test_portfolio_combos_overnight(sweep) -> None:
    combos = sweep.portfolio_combos(overnight=True)
    cnn_folds = next(f for m, _, f in combos if m == "cnn1d")
    assert cnn_folds == [0, 1, 2, 3, 4]


@patch("run_portfolio_sweep.run_subprocess")
@patch("run_portfolio_sweep.combo_done", return_value=True)
@patch("run_portfolio_sweep._run_figures")
@patch("run_portfolio_sweep.write_portfolio_results_md")
def test_main_skips_done_combos(mock_write, mock_figures, mock_done, mock_run, sweep) -> None:
    mock_run.return_value = (0, "")
    with patch.object(sys, "argv", ["run_portfolio_sweep.py"]):
        sweep.main()
    mock_run.assert_not_called()


@patch("run_portfolio_sweep.run_subprocess")
@patch("run_portfolio_sweep.combo_done")
@patch("run_portfolio_sweep.refresh_aggregates")
@patch("run_portfolio_sweep._run_figures")
@patch("run_portfolio_sweep.write_portfolio_results_md")
@patch("run_portfolio_sweep.update_combo_status")
def test_main_runs_pending_combo(
    mock_status, mock_write, mock_figures, mock_refresh, mock_done, mock_run, sweep
) -> None:
    mock_done.side_effect = lambda exp, seed, fold: fold != 0 or exp != "poc_cnn1d_sleep_edf_sc"
    mock_run.return_value = (0, "ok")
    with patch.object(sys, "argv", ["run_portfolio_sweep.py"]):
        with patch.object(
            sweep, "portfolio_combos", return_value=[("cnn1d", "poc_cnn1d_sleep_edf_sc", [0])]
        ):
            sweep.main()
    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert "model=cnn1d" in cmd
    assert "fold_only=0" in cmd
