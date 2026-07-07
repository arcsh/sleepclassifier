"""Watchdog supervisor tests (archived large-scale pipeline)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src"))


def _load_watchdog():
    spec = importlib.util.spec_from_file_location(
        "watchdog", ROOT / "large-scale-test" / "scripts" / "watchdog.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["watchdog"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_watchdog_self_check() -> None:
    wd = _load_watchdog()
    wd._self_check()


def test_watchdog_dry_run() -> None:
    wd = _load_watchdog()
    wd.run_dry_run()
