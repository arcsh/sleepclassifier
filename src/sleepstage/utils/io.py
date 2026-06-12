"""I/O helpers for configs, manifests, and experiment artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def save_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Write a JSON file with pretty formatting.

    Args:
        path: Destination file path.
        data: Serializable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    """Load a JSON file.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON content.
    """
    return json.loads(path.read_text())


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a YAML file.

    Args:
        path: Destination file path.
        data: Serializable mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def get_git_hash() -> str:
    """Return current git commit hash or 'unknown' if unavailable.

    Returns:
        Short git hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def snapshot_environment(path: Path) -> None:
    """Save pip freeze output alongside an experiment.

    Args:
        path: Destination file path.
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        path.write_text("# pip freeze unavailable\n")
