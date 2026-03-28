"""Persist git revision and CLI args for experiment reproducibility."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def get_git_commit(repo_root: Path) -> str:
    """Return full ``git rev-parse HEAD`` hash, or empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def namespace_to_jsonable(ns: argparse.Namespace) -> Dict[str, Any]:
    """Convert ``argparse.Namespace`` to a JSON-serializable flat dict."""
    out: Dict[str, Any] = {}
    for key, value in vars(ns).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def write_run_metadata_json(
    output_dir: str,
    repo_root: Path,
    args: argparse.Namespace,
) -> None:
    """Write ``run_metadata.json`` under ``output_dir`` with commit hash and args."""
    payload = {
        "git_commit": get_git_commit(repo_root),
        "args": namespace_to_jsonable(args),
    }
    out_path = Path(output_dir).expanduser() / "run_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)
