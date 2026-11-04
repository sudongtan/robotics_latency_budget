"""Run provenance (R5): every output directory records config, commit and seed."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any


def git_commit() -> tuple[str, bool]:
    """(short hash, dirty) of the public repo."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return sha, dirty


def write_run_meta(run_dir: pathlib.Path, config: dict[str, Any], seed: int) -> None:
    sha, dirty = git_commit()
    meta = {"commit": sha, "dirty": dirty, "seed": seed, "config": config}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
