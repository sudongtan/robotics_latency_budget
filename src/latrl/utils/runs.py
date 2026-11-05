"""Run-directory bookkeeping for restartable sweeps (session S, D31).

A run directory is **finished** iff it contains `DONE`, written last and atomically. Then:

- `prepare(run_dir)` at the start of a job: finished → "skip" (the job exits at once);
  exists but unfinished (the job a crash interrupted) → moved to `runs/_partial/` and
  "new"; absent → "new".
- `finish(run_dir, info)` writes `DONE` and appends one line to `runs/_ledger.jsonl`, the
  list of trained policies (pulled to the Mac with everything else).
- `finished(runs_dir)` lists finished runs; analysis must read only these (R5): a directory
  without `DONE` is an interrupted run, not a result.

Plain Python (no JAX), so a skipped job costs seconds.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import shutil
from typing import Any

DONE = "DONE"
LEDGER = "_ledger.jsonl"
PARTIAL = "_partial"


def is_finished(run_dir: pathlib.Path) -> bool:
    return (run_dir / DONE).is_file()


def _stamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def prepare(run_dir: pathlib.Path) -> str:
    """Return "skip" if the run is finished; otherwise make sure `run_dir` does not exist
    (moving an interrupted one aside) and return "new"."""
    if is_finished(run_dir):
        return "skip"
    if run_dir.exists():
        dest = run_dir.parent / PARTIAL / f"{run_dir.name}_{_stamp()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(dest))
        print(f"runs: moved interrupted {run_dir.name} -> {dest}", flush=True)
    return "new"


def finish(run_dir: pathlib.Path, info: dict[str, Any]) -> None:
    """Mark the run finished (atomic) and record it in the ledger."""
    tmp = run_dir / (DONE + ".tmp")
    tmp.write_text(json.dumps(info))
    tmp.replace(run_dir / DONE)
    row = {"run_id": run_dir.name, "finished_utc": _stamp()} | info
    with (run_dir.parent / LEDGER).open("a") as f:
        f.write(json.dumps(row) + "\n")


def finished(runs_dir: pathlib.Path) -> list[pathlib.Path]:
    """Finished run directories (excludes `_`-prefixed bookkeeping directories)."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        d
        for d in runs_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and is_finished(d)
    )
