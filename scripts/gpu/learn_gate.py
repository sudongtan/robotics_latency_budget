"""Learnability gate (F50): stop the sweep if a condition's pilot seeds never walked.

    uv run python scripts/gpu/learn_gate.py --seeds 0,1 A_mlp C_r40_mlp ...

Each name is `<experiment>_<policy>` (the run id without seed and commit). A condition
passes if at least --min-walk of its pilot runs at this commit are finished (DONE) and end
with eval return > --walk (25: the walk/stand split of F43). Exit 1 if any condition fails
or a pilot run is missing, so the sweep stops before spending the main budget on it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def final_return(run: pathlib.Path) -> float | None:
    f = run / "eval.jsonl"
    if not (run / "DONE").exists() or not f.exists():
        return None
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return float(rows[-1]["eval/episode_return"]) if rows else None


def check(
    runs_dir: pathlib.Path, names: list[str], seeds: list[int], commit: str, walk: float, k: int
) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for name in names:
        rets = [final_return(runs_dir / f"{name}_s{s}_{commit}") for s in seeds]
        n_walk = sum(r is not None and r > walk for r in rets)
        missing = sum(r is None for r in rets)
        good = n_walk >= k and missing == 0
        ok &= good
        shown = " ".join("missing" if r is None else f"{r:.1f}" for r in rets)
        lines.append(f"{'PASS' if good else 'FAIL'} {name}: walks {n_walk}/{len(seeds)} ({shown})")
    return ok, lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    # One comma-separated value: with nargs="+" the seed list swallowed the names after it
    # and the F2 gate failed on a usage error (D51).
    ap.add_argument("--seeds", type=lambda v: [int(x) for x in v.split(",")], required=True)
    ap.add_argument("--min-walk", type=int, default=1)
    ap.add_argument("--walk", type=float, default=25.0)
    ap.add_argument("--runs-dir", type=pathlib.Path, default=ROOT / "runs")
    ap.add_argument("--commit", default=None)
    a = ap.parse_args(argv)
    commit = (
        a.commit
        or subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    ok, lines = check(a.runs_dir, a.names, a.seeds, commit, a.walk, a.min_walk)
    print("\n".join(lines))
    print("learnability gate:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
