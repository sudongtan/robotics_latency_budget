"""Follow-up F1 at a glance: latest eval return per arm and seed, and who walks so far.

    uv run python scripts/followup_status.py                 # commit = HEAD
    uv run python scripts/followup_status.py <commit>

Reads runs/{C,C_ft,C_r40,C_hcrit}_mlp_s*_<commit>/eval.jsonl (pulled by sync_down.sh while
the sweep runs). "walks" = return > 25, the walk/stand split of the main results (F43). A
run that is still training can still escape later (late escapes happened at 140-180M), so
only finished runs (DONE) count toward the pre-registered readings (prereg post-hoc F1 plan).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = ["C", "C_ft", "C_r40", "C_hcrit"]
WALK = 25.0


def main() -> None:
    commit = sys.argv[1] if len(sys.argv) > 1 else None
    if commit is None:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    print(f"follow-up F1 at {commit}   (return @ M steps; * = finished)")
    for arm in ARMS:
        cells, walk, done = [], 0, 0
        for seed in range(5):
            d = ROOT / "runs" / f"{arm}_mlp_s{seed}_{commit}"
            f = d / "eval.jsonl"
            if not f.exists():
                cells.append("   -    ")
                continue
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            if not rows:
                cells.append(" start  ")
                continue
            r = rows[-1]
            fin = (d / "DONE").exists()
            done += fin
            walk += fin and r["eval/episode_return"] > WALK
            mark = "*" if fin else " "
            cells.append(f"{r['eval/episode_return']:5.1f}@{r['env_steps'] / 1e6:3.0f}{mark}")
        print(f"{arm:8s} " + " ".join(cells) + f"   walks {walk}/{done} finished")


if __name__ == "__main__":
    main()
