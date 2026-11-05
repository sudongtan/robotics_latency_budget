"""Pick the MJX backend for the rest of session S from the two benchmark runs (D3, D29).

    uv run python scripts/gpu/pick_backend.py runs/_sweeps/<sweep>/impl

Compares training wall-clock (`train_s` at the last metrics line, same step count) of
runs/bench_jax_A_mlp_s0_<commit> and runs/bench_warp_A_mlp_s0_<commit>; a run that did not
finish loses. Writes "jax" or "warp" to the given file and a JSON record next to it.
"""

import json
import pathlib
import subprocess
import sys


def train_s(run: pathlib.Path) -> float:
    if not (run / "DONE").exists():
        return float("inf")
    lines = (run / "metrics.jsonl").read_text().splitlines()
    return float(json.loads(lines[-1])["train_s"])


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    runs = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("runs")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    t = {impl: train_s(runs / f"bench_{impl}_A_mlp_s0_{sha}") for impl in ("jax", "warp")}
    if t["jax"] == float("inf") and t["warp"] == float("inf"):
        print("pick_backend: neither benchmark finished", t)
        sys.exit(1)
    choice = min(t, key=t.get)
    out.write_text(choice)
    out.with_suffix(".json").write_text(json.dumps({"train_s": t, "choice": choice}))
    print("pick_backend:", t, "->", choice)


if __name__ == "__main__":
    main()
