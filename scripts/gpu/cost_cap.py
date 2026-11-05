"""Cost cap for session S (Q17, D35): stop the queue if the projected hours exceed a limit.

Runs as a STEP right after `pick_backend.py`. Measured speed = 20M steps / train time of the
chosen backend's benchmark run (includes compile, so conservative). Projection over every
not-yet-finished RUN job after this step in the sweep file:

    hours = Σ (train steps + eval-grid steps) × arch factor / speed  +  jobs × overhead

GRU policies cost `--gru-factor` (1.5, Mac measurement F24) per step; Brax reference runs
count as 200M steps at the MLP rate. Exit 1 (queue stops) if hours > --max-hours.
Writes <impl file dir>/cost_cap.json with every number.

    uv run python scripts/gpu/cost_cap.py configs/sweeps/session_s.yaml @LOGDIR@/impl --max-hours 50
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))
import sweep  # noqa: E402

CFG = ROOT / "configs"


def _bench_rate(runs: pathlib.Path, impl: str, sha: str) -> float:
    run = runs / f"bench_{impl}_A_mlp_s0_{sha}"
    last = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    return float(last["env_steps"]) / float(last["train_s"])


def _job(cmd: str, grid_steps: float) -> dict:
    """Parse one RUN command: script, experiment, overrides → steps, factor, run id parts."""
    args = shlex.split(cmd)[3:]  # drop "uv run python"
    script, kv = args[0], dict(a.split("=", 1) for a in args[1:] if "=" in a)
    if script.endswith("brax_reference.py"):
        return {
            "steps": 2e8,
            "factor": 1.0,
            "run_id": f"{kv.get('prefix', '')}brax_{kv.get('env', 'ours')}_s{kv['seed']}",
        }
    exp = OmegaConf.load(CFG / "experiment" / f"{kv['experiment']}.yaml")
    policy = next(d["override /policy"] for d in exp.defaults if "override /policy" in d)
    pol = OmegaConf.load(CFG / "policy" / f"{policy}.yaml")
    steps = float(kv.get("train.total_steps", 2e8))
    if kv.get("mjx_eval.enabled", "true") != "false":
        steps += grid_steps
    return {
        "steps": steps,
        "factor": None if pol.arch == "gru" else 1.0,  # filled with --gru-factor
        "run_id": f"{kv.get('train.run_prefix', '')}{exp.experiment.name}_{pol.name}_s{kv['seed']}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_file")
    ap.add_argument("impl_file")
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--gru-factor", type=float, default=1.5)
    ap.add_argument("--overhead-min", type=float, default=1.5, help="compile/startup per job")
    ap.add_argument("--runs-dir", type=pathlib.Path, default=ROOT / "runs")
    args = ap.parse_args()

    impl_file = pathlib.Path(args.impl_file)
    impl = impl_file.read_text().strip()
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()
    rate = _bench_rate(args.runs_dir, impl, sha)

    from latrl.eval.mjx_eval import grid

    full = OmegaConf.load(CFG / "mjx_eval" / "full.yaml")
    ppo = OmegaConf.load(CFG / "ppo" / "default.yaml")
    grid_steps = len(grid(full)) * full.episodes * ppo.episode_length

    lines = sweep.commands(args.sweep_file)
    me = next(i for i, c in enumerate(lines) if c.startswith("STEP") and "cost_cap.py" in c)
    jobs, skipped = [], 0
    for line in lines[me + 1 :]:
        if not line.startswith("RUN "):
            continue
        j = _job(line[4:], grid_steps)
        if (args.runs_dir / f"{j['run_id']}_{sha}" / "DONE").exists():
            skipped += 1
            continue
        j["factor"] = j["factor"] or args.gru_factor
        jobs.append(j)
    compute_h = sum(j["steps"] * j["factor"] for j in jobs) / rate / 3600
    hours = compute_h + len(jobs) * args.overhead_min / 60
    ok = hours <= args.max_hours
    record = {
        "impl": impl,
        "env_steps_per_s": rate,
        "jobs_remaining": len(jobs),
        "jobs_already_done": skipped,
        "projected_hours": hours,
        "compute_hours": compute_h,
        "max_hours": args.max_hours,
        "gru_factor": args.gru_factor,
        "pass": ok,
    }
    (impl_file.parent / "cost_cap.json").write_text(json.dumps(record, indent=2))
    print("cost_cap:", json.dumps(record))
    print("COST CAP", "OK" if ok else f"EXCEEDED: {hours:.1f} h > {args.max_hours} h; queue stops")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
