"""Cost cap for session S (D35, D36): run only the priority tiers that fit in the budget.

Runs as a STEP right after `pick_backend.py`. Measured speed = 20M steps / train time of the
chosen backend's benchmark run (includes compile, so conservative). Projection over every
not-yet-finished RUN job after this step in the sweep file:

    hours = Σ (train steps + eval-grid steps) × arch factor / speed  +  jobs × overhead

GRU policies cost `--gru-factor` (1.5, Mac measurement F24) per step; Brax reference runs
count as 200M steps at the MLP rate. Hours are summed per tier (`TIER` markers); tiers are
kept in order while the cumulative projection is ≤ --max-hours. The first tier that does not
fit is written to <logdir>/stop_before_tier, where launch_sweep's runner stops. Exit 1 (queue
stops now) only if not even the first tier fits. Writes <logdir>/cost_cap.json.

    uv run python scripts/gpu/cost_cap.py configs/sweeps/session_s.yaml @LOGDIR@/impl --max-hours 50
"""

from __future__ import annotations

import argparse
import json
import os
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
    # Owner can raise the budget at relaunch without a commit (a commit would change run ids):
    #   COST_CAP_HOURS=80 bash scripts/gpu/launch_sweep.sh configs/sweeps/session_s.yaml
    if os.environ.get("COST_CAP_HOURS"):
        args.max_hours = float(os.environ["COST_CAP_HOURS"])

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
    tier, skipped = None, 0
    per_tier: dict = {}  # tier -> [projected hours, jobs]
    for line in lines[me + 1 :]:
        if line.startswith("TIER "):
            tier = int(line.split()[1])
            per_tier.setdefault(tier, [0.0, 0])
            continue
        if not line.startswith("RUN "):
            continue
        j = _job(line[4:], grid_steps)
        if (args.runs_dir / f"{j['run_id']}_{sha}" / "DONE").exists():
            skipped += 1
            continue
        h = j["steps"] * (j["factor"] or args.gru_factor) / rate / 3600 + args.overhead_min / 60
        per_tier.setdefault(tier, [0.0, 0])
        per_tier[tier][0] += h
        per_tier[tier][1] += 1

    # Run whole tiers in order while the cumulative projection fits the cap (D36).
    cum, fits, cumulative = 0.0, [], {}
    for t in sorted(per_tier, key=lambda x: -1 if x is None else x):
        cum += per_tier[t][0]
        cumulative[t] = cum
        if cum <= args.max_hours:
            fits.append(t)
        else:
            break
    tiers = sorted(k for k in per_tier if k is not None)
    stop_tier = next((t for t in tiers if t not in fits), None)
    stop_file = impl_file.parent / "stop_before_tier"
    if stop_tier is not None:
        stop_file.write_text(str(stop_tier))
    elif stop_file.exists():
        stop_file.unlink()
    ok = bool(fits) and (None not in per_tier or None in fits)
    record = {
        "impl": impl,
        "env_steps_per_s": rate,
        "jobs_already_done": skipped,
        "tiers": {
            str(k): {"hours": v[0], "jobs": v[1], "cumulative_hours": cumulative.get(k)}
            for k, v in per_tier.items()
        },
        "projected_hours_all": sum(v[0] for v in per_tier.values()),
        "max_hours": args.max_hours,
        "gru_factor": args.gru_factor,
        "runs_tiers": [t for t in fits if t is not None],
        "stop_before_tier": stop_tier,
        "pass": ok,
    }
    (impl_file.parent / "cost_cap.json").write_text(json.dumps(record, indent=2))
    print("cost_cap:", json.dumps(record))
    if not ok:
        print(f"COST CAP: not even the first tier fits in {args.max_hours} h; queue stops")
    elif stop_tier is None:
        print(f"COST CAP OK: all tiers fit ({record['projected_hours_all']:.1f} h)")
    else:
        print(
            f"COST CAP: running tiers {record['runs_tiers']}, skipping tier {stop_tier} and later"
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
