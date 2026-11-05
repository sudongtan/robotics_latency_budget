"""Expand a sweep file (configs/sweeps/*.yaml) into one shell command per line.

Sweep format (jobs run in list order):
    jobs:
      - {experiment: A, seeds: [0, 1], overrides: ["env.impl=@IMPL@"]}
      - {script: scripts/brax_reference.py, seeds: [0, 1], overrides: ["env=ours"]}
      - {step: "uv run python scripts/gpu/pick_backend.py @LOGDIR@/impl"}
      - {gate: "uv run python scripts/gpu/gate.py"}
      - {tier: 1, label: "core to 5 seeds"}

- `experiment` / `script` jobs expand to one command per seed (`script` defaults to
  scripts/train.py; overrides are Hydra overrides or the script's key=value args).
- `step` runs a command once; if it fails the queue stops.
- `gate` is a step whose failure is reported as "GATE FAILED" (D27).
- `tier` marks the start of a priority tier (D36). If `@LOGDIR@/stop_before_tier` holds a
  number n (written by cost_cap.py), the queue stops at the marker of tier n.
Placeholders, filled by launch_sweep.sh at run time: `@IMPL@` = contents of @LOGDIR@/impl
(default "jax"), `@LOGDIR@` = the sweep's log directory. Output lines are prefixed `RUN `,
`STEP `, `GATE ` or `TIER `.
"""

import shlex
import sys

from omegaconf import OmegaConf


def commands(path: str) -> list[str]:
    sweep = OmegaConf.load(path)
    out = []
    for job in sweep.jobs:
        if "tier" in job:
            out.append(f"TIER {int(job.tier)} {job.get('label', '')}".rstrip())
            continue
        if "gate" in job:
            out.append(f"GATE {job.gate}")
            continue
        if "step" in job:
            out.append(f"STEP {job.step}")
            continue
        script = job.get("script", "scripts/train.py")
        for seed in job.seeds:
            args = [f"seed={seed}"] + list(job.get("overrides", []))
            if "experiment" in job:
                args.insert(0, f"experiment={job.experiment}")
            out.append("RUN " + shlex.join(["uv", "run", "python", script, *args]))
    return out


if __name__ == "__main__":
    print("\n".join(commands(sys.argv[1])))
