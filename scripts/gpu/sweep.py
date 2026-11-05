"""Expand a sweep file (configs/sweeps/*.yaml) into one shell command per job.

Sweep format:
    jobs:
      - {experiment: A, seeds: [0, 1], overrides: ["train.total_steps=2e8"]}
      - {script: scripts/brax_reference.py, seeds: [0, 1], overrides: ["env=ours"]}

`script` defaults to scripts/train.py. Overrides are Hydra overrides (train.py) or the
script's own key=value args. Used by launch_sweep.sh; `--dry-run` there prints the list.
"""

import shlex
import sys

from omegaconf import OmegaConf


def commands(path: str) -> list[str]:
    sweep = OmegaConf.load(path)
    out = []
    for job in sweep.jobs:
        script = job.get("script", "scripts/train.py")
        for seed in job.seeds:
            args = [f"seed={seed}"] + list(job.get("overrides", []))
            if "experiment" in job:
                args.insert(0, f"experiment={job.experiment}")
            out.append(shlex.join(["uv", "run", "python", script, *args]))
    return out


if __name__ == "__main__":
    print("\n".join(commands(sys.argv[1])))
