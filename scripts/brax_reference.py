"""Reference run for S1: Brax PPO (brax 0.14.2) with Playground's Go1 joystick config.

Compares our PPO against the implementation the Playground hyperparameters come from.

- `env=ours` (default): our LatencyJoystick with condition A's settings (no latency,
  explicit PD). Differs from our runs only in the PPO implementation. Brax's auto-reset
  does not reset `info` (F16), which is harmless here: with zero delay and T_m = 0 nothing
  stale in `info["latency"]` is ever read.
- `env=upstream`: Playground's own Go1JoystickFlatTerrain (position servos, upstream
  last_act; D16). Checks that our env changes did not make the task harder.

Writes runs/brax_{env}_s{seed}_{commit}/: meta.json, config.yaml, eval.jsonl (Brax eval
metrics at each of `num_evals` points, deterministic policy to match our eval).

    uv run python scripts/brax_reference.py seed=0 env=ours
    # Mac smoke:
    uv run python scripts/brax_reference.py seed=0 num_timesteps=200000 num_envs=64 \
        batch_size=16 num_evals=2 num_eval_envs=16 prefix=smoke_
"""

from __future__ import annotations

import functools
import json
import pathlib
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from omegaconf import OmegaConf

from latrl.envs.go1 import ENV_NAME, make_env
from latrl.utils.provenance import git_commit, write_run_meta

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULTS = {"seed": 0, "env": "ours", "prefix": "", "runs_dir": "runs"}
# Brax keys that may be overridden from the CLI (scale only, for Mac smoke tests).
SCALE_KEYS = (
    "num_timesteps",
    "num_envs",
    "batch_size",
    "num_minibatches",
    "num_evals",
    "num_eval_envs",
)


def _env_cfg():
    cfg = OmegaConf.load(ROOT / "configs" / "env" / "go1_flat.yaml")
    exp = OmegaConf.load(ROOT / "configs" / "experiment" / "A.yaml")
    cfg.latency.action.merge_with(exp.env.latency.action)
    return cfg


def _make(which: str, env_cfg):
    if which == "ours":
        return make_env(env_cfg)
    if which == "upstream":
        return registry.load(
            ENV_NAME,
            config_overrides={"impl": env_cfg.impl, "noise_config.level": env_cfg.noise_level},
        )
    raise ValueError(f"env must be 'ours' or 'upstream', got {which!r}")


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    env_cfg = _env_cfg()
    rl = locomotion_params.brax_ppo_config(ENV_NAME, impl=env_cfg.impl)
    for k in SCALE_KEYS:
        if k in args:
            rl[k] = type(rl[k])(float(args[k]))
    sha, dirty = git_commit()
    run_id = f"{args.prefix}brax_{args.env}_s{args.seed}_{sha}"
    run_dir = ROOT / args.runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    config = {
        "args": OmegaConf.to_container(args),
        "brax_ppo": rl.to_dict(),
        "env": OmegaConf.to_container(env_cfg),
    }
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
    write_run_meta(run_dir, config, int(args.seed))
    print(f"run {run_id} (dirty={dirty})")

    params = rl.to_dict()
    net = params.pop("network_factory")
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net)
    t0 = time.time()
    eval_f = (run_dir / "eval.jsonl").open("a")

    def progress(step, metrics):
        row = {"env_steps": int(step), "wall_s": time.time() - t0}
        row |= {k: float(np.asarray(v)) for k, v in metrics.items()}
        eval_f.write(json.dumps(row) + "\n")
        eval_f.flush()
        print(
            step,
            {k: round(v, 4) for k, v in row.items() if k.startswith("eval/episode_r")},
            flush=True,
        )

    ppo.train(
        environment=_make(args.env, env_cfg),
        eval_env=_make(args.env, env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=registry.get_domain_randomizer(ENV_NAME),
        network_factory=network_factory,
        progress_fn=progress,
        seed=int(args.seed),
        deterministic_eval=True,
        **params,
    )
    eval_f.close()
    (run_dir / "DONE").write_text(
        json.dumps({"wall_s": time.time() - t0, "devices": str(jax.devices())})
    )


if __name__ == "__main__":
    main()
