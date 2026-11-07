"""Load a finished run for evaluation on the Mac: config, env, trainer, trained parameters.

    cfg, env, trainer, ts = load_run(pathlib.Path("runs/A_mlp_s0_4a4a467"))          # final
    ..., ts = load_run(path, checkpoint="it000610.msgpack")                         # earlier

The parameter template is built without resetting any env (cheap): actor/critic init,
normalizer init, optimizer init. Checkpoints hold params + normalizer stats only.
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax
from omegaconf import DictConfig, OmegaConf

from latrl.envs.go1 import LatencyJoystick, make_env
from latrl.envs.specs import PRIVILEGED_OBS_SIZE
from latrl.ppo import checkpoint, normalizer
from latrl.ppo.trainer import Norm, Trainer, TrainState


def checkpoints(run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted((run_dir / "checkpoints").glob("it*.msgpack"))


def load_run(
    run_dir: pathlib.Path, checkpoint_name: str | None = None, impl: str = "jax"
) -> tuple[DictConfig, LatencyJoystick, Trainer, TrainState]:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.env.impl = impl  # CLAUDE.md: jax on the Mac (warp is slower on CPU)
    env = make_env(cfg.env)
    tr = Trainer(env, cfg.env, cfg.ppo, cfg.policy)
    k_a, k_c = jax.random.split(jax.random.PRNGKey(0))
    params: dict[str, Any] = {"actor": tr.actor.init(k_a), "critic": tr.critic.init(k_c)}
    template = TrainState(
        params=params,
        opt_state=tr.optimizer.init(params),
        norm=Norm(normalizer.init(tr.inp.size), normalizer.init(PRIVILEGED_OBS_SIZE)),
        key=jax.random.PRNGKey(0),
    )
    path = (
        checkpoints(run_dir)[-1]
        if checkpoint_name is None
        else run_dir / "checkpoints" / checkpoint_name
    )
    restored = checkpoint.load(path, template)
    ts = template.replace(params=restored["params"], norm=restored["norm"])
    return cfg, env, tr, ts
