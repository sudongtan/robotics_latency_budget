"""Train one condition. Writes runs/<run_id>/: config.yaml, meta.json (commit, seed),
metrics.jsonl (training), eval.jsonl, checkpoints/, params.json (parameter counts).

    uv run python scripts/train.py experiment=A seed=0
    # Mac smoke:
    uv run python scripts/train.py experiment=A ppo.num_envs=64 ppo.num_minibatches=4 \
        train.total_steps=2e6 train.run_prefix=smoke_

Run id: {prefix}{experiment}_{policy}_s{seed}_{shortcommit} (CLAUDE.md). Refuses to
overwrite an existing run directory. Seeds (R9): one root key from `seed`, split into fixed
named streams below; nothing else draws randomness.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from latrl.envs.go1 import make_env
from latrl.eval.mjx_eval import Evaluator
from latrl.policies.networks import num_params
from latrl.ppo import checkpoint
from latrl.ppo.trainer import Trainer
from latrl.utils.provenance import git_commit, write_run_meta

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _jsonable(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        v = float(np.asarray(v))
        out[k] = None if math.isnan(v) else v
    return out


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    sha, dirty = git_commit()
    run_id = f"{cfg.train.run_prefix}{cfg.experiment.name}_{cfg.policy.name}_s{cfg.seed}_{sha}"
    run_dir = ROOT / cfg.train.runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; delete it or change seed/prefix")
    run_dir.mkdir(parents=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    write_run_meta(run_dir, resolved, int(cfg.seed))
    print(f"run {run_id} (dirty={dirty}) -> {run_dir}")

    env = make_env(cfg.env)
    trainer = Trainer(env, cfg.env, cfg.ppo, cfg.policy)
    c = trainer.cfg

    # R9: fixed key streams.
    root = jax.random.PRNGKey(int(cfg.seed))
    k_dr_train, k_dr_eval, k_init, k_eval = jax.random.split(root, 4)
    models, in_axes = trainer.models(k_dr_train, c.num_envs)
    eval_models, eval_in_axes = trainer.models(k_dr_eval, int(cfg.train.eval_envs))

    ts, carry = jax.jit(lambda k: trainer.init(k, models, in_axes))(k_init)
    counts = {
        "actor": num_params(ts.params["actor"]),
        "critic": num_params(ts.params["critic"]),
        "actor_input_size": trainer.inp.size,
    }
    (run_dir / "params.json").write_text(json.dumps(counts, indent=2))
    print("params", counts)

    iteration = jax.jit(lambda ts, carry: trainer.iteration(ts, carry, models, in_axes))
    evaluate = jax.jit(
        lambda ts, k: trainer.evaluate(ts, eval_models, eval_in_axes, k, int(cfg.train.eval_envs))
    )

    n_iter = max(1, math.ceil(float(cfg.train.total_steps) / c.steps_per_iteration))
    n_evals = max(1, int(cfg.train.num_evals))
    eval_at = {round(n_iter * (i + 1) / n_evals) for i in range(n_evals)}

    wb = None
    if cfg.wandb.enabled:
        import wandb  # optional; credentials from env / ~/.netrc only

        wb = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, name=run_id, config=resolved
        )

    t0 = time.time()
    t_train = 0.0  # wall-clock excluding evals
    metrics_f = (run_dir / "metrics.jsonl").open("a")
    eval_f = (run_dir / "eval.jsonl").open("a")
    for it in range(1, n_iter + 1):
        ti = time.time()
        ts, carry, m = iteration(ts, carry)
        if it % int(cfg.train.log_every) == 0 or it == 1 or it == n_iter:
            m = jax.device_get(m)  # syncs
            t_train += time.time() - ti
            row = {
                "iter": it,
                "env_steps": it * c.steps_per_iteration,
                "wall_s": time.time() - t0,
                "train_s": t_train,
            } | _jsonable(m)
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            if wb:
                wb.log(row, step=row["env_steps"])
            print(
                f"it {it}/{n_iter} steps {row['env_steps']:.3g} "
                f"r/step {row['rollout/reward_per_step']:.4f} "
                f"ep_ret {row['rollout/episode_return']} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            t_train += time.time() - ti  # async dispatch: approximate between syncs
        if it in eval_at:
            e = _jsonable(jax.device_get(evaluate(ts, jax.random.fold_in(k_eval, it))))
            row = {
                "iter": it,
                "env_steps": it * c.steps_per_iteration,
                "wall_s": time.time() - t0,
                "train_s": t_train,
            } | e
            eval_f.write(json.dumps(row) + "\n")
            eval_f.flush()
            if wb:
                wb.log(row, step=row["env_steps"])
            print("eval", row, flush=True)
            checkpoint.save(run_dir / "checkpoints" / f"it{it:06d}.msgpack", ts)
    metrics_f.close()
    eval_f.close()
    t_trained = time.time() - t0

    # In-job MJX eval grid + traces (Phase 2b). Fixed eval seed: same episodes for every run.
    timing = {"train_wall_s": t_trained, "iterations": n_iter}
    if cfg.mjx_eval.enabled:
        evaluator = Evaluator(trainer, cfg.env, cfg.mjx_eval)
        k_grid, k_traces = jax.random.split(jax.random.PRNGKey(int(cfg.mjx_eval.seed)))
        te = time.time()
        df = evaluator.run(ts, k_grid, run_id=run_id)
        df.to_parquet(run_dir / "eval_mjx.parquet")
        timing["eval_grid_s"] = time.time() - te
        print(f"eval grid: {len(df)} episodes in {timing['eval_grid_s']:.0f}s", flush=True)
        if cfg.mjx_eval.traces.enabled:
            te = time.time()
            (run_dir / "traces").mkdir()
            for name, record in evaluator.traces(k_traces).items():
                np.savez_compressed(run_dir / "traces" / f"{name}.npz", **record(ts))
            timing["traces_s"] = time.time() - te
    (run_dir / "DONE").write_text(json.dumps(timing | {"wall_s": time.time() - t0}))
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
