"""Sim-to-sim evaluation in native MuJoCo (plan step 23): the MJX eval episodes, replayed.

Episode i here is episode i of the in-job MJX grid (`latrl.eval.mjx_eval`): same eval seed,
same reset keys, same domain-randomization draws, same command and noise sequences, so
native − MJX differences are paired and come only from the simulator (and, under dynamics
perturbations, from the perturbation). Metrics follow `mjx_eval` (fall D17, tracking errors,
CoT, jerk); gait metrics are not computed here.

A *cell* fixes action delay (substeps), obs delay (control steps), T_m (s), jitter on/off,
and an optional dynamics perturbation applied on top of the DR draw:
  friction × f (floor sliding friction), payload kg added to the trunk, motor strength × s
  (torque limit and PD gains scaled together, as a weaker/stronger motor).
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
from flax import struct

from latrl.envs.native.go1_native import Exogenous, NativeGo1, exogenous
from latrl.eval.load import load_run

MS_PER_SUBSTEP, MS_PER_STEP = 4.0, 20.0


@dataclasses.dataclass(frozen=True)
class Cell:
    axis: str
    action_delay: int = 0
    obs_delay: int = 0
    t_m: float = 0.0
    jitter: bool = False
    friction: float = 1.0
    payload_kg: float = 0.0
    motor: float = 1.0


@struct.dataclass
class _Lat:
    action_delay: jax.Array
    obs_delay: jax.Array


class NativeEvaluator:
    def __init__(self, run_dir: pathlib.Path, ecfg: dict[str, Any], checkpoint: str | None = None):
        self.run_dir = run_dir
        self.cfg, self.env, self.tr, self.ts = load_run(run_dir, checkpoint)
        self.ecfg = ecfg
        e_mjx = int(ecfg["mjx_episodes"])  # the MJX grid's episode count: same key splits
        self.e = int(ecfg["episodes"])
        self.length = int(ecfg["episode_length"])
        k_grid, _ = jax.random.split(jax.random.PRNGKey(int(ecfg["seed"])))
        k_models, k_reset = jax.random.split(k_grid)
        models, in_axes = self.tr.models(k_models, e_mjx)
        keys = jax.random.split(k_reset, e_mjx)[: self.e]
        if in_axes is not None:
            models = jax.tree.map(
                lambda x, ax: x if ax is None else x[: self.e],
                models,
                in_axes,
                is_leaf=lambda x: x is None,
            )
        j = ecfg["jitter"]
        self.exo: Exogenous = exogenous(
            self.env, models, in_axes, keys, self.length, float(j["p"]), int(j["width"])
        )
        tr, ts = self.tr, self.ts

        def policy(hist, h, first, a_d, o_d):
            x = jax.vmap(tr.inp.features)(hist, {"latency": _Lat(a_d, o_d)})
            a, _, _, h = tr.act(ts, x, h, first, None)
            return a, h

        self._policy = jax.jit(policy)
        self._push = jax.jit(jax.vmap(tr.inp.push))
        self._init_hist = jax.jit(jax.vmap(tr.inp.init))

    def _env(self, cell: Cell) -> NativeGo1:
        n = self.e
        exo = self.exo
        if cell.friction != 1.0 or cell.payload_kg != 0.0:
            exo = dataclasses.replace(exo, dr={k: v.copy() for k, v in exo.dr.items()})
        env = NativeGo1(
            self.env,
            exo,
            action_delay=np.full(n, cell.action_delay),
            obs_delay=np.full(n, cell.obs_delay),
            t_m=np.full(n, cell.t_m),
            jitter=cell.jitter,
            action_max=int(self.env.spec.action_delay.max),
        )
        trunk = self.env._torso_body_id
        floor = self.env._floor_geom_id
        for m in env.models:
            m.geom_friction[floor, 0] *= cell.friction
            m.body_mass[trunk] += cell.payload_kg
        if cell.motor != 1.0:
            env.kp *= cell.motor
            env.kd *= cell.motor
            env.tau_max = env.tau_max * cell.motor
        if cell.friction != 1.0 or cell.payload_kg != 0.0:
            import mujoco

            for m, d in zip(env.models, env.datas, strict=True):
                mujoco.mj_forward(m, d)
        return env

    def run_cell(self, cell: Cell) -> pd.DataFrame:
        env = self._env(cell)
        n, t0 = self.e, int(self.ecfg["metric_start_step"])
        min_h, min_upz = float(self.ecfg["fall"]["min_height"]), float(self.ecfg["fall"]["min_upz"])
        hist = self._init_hist(jp.asarray(env.obs, jp.float32))
        h = self.tr.actor.initial_carry(n)
        first = jp.ones(n, bool)
        alive = np.ones(n, bool)
        fall_step = np.full(n, -1)
        acc = {k: np.zeros(n) for k in ("m", "lin", "yaw", "power", "speed", "jerk")}
        a1 = a2 = np.zeros((n, 12))
        mass = float(self.env._torso_mass) + cell.payload_kg
        g = float(-self.env.mj_model.opt.gravity[2])
        for t in range(self.length):
            a, h = self._policy(
                hist,
                h,
                first,
                jp.asarray(env.action_delay, jp.int32),
                jp.asarray(env.obs_delay, jp.int32),
            )
            a = np.asarray(a, np.float64)
            env.step(a)
            fallen = (env.height() < min_h) | (env.upz() < min_upz)
            fall_step = np.where(alive & fallen, t, fall_step)
            alive = alive & ~fallen
            w = (alive & (t >= t0)).astype(float)
            cmd = self.exo.command[:, t + 1]
            v = env.local_linvel()
            acc["m"] += w
            acc["lin"] += w * np.linalg.norm(cmd[:, :2] - v[:, :2], axis=-1)
            acc["yaw"] += w * np.abs(cmd[:, 2] - env.gyro()[:, 2])
            tau = env.torque_sum / env.n_sub
            acc["power"] += w * np.sum(np.abs(tau * env.qvel_joints()), -1)
            acc["speed"] += w * env.speed_xy()
            acc["jerk"] += w * (t >= 2) * np.sum((a - 2 * a1 + a2) ** 2, -1)
            a1, a2 = a, a1
            hist = self._push(hist, jp.asarray(env.obs, jp.float32))
            first = jp.zeros(n, bool)
        ms = np.maximum(acc["m"], 1.0)
        df = pd.DataFrame(
            {
                "run_id": self.run_dir.name,
                "sim": "native",
                **dataclasses.asdict(cell),
                "episode": np.arange(n),
                "fell": ~alive,
                "fall_step": fall_step,
                "metric_steps": acc["m"].astype(int),
                "lin_err": acc["lin"] / ms,
                "yaw_err": acc["yaw"] / ms,
                "cot": acc["power"] / np.maximum(mass * g * acc["speed"], 1e-6),
                "jerk": acc["jerk"] / ms,
            }
        )
        df.loc[df.metric_steps == 0, ["lin_err", "yaw_err", "cot", "jerk"]] = np.nan
        df["action_delay_ms"] = df["action_delay"] * MS_PER_SUBSTEP
        df["obs_delay_ms"] = df["obs_delay"] * MS_PER_STEP
        return df
