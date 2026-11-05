"""In-job MJX evaluation: latency / motor-lag grid → per-episode metrics (plan §6, Phase 2b).

Grid cells (configs/mjx_eval/full.yaml):
- `action`: action delay 0..30 substeps (0–120 ms), no jitter
- `action_jitter`: same bases with jitter on (D14 semantics, D28 parameters)
- `obs`: observation delay 0..6 control steps, action delay 0
- `tm`: motor time constant T_m, delays 0

Every cell runs the same `episodes` episodes: same reset keys (so the same initial states and
command sequences) and the same domain-randomized models. Cells therefore differ only in the
latency setting, and per-latency differences are paired comparisons.

Per episode (true, noise-free state; deterministic policy):
- `fell`, `fall_step`: D17 fall (trunk z < 0.15 m or up-vector z < 0.5) at any control step
- `return`: summed reward until the fall (or the full episode)
- rate metrics (NaN if the window is empty) over the *metric window*: steps ≥
  `metric_start_step` (start-up transient, plan §10) and before any fall.
  `lin_err` = mean ‖cmd_xy − v_xy‖ (body frame), `yaw_err` = mean |cmd_yaw − ω_z|,
  `cot` = Σ mechanical power / (m g Σ speed) with power = Σ_joints |τ·q̇|
  (τ averaged over the step's substeps, q̇ at its end), `jerk` = mean ‖a_t − 2a_{t−1} + a_{t−2}‖²
- gait (robotics context): `duty_factor` = fraction of foot-steps in ground contact (≈0.5 for a
  trot; higher = more feet down, e.g. shuffling); `step_freq_hz` = touchdowns per foot per
  second; `apex_m` = mean peak foot height during swing (ground clearance); `stance_width_m` =
  mean lateral distance between left and right feet in the body frame.

Success (plan §6) needs a tracking threshold set from A's 0 ms runs; it is computed later on
the Mac from these rows, not here.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from latrl.envs.go1 import LatencyJoystick, make_env
from latrl.ppo.trainer import Trainer, TrainState

MS_PER_SUBSTEP = 4.0  # display only (CLAUDE.md: latency stored as integers)
MS_PER_STEP = 20.0
LEFT_FEET, RIGHT_FEET = (1, 3), (0, 2)  # Playground FEET_SITES order: FR, FL, RR, RL


@dataclasses.dataclass(frozen=True)
class Cell:
    axis: str
    action_delay: int  # substeps
    obs_delay: int  # control steps
    t_m: float  # seconds
    jitter: bool


def grid(ecfg: Mapping[str, Any]) -> list[Cell]:
    a = ecfg["action_delays"]
    cells = [Cell("action", d, 0, 0.0, False) for d in range(a["lo"], a["hi"] + 1)]
    cells += [Cell("action_jitter", d, 0, 0.0, True) for d in range(a["lo"], a["hi"] + 1)]
    cells += [Cell("obs", 0, int(d), 0.0, False) for d in ecfg["obs_delays"]]
    cells += [Cell("tm", 0, 0, float(t) / 1000.0, False) for t in ecfg["t_m_ms"]]
    return cells


def _jitter_env_cfg(env_cfg: Mapping[str, Any], ecfg: Mapping[str, Any]) -> Any:
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(dict(env_cfg)), resolve=True))
    cfg.latency.action.jitter_p = float(ecfg["jitter"]["p"])
    cfg.latency.action.jitter_width = int(ecfg["jitter"]["width"])
    return cfg


def _tile(models: Any, in_axes: Any, k: int) -> Any:
    if in_axes is None:
        return models
    return jax.tree.map(
        lambda x, ax: x if ax is None else jp.concatenate([x] * k, axis=0),
        models,
        in_axes,
        is_leaf=lambda x: x is None,
    )


class Evaluator:
    def __init__(self, trainer: Trainer, env_cfg: Mapping[str, Any], ecfg: Mapping[str, Any]):
        self.tr = trainer
        self.ecfg = ecfg
        self.envs = {False: trainer.env, True: make_env(_jitter_env_cfg(env_cfg, ecfg))}
        self.length = int(ecfg["episode_length"] or trainer.cfg.episode_length)
        self.t0 = int(ecfg["metric_start_step"])
        self.min_h = float(ecfg["fall"]["min_height"])
        self.min_upz = float(ecfg["fall"]["min_upz"])
        env = trainer.env
        self.mass = float(env._torso_mass)  # nominal subtree mass of the trunk (whole robot)
        self.g = float(-env.mj_model.opt.gravity[2])
        self._compiled: dict[Any, Any] = {}

    # ------------------------------------------------------------------ helpers

    @contextlib.contextmanager
    def _with_model(self, env: LatencyJoystick, model: Any):
        old = env._mjx_model
        try:
            env._mjx_model = model
            yield
        finally:
            env._mjx_model = old

    def _reset(self, env, models, in_axes, keys, d_a, d_o, t_m):
        def f(m, k, a, o, t):
            with self._with_model(env, m):
                s = env.reset(k)
            return env.with_latency(s, action_delay=a, obs_delay=o, t_m=t)

        return jax.vmap(f, in_axes=(in_axes, 0, 0, 0, 0))(models, keys, d_a, d_o, t_m)

    def _step(self, env, models, in_axes, s, a):
        def f(m, s_, a_):
            with self._with_model(env, m):
                return env.step(s_, a_)

        return jax.vmap(f, in_axes=(in_axes, 0, 0))(models, s, a)

    # ---------------------------------------------------------------- episodes

    def _episodes(self, ts, env, models, in_axes, keys, d_a, d_o, t_m):
        """Run one episode per env; return per-env metric sums (dict of [N] arrays)."""
        tr, n = self.tr, keys.shape[0]
        s = self._reset(env, models, in_axes, keys, d_a, d_o, t_m)
        z = jp.zeros(n)
        z4 = jp.zeros((n, 4))
        init = dict(
            s=s,
            hist=tr.init_hist(s),
            h=tr.actor.initial_carry(n),
            first=jp.ones(n, bool),
            alive=jp.ones(n, bool),
            fall_step=jp.full(n, -1.0),
            ret=z,
            m_steps=z,
            lin=z,
            yaw=z,
            power=z,
            speed=z,
            jerk=z,
            contact=z4,
            touchdowns=z4,
            apex_sum=z,
            apex_n=z,
            width=z,
            swing_max=z4,
            prev_contact=jp.ones((n, 4), bool),  # reset pose: all feet down
            a1=jp.zeros((n, 12)),
            a2=jp.zeros((n, 12)),
        )

        def body(c, t):
            s = c["s"]
            x = tr.features(c["hist"], s)
            a, _, _, h = tr.act(ts, x, c["h"], c["first"], None)
            s = self._step(env, models, in_axes, s, a)
            data = s.data
            upz = jax.vmap(env.get_upvector)(data)[:, -1]
            fallen_now = (data.qpos[:, 2] < self.min_h) | (upz < self.min_upz)
            alive = c["alive"] & ~fallen_now
            fall_step = jp.where(c["alive"] & fallen_now, t.astype(jp.float32), c["fall_step"])
            w = (alive & (t >= self.t0)).astype(jp.float32)  # metric window
            cmd = s.info["command"]
            v = jax.vmap(env.get_local_linvel)(data)
            gyro = jax.vmap(env.get_gyro)(data)
            tau = jp.mean(s.info["latency"].applied_torque, axis=1)  # [N, 12]
            power = jp.sum(jp.abs(tau * data.qvel[:, 6:]), axis=-1)
            speed = jp.linalg.norm(jax.vmap(env.get_global_linvel)(data)[:, :2], axis=-1)
            jerk = jp.sum((a - 2 * c["a1"] + c["a2"]) ** 2, axis=-1)
            contact = s.info["last_contact"]  # this step's foot contacts (upstream sensors)
            touchdown = contact & ~c["prev_contact"]
            foot_z = data.site_xpos[:, env._feet_site_id, 2]
            swing_max = jp.where(contact, 0.0, jp.maximum(c["swing_max"], foot_z))
            apex = jp.where(touchdown, c["swing_max"], 0.0)  # peak of the swing that just ended
            feet = jax.vmap(env.get_feet_pos)(data)  # body (IMU) frame [N, 4, 3]
            width = jp.mean(feet[:, LEFT_FEET, 1], -1) - jp.mean(feet[:, RIGHT_FEET, 1], -1)
            wc = w[:, None]
            c = dict(
                s=s,
                hist=jax.vmap(tr.inp.push)(c["hist"], s.obs["state"]),
                h=h,
                first=jp.zeros(n, bool),
                alive=alive,
                fall_step=fall_step,
                ret=c["ret"] + alive * s.reward,
                m_steps=c["m_steps"] + w,
                lin=c["lin"] + w * jp.linalg.norm(cmd[:, :2] - v[:, :2], axis=-1),
                yaw=c["yaw"] + w * jp.abs(cmd[:, 2] - gyro[:, 2]),
                power=c["power"] + w * power,
                speed=c["speed"] + w * speed,
                jerk=c["jerk"] + w * (t >= 2) * jerk,
                contact=c["contact"] + wc * contact,
                touchdowns=c["touchdowns"] + wc * touchdown,
                apex_sum=c["apex_sum"] + jp.sum(wc * apex, -1),
                apex_n=c["apex_n"] + jp.sum(wc * touchdown, -1),
                width=c["width"] + w * width,
                swing_max=swing_max,
                prev_contact=contact,
                a1=a,
                a2=c["a1"],
            )
            return c, None

        c, _ = jax.lax.scan(body, init, jp.arange(self.length))
        keep = (
            "alive fall_step ret m_steps lin yaw power speed jerk contact touchdowns "
            "apex_sum apex_n width"
        ).split()
        return {k: c[k] for k in keep}

    def _fn(self, name: str, jitter: bool, n: int, in_axes: Any):
        # in_axes is a pytree (not hashable), so it is closed over; it depends only on whether
        # domain randomization is on, which is part of the cache key.
        key = (name, jitter, n, in_axes is None)
        if key not in self._compiled:
            env = self.envs[jitter]
            body = self._episodes if name == "grid" else self._trace_episodes
            self._compiled[key] = jax.jit(
                lambda ts, models, keys, d_a, d_o, t_m: body(
                    ts, env, models, in_axes, keys, d_a, d_o, t_m
                )
            )
        return self._compiled[key]

    # --------------------------------------------------------------------- run

    def run(self, ts: TrainState, key: jax.Array, run_id: str = "") -> pd.DataFrame:
        e = int(self.ecfg["episodes"])
        chunk = int(self.ecfg["chunk_envs"])
        if chunk % e:
            raise ValueError("mjx_eval.chunk_envs must be a multiple of mjx_eval.episodes")
        k_models, k_reset = jax.random.split(key)
        models, in_axes = self._models(k_models, e, bool(self.ecfg["domain_randomization"]))
        ep_keys = jax.random.split(k_reset, e)
        rows = []
        cells = grid(self.ecfg)
        for jitter in (False, True):
            group = [c for c in cells if c.jitter == jitter]
            per = chunk // e
            for i in range(0, len(group), per):
                part = group[i : i + per]
                k = len(part)
                rep = lambda v, dt: jp.repeat(jp.asarray(v, dt), e)  # noqa: E731
                d_a = rep([c.action_delay for c in part], jp.int32)
                d_o = rep([c.obs_delay for c in part], jp.int32)
                t_m = rep([c.t_m for c in part], jp.float32)
                keys = jp.concatenate([ep_keys] * k, axis=0)
                m = _tile(models, in_axes, k)
                out = self._fn("grid", jitter, k * e, in_axes)(ts, m, keys, d_a, d_o, t_m)
                out = jax.device_get(out)
                rows.append(self._rows(part, out, e, run_id))
        return pd.concat(rows, ignore_index=True)

    def _models(self, key, n, dr: bool):
        if not dr:
            return self.tr.env.mjx_model, None
        return self.tr.models(key, n)

    def _rows(self, part: list[Cell], out: dict, e: int, run_id: str) -> pd.DataFrame:
        o = {k: np.asarray(v) for k, v in out.items()}
        ms = np.maximum(o["m_steps"], 1.0)
        dt = float(self.tr.env.dt)
        df = pd.DataFrame(
            {
                "run_id": run_id,
                "axis": np.repeat([c.axis for c in part], e),
                "jitter": np.repeat([c.jitter for c in part], e),
                "action_delay": np.repeat([c.action_delay for c in part], e),
                "obs_delay": np.repeat([c.obs_delay for c in part], e),
                "t_m_s": np.repeat([c.t_m for c in part], e),
                "episode": np.tile(np.arange(e), len(part)),
                "fell": ~o["alive"],
                "fall_step": o["fall_step"].astype(np.int32),
                "return": o["ret"],
                "metric_steps": o["m_steps"].astype(np.int32),
                "lin_err": o["lin"] / ms,
                "yaw_err": o["yaw"] / ms,
                "cot": o["power"] / np.maximum(self.mass * self.g * o["speed"], 1e-6),
                "jerk": o["jerk"] / ms,
                "duty_factor": o["contact"].mean(-1) / ms,
                "step_freq_hz": o["touchdowns"].mean(-1) / (ms * dt),
                "apex_m": o["apex_sum"] / np.maximum(o["apex_n"], 1.0),
                "stance_width_m": o["width"] / ms,
            }
        )
        # Rate metrics are undefined when the robot fell before the metric window opened.
        rate = [
            "lin_err",
            "yaw_err",
            "cot",
            "jerk",
            "duty_factor",
            "step_freq_hz",
            "stance_width_m",
        ]
        df.loc[df.metric_steps == 0, rate] = np.nan
        df.loc[(df.metric_steps == 0) | (o["apex_n"] == 0), "apex_m"] = np.nan
        df["action_delay_ms"] = df["action_delay"] * MS_PER_SUBSTEP  # display
        df["obs_delay_ms"] = df["obs_delay"] * MS_PER_STEP
        return df

    # ------------------------------------------------------------------ traces

    def _trace_episodes(self, ts, env, models, in_axes, keys, d_a, d_o, t_m):
        """Per-step records for probing / gait analysis: [T, N, ...] arrays."""
        tr, n = self.tr, keys.shape[0]
        s = self._reset(env, models, in_axes, keys, d_a, d_o, t_m)
        init = (s, tr.init_hist(s), tr.actor.initial_carry(n), jp.ones(n, bool), jp.ones(n, bool))

        def body(c, _):
            s, hist, h, first, alive = c
            x = tr.features(hist, s)
            dparams, h_new, feats = tr.actor.step_features(
                ts.params["actor"], tr._norm_policy(ts.norm, x), h, first
            )
            a = jp.tanh(jp.split(dparams, 2, axis=-1)[0])  # deterministic, as dist.mode
            lat = s.info["latency"]
            rec = dict(
                obs=s.obs["state"],  # what the actor saw (before its own input assembly)
                features=feats,  # actor's last hidden layer
                hidden=h_new,  # GRU state after this step's input (size 0 for MLPs)
                action=a,
                action_delay=lat.action_delay,  # effective, substeps (probing target; R4)
                obs_delay=lat.obs_delay,
                alive=alive,
            )
            s = self._step(env, models, in_axes, s, a)
            upz = jax.vmap(env.get_upvector)(s.data)[:, -1]
            alive = alive & (s.data.qpos[:, 2] >= self.min_h) & (upz >= self.min_upz)
            rec |= dict(
                qpos=s.data.qpos,
                qvel=s.data.qvel,
                contact=s.info["last_contact"],
                command=s.info["command"],
                feet_pos=jax.vmap(env.get_feet_pos)(s.data),
            )
            hist = jax.vmap(tr.inp.push)(hist, s.obs["state"])
            return (s, hist, h_new, jp.zeros(n, bool), alive), rec

        _, recs = jax.lax.scan(body, init, None, length=self.length)
        return recs

    def traces(self, key: jax.Array) -> dict[str, Any]:
        """{set name: (ts -> dict of numpy arrays [episodes, T, ...])} recorders; latency per
        episode ~ U{lo..hi} substeps, the same draws for every set and every run seed."""
        tc = self.ecfg["traces"]
        e = int(tc["episodes"])
        k_lat, k_models, k_reset = jax.random.split(key, 3)
        d_a = jax.random.randint(k_lat, (e,), tc["latency"]["lo"], tc["latency"]["hi"] + 1)
        zeros_i, zeros_f = jp.zeros(e, jp.int32), jp.zeros(e, jp.float32)
        keys = jax.random.split(k_reset, e)
        out = {}
        for tset in tc["sets"]:
            dr, jit_ = bool(tset["domain_randomization"]), bool(tset["jitter"])
            models, in_axes = self._models(k_models, e, dr)
            fn = self._fn("trace", jit_, e, in_axes)

            def record(ts, fn=fn, models=models):
                recs = jax.device_get(fn(ts, models, keys, d_a, zeros_i, zeros_f))
                arrs = {}
                for k, v in recs.items():
                    v = np.swapaxes(np.asarray(v), 0, 1)  # [N, T, ...]
                    arrs[k] = v.astype(np.float16) if v.dtype == np.float32 else v
                return arrs

            out[str(tset["name"])] = record
        return out
