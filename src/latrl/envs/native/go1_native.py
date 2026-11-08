"""Native MuJoCo (C) version of the latency Go1 env, for sim-to-sim evaluation (plan step 22).

Same episodes, different simulator. The *exogenous* inputs of an episode (initial state,
command sequence, sensor noise, jitter offsets, domain-randomized dynamics) come from the
MJX env's own random streams (`exogenous`), so native episode i is MJX episode i with only the
physics engine swapped (MJX JAX → MuJoCo C). What runs here, each control step, mirrors
`latrl.envs.go1.LatencyJoystick.step` (R10):

1. the action enters the action delay line (numpy mirror of `latrl.envs.latency`);
2. `n_substeps` physics substeps, each: delayed target → clip to joint range → actuator lag
   (D33) → PD torque → clip to torque limit → `mj_step`;
3. sensors are read, the step's noise is added, and the proprio part goes through the
   observation delay line; `last_act` = the action just sent; command = upstream's command
   before this step's update (upstream builds the obs before updating the command).

Reward and upstream's termination are not computed: evaluation needs only the D17 fall and
tracking errors. Delay-line semantics are shared with the JAX version by construction and
checked by `tests/test_native_parity.py`.
"""

from __future__ import annotations

import copy
import dataclasses

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

from latrl.envs import latency
from latrl.envs.go1 import LatencyJoystick
from latrl.envs.specs import LAST_ACT, NUM_ACTIONS, OBS_SIZE, PROPRIO

# Domain-randomized fields (Playground go1/randomize.py, mujoco_playground 0.2.0).
DR_FIELDS = ("geom_friction", "body_ipos", "body_mass", "qpos0", "dof_frictionloss", "dof_armature")
# Order of the noisy sensors in upstream `_get_obs` (RNG split order) and their obs slices.
_NOISE_ORDER = (
    ("gyro", slice(3, 6)),
    ("gravity", slice(6, 9)),
    ("joint_pos", slice(9, 21)),
    ("joint_vel", slice(21, 33)),
    ("linvel", slice(0, 3)),
)
_LONG_AGO = -(2**20)


# ----------------------------------------------------------------- delay lines (numpy)
@dataclasses.dataclass
class DelayLine:
    """numpy mirror of `latrl.envs.latency.DelayLine` for a batch: values [N, K, ...]."""

    values: np.ndarray
    arrival: np.ndarray  # [N, K] int

    @classmethod
    def init(cls, value: np.ndarray, length: int) -> DelayLine:
        n = value.shape[0]
        vals = np.repeat(value[:, None], length, axis=1).copy()
        return cls(vals, np.full((n, length), _LONG_AGO, dtype=np.int64))

    def push(self, value: np.ndarray, delay: np.ndarray, elapsed: int) -> None:
        arr = np.maximum(self.arrival - elapsed, _LONG_AGO)
        self.arrival = np.concatenate([np.asarray(delay, np.int64)[:, None], arr[:, :-1]], 1)
        self.values = np.concatenate([value[:, None], self.values[:, :-1]], axis=1)

    def read(self, now: int) -> np.ndarray:
        arrived = self.arrival <= now
        idx = np.where(arrived.any(1), arrived.argmax(1), self.arrival.shape[1] - 1)
        return self.values[np.arange(len(idx)), idx]


# ------------------------------------------------------------- exogenous streams (JAX)
@dataclasses.dataclass
class Exogenous:
    """Per-episode inputs that do not depend on the physics. Arrays are numpy, [N, ...]."""

    qpos: np.ndarray  # [N, nq] initial state
    qvel: np.ndarray  # [N, nv]
    noise0: np.ndarray  # [N, 33] noise of the reset observation (proprio layout)
    noise: np.ndarray  # [N, T, 33] noise added at step t
    command: np.ndarray  # [N, T + 1, 3]: obs at step t uses command[t]; after it, command[t+1]
    jitter: np.ndarray  # [N, T] action-delay jitter offset at step t (substeps)
    dr: dict[str, np.ndarray]  # DR_FIELDS -> [N, ...]; empty without domain randomization


def exogenous(
    env: LatencyJoystick,
    models,
    in_axes,
    keys: jax.Array,
    length: int,
    jitter_p: float = 0.0,
    jitter_width: int = 0,
) -> Exogenous:
    """Replay the MJX env's random streams for episodes `keys` (as `Evaluator.run`)."""
    cfg = env._config
    nz = cfg.noise_config
    scale = {k: float(nz.level * nz.scales[k]) for k, _ in _NOISE_ORDER}
    dt = float(env.dt)

    def one(m, key):
        old = env._mjx_model
        env._mjx_model = m
        try:
            s = env.reset(key)
        finally:
            env._mjx_model = old
        clean = _clean_proprio_jax(env, s.data)
        noise0 = s.obs["state"][PROPRIO] - clean

        def step(c, _):
            rng, cmd, steps, lat_rng, offset = c
            n = jp.zeros(33)
            for name, sl in _NOISE_ORDER:  # upstream _get_obs split order
                rng, k = jax.random.split(rng)
                shape = (sl.stop - sl.start,)
                n = n.at[sl].set((2 * jax.random.uniform(k, shape=shape) - 1) * scale[name])
            rng, k1, k2 = jax.random.split(rng, 3)
            steps = steps - 1
            new_cmd = jp.where(steps <= 0, env.sample_command(k1, cmd), cmd)
            new_steps = jp.where(
                steps <= 0, jp.round(jax.random.exponential(k2) * 5.0 / dt).astype(jp.int32), steps
            )
            lat_rng, k_a, _ = jax.random.split(lat_rng, 3)
            offset = latency.jitter_offset(k_a, offset, jitter_p, jitter_width)
            return (rng, new_cmd, new_steps, lat_rng, offset), (n, cmd, new_cmd, offset)

        lat = s.info["latency"]
        init = (
            s.info["rng"],
            s.info["command"],
            s.info["steps_until_next_cmd"],
            lat.rng,
            jp.zeros((), jp.int32),
        )
        _, (noise, cmd_before, cmd_after, offs) = jax.lax.scan(step, init, None, length=length)
        commands = jp.concatenate([cmd_before[:1], cmd_after], axis=0)
        return s.data.qpos, s.data.qvel, noise0, noise, commands, offs

    in_m = in_axes if in_axes is not None else None
    out = jax.jit(jax.vmap(one, in_axes=(in_m, 0)))(models, keys)
    qpos, qvel, noise0, noise, commands, offs = (np.asarray(x) for x in out)
    dr = {}
    if in_axes is not None:
        dr = {f: np.asarray(getattr(models, f)) for f in DR_FIELDS}
    return Exogenous(qpos, qvel, noise0, noise, commands, offs.astype(np.int64), dr)


def _clean_proprio_jax(env: LatencyJoystick, data) -> jax.Array:
    return jp.concatenate(
        [
            env.get_local_linvel(data),
            env.get_gyro(data),
            env.get_gravity(data),
            data.qpos[7:] - env._default_pose,
            data.qvel[6:],
        ]
    )


# ------------------------------------------------------------------------ native env
class NativeGo1:
    """A batch of N native MuJoCo episodes stepped in lockstep (one MjModel per episode
    when domain randomization is on). Fixed per-episode latency values; jitter offsets come
    from `Exogenous.jitter` when `jitter_width > 0`."""

    def __init__(
        self,
        env: LatencyJoystick,
        exo: Exogenous,
        action_delay: np.ndarray,
        obs_delay: np.ndarray,
        t_m: np.ndarray,
        jitter: bool,
        action_max: int,
    ):
        if env.spec.actuator.mode != "explicit_pd":
            raise ValueError("native env implements explicit_pd only (D11)")
        self.env, self.exo = env, exo
        c = env._config
        self.kp, self.kd, self.scale = float(c.Kp), float(c.Kd), float(c.action_scale)
        self.default = np.asarray(env._default_pose, np.float64)
        self.q_lo, self.q_hi = np.asarray(env._q_lo), np.asarray(env._q_hi)
        self.tau_max = np.asarray(env._tau_max)
        self.n_sub = int(env.n_substeps)
        self.dt_sim = float(env.sim_dt)
        n = exo.qpos.shape[0]
        self.n = n
        base = env.mj_model  # already converted to torque motors (explicit_pd)
        self.models, self.datas = [], []
        for i in range(n):
            m = copy.deepcopy(base)
            for f, v in exo.dr.items():
                getattr(m, f)[:] = v[i]
            d = mujoco.MjData(m)
            d.qpos[:] = exo.qpos[i]
            d.qvel[:] = exo.qvel[i]
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            self.models.append(m)
            self.datas.append(d)
        self._sensor = {
            name: (base.sensor_adr[base.sensor(name).id], base.sensor_dim[base.sensor(name).id])
            for name in (
                consts.GYRO_SENSOR,
                consts.LOCAL_LINVEL_SENSOR,
                consts.UPVECTOR_SENSOR,
                consts.GLOBAL_LINVEL_SENSOR,
            )
        }
        self._imu = base.site("imu").id
        self.action_base = np.asarray(action_delay, np.int64)
        self.obs_delay = np.asarray(obs_delay, np.int64)
        self.jitter = jitter
        self.action_max = int(action_max)
        self.alpha = np.asarray(self.dt_sim / (np.asarray(t_m, np.float64) + self.dt_sim))[:, None]
        k_act = latency.action_buffer_len(env.spec.action_delay.max, self.n_sub)
        k_obs = latency.obs_buffer_len(env.spec.obs_delay.max)
        self.actions = DelayLine.init(np.zeros((n, NUM_ACTIONS)), k_act)
        self.q_lag = np.repeat(self.default[None], n, 0)
        self.t = 0
        self.last_act = np.zeros((n, NUM_ACTIONS))
        proprio0 = self.clean_proprio() + exo.noise0
        self.obs_line = DelayLine.init(proprio0, k_obs)
        self.action_delay = self.action_base.copy()
        self.obs = self._assemble(proprio0, exo.command[:, 0])

    # sensors
    def _s(self, d, name):
        adr, dim = self._sensor[name]
        return d.sensordata[adr : adr + dim]

    def clean_proprio(self) -> np.ndarray:
        out = np.empty((self.n, 33))
        for i, d in enumerate(self.datas):
            grav = d.site_xmat[self._imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
            out[i] = np.concatenate(
                [
                    self._s(d, consts.LOCAL_LINVEL_SENSOR),
                    self._s(d, consts.GYRO_SENSOR),
                    grav,
                    d.qpos[7:] - self.default,
                    d.qvel[6:],
                ]
            )
        return out

    def _assemble(self, proprio: np.ndarray, command: np.ndarray) -> np.ndarray:
        obs = np.zeros((self.n, OBS_SIZE))
        obs[:, PROPRIO] = proprio
        obs[:, LAST_ACT] = self.last_act
        obs[:, 45:48] = command
        return obs

    def step(self, action: np.ndarray) -> None:
        t = self.t
        action = np.asarray(action, np.float64)
        if self.jitter:
            d_a = np.clip(self.action_base + self.exo.jitter[:, t], 0, self.action_max)
        else:
            d_a = self.action_base
        self.action_delay = d_a
        self.actions.push(action, d_a, elapsed=self.n_sub)
        self.torque_sum = np.zeros((self.n, NUM_ACTIONS))
        self.targets = np.zeros((self.n, self.n_sub, NUM_ACTIONS))  # as MJX applied_targets
        for i in range(self.n_sub):
            q_t = np.clip(self.default + self.actions.read(i) * self.scale, self.q_lo, self.q_hi)
            self.targets[:, i] = q_t
            self.q_lag = self.q_lag + self.alpha * (q_t - self.q_lag)
            for j, (m, d) in enumerate(zip(self.models, self.datas, strict=True)):
                tau = self.kp * (self.q_lag[j] - d.qpos[7:]) - self.kd * d.qvel[6:]
                d.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
                mujoco.mj_step(m, d)
                self.torque_sum[j] += d.actuator_force
        proprio = self.clean_proprio() + self.exo.noise[:, t]
        self.obs_line.push(proprio, self.obs_delay, elapsed=1)
        self.last_act = action
        self.obs = self._assemble(self.obs_line.read(0), self.exo.command[:, t])
        self.t += 1

    # measurements for metrics (true state)
    def upz(self) -> np.ndarray:
        return np.array([self._s(d, consts.UPVECTOR_SENSOR)[2] for d in self.datas])

    def height(self) -> np.ndarray:
        return np.array([d.qpos[2] for d in self.datas])

    def local_linvel(self) -> np.ndarray:
        return np.stack([self._s(d, consts.LOCAL_LINVEL_SENSOR) for d in self.datas])

    def gyro(self) -> np.ndarray:
        return np.stack([self._s(d, consts.GYRO_SENSOR) for d in self.datas])

    def qvel_joints(self) -> np.ndarray:
        return np.stack([d.qvel[6:] for d in self.datas])

    def speed_xy(self) -> np.ndarray:
        return np.array(
            [np.linalg.norm(self._s(d, consts.GLOBAL_LINVEL_SENSOR)[:2]) for d in self.datas]
        )
