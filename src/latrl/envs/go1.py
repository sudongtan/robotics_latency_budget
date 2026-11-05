"""Go1 joystick env with action/observation latency and explicit PD + motor lag.

Thin subclass of Playground's `Joystick` (mujoco_playground 0.2.0). Reward, termination,
command sampling and observation noise are upstream's. What changes:

1. **Physics substeps** (`_physics`). Upstream holds one joint target for all 5 substeps
   and lets MuJoCo's position actuator compute Kp(q* − q), with Kd as passive joint
   damping. We loop over substeps ourselves so that
   - the target applied at each substep comes from the action delay line (resolution one
     substep = 4 ms; D14 semantics under jitter), and
   - in `explicit_pd` mode (D11) the whole PD torque is computed here, low-pass filtered
     (motor lag), and clipped to the torque limit. The model's actuators become plain
     torque motors (gain 1, no bias) and `dof_damping` is 0.
2. **Observation** (`_policy_obs`). The proprioceptive part goes through the observation
   delay line. `last_act` holds the action just sent (D10, D16), undelayed. The command is
   undelayed.

Latency state lives in `state.info["latency"]` (R3). It is sampled in `reset`, so the
training loop must call `reset` for finished episodes; Playground's default auto-reset
restores only data/obs, not info (F16).

The bookkeeping after physics in `step` copies upstream `Joystick.step` line for line; the
`position` mode test (tests/test_go1_wrapper.py) checks we reproduce upstream exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import joystick

from latrl.envs import actuator, latency
from latrl.envs.specs import (
    LAST_ACT,
    NUM_ACTIONS,
    OBS_SIZE,
    PROPRIO,
    EnvSpec,
)

ENV_NAME = "Go1JoystickFlatTerrain"

# fold_in tag for the latency RNG stream, so latency sampling never consumes the upstream
# env's RNG and the upstream trajectory (reset pose, commands, noise) is unchanged.
_LATENCY_STREAM = 0x1A7E


@struct.dataclass
class LatencyState:
    rng: jax.Array
    # Action delay, substeps: per-episode base, jitter offset, effective value at last send.
    action_base: jax.Array
    action_offset: jax.Array
    action_delay: jax.Array
    # Observation delay, control steps.
    obs_base: jax.Array
    obs_offset: jax.Array
    obs_delay: jax.Array
    # Motor time constant (s) and its filter coefficient.
    t_m: jax.Array
    alpha: jax.Array
    actions: latency.DelayLine  # policy outputs [K_a, 12]
    obs: latency.DelayLine  # proprioceptive obs [K_o, 33]
    tau: jax.Array  # delivered motor torque, lag-filter state [12]
    applied_targets: jax.Array  # joint targets used at each substep of the last step [n_sub, 12]
    applied_torque: jax.Array  # actuator torque delivered at each substep of the last step


class LatencyJoystick(joystick.Joystick):
    """Playground's Go1 joystick task with action delay, observation delay and motor lag.

    Each control step: the policy action enters the action delay line; physics runs
    `n_substeps` substeps, each applying the newest arrived target through the actuator
    (explicit PD + lag, or the model's position actuators); then upstream's reward,
    termination and command bookkeeping run unchanged; finally the measured proprio enters
    the observation delay line and the policy sees the newest arrived one. Last action and
    command are never delayed (D10, D16). Delays are sampled per episode at reset and live
    in `state.info["latency"]` (R3); `with_latency` pins them for tests and evaluation.
    """

    def __init__(
        self,
        spec: EnvSpec,
        task: str = "flat_terrain",
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        # Playground fetches the Go1 meshes (mujoco_menagerie, pinned commit) only in
        # `locomotion.load`; we construct the class directly, so do it here. No-op when
        # present; on a fresh GPU box this is the first clone.
        mjx_env.ensure_menagerie_exists()
        # Fresh config: upstream's default argument is a shared mutable ConfigDict.
        super().__init__(
            task=task,
            config=joystick.default_config(),
            config_overrides=dict(config_overrides or {}),
        )
        self._spec = spec
        m = self._mj_model
        # Actuator i drives joint i (qpos[7 + i]); checked so a model change fails loudly.
        assert m.nu == NUM_ACTIONS
        assert list(m.jnt_qposadr[m.actuator_trnid[:, 0]]) == list(range(7, 7 + NUM_ACTIONS))
        # Joint-target limits = original position-actuator ctrlrange (= joint range).
        self._q_lo = jp.array(m.actuator_ctrlrange[:, 0])
        self._q_hi = jp.array(m.actuator_ctrlrange[:, 1])
        self._tau_max = jp.array(m.actuator_forcerange[:, 1])

        if spec.actuator.mode == "explicit_pd":
            # Turn the position servos into plain torque motors: force = ctrl.
            m.dof_damping[6:] = 0.0
            m.actuator_gainprm[:, 0] = 1.0
            m.actuator_biasprm[:, :3] = 0.0
            m.actuator_ctrlrange[:] = m.actuator_forcerange
            self._mjx_model = mjx.put_model(m, impl=self._config.impl)

        self._k_act = latency.action_buffer_len(spec.action_delay.max, self.n_substeps)
        self._k_obs = latency.obs_buffer_len(spec.obs_delay.max)

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    # ------------------------------------------------------------------ reset

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        assert state.obs["state"].shape == (OBS_SIZE,)
        a, o, act = self._spec.action_delay, self._spec.obs_delay, self._spec.actuator
        key = jax.random.fold_in(rng, _LATENCY_STREAM)
        k_a, k_o, k_t, k_next = jax.random.split(key, 4)
        action_base = latency.sample_base(k_a, a.lo, a.hi)
        obs_base = latency.sample_base(k_o, o.lo, o.hi)
        t_m = jax.random.uniform(k_t, (), minval=act.tm_lo, maxval=act.tm_hi)
        zero = jp.zeros((), jp.int32)
        state.info["latency"] = LatencyState(
            rng=k_next,
            action_base=action_base,
            action_offset=zero,
            action_delay=action_base,
            obs_base=obs_base,
            obs_offset=zero,
            obs_delay=obs_base,
            t_m=t_m,
            alpha=actuator.lag_alpha(t_m, self.sim_dt),
            # Zero action = default pose, as upstream's reset ctrl.
            actions=latency.init(jp.zeros(NUM_ACTIONS), self._k_act),
            obs=latency.init(state.obs["state"][PROPRIO], self._k_obs),
            # At reset joints are at the target with zero velocity, so the PD torque is 0.
            tau=jp.zeros(NUM_ACTIONS),
            applied_targets=jp.broadcast_to(self._default_pose, (self.n_substeps, NUM_ACTIONS)),
            applied_torque=jp.zeros((self.n_substeps, NUM_ACTIONS)),
        )
        return state

    def with_latency(
        self,
        state: mjx_env.State,
        action_delay: jax.Array | int | None = None,
        obs_delay: jax.Array | int | None = None,
        t_m: jax.Array | float | None = None,
    ) -> mjx_env.State:
        """Override the sampled per-episode values right after reset (eval sweeps).

        Values are data, not static, so one compiled function serves every latency.
        """
        lat: LatencyState = state.info["latency"]
        zero = jp.zeros((), jp.int32)
        if action_delay is not None:
            d = jp.asarray(action_delay, jp.int32)
            lat = lat.replace(action_base=d, action_offset=zero, action_delay=d)
        if obs_delay is not None:
            d = jp.asarray(obs_delay, jp.int32)
            lat = lat.replace(obs_base=d, obs_offset=zero, obs_delay=d)
        if t_m is not None:
            t = jp.asarray(t_m, jp.float32)
            lat = lat.replace(t_m=t, alpha=actuator.lag_alpha(t, self.sim_dt))
        info = dict(state.info)
        info["latency"] = lat
        return state.replace(info=info)

    # ------------------------------------------------------------------- step

    def _physics(
        self, data: mjx.Data, lat: LatencyState, actions: latency.DelayLine
    ) -> tuple[mjx.Data, jax.Array, jax.Array, jax.Array]:
        explicit = self._spec.actuator.mode == "explicit_pd"
        kp, kd = self._config.Kp, self._config.Kd

        def substep(carry, i):
            data, tau = carry
            q_target = self._default_pose + latency.read(actions, i) * self._config.action_scale
            if explicit:
                q_target = jp.clip(q_target, self._q_lo, self._q_hi)
                tau_cmd = actuator.pd_torque(q_target, data.qpos[7:], data.qvel[6:], kp, kd)
                tau = actuator.lag_step(tau, tau_cmd, lat.alpha)
                ctrl = jp.clip(tau, -self._tau_max, self._tau_max)
            else:  # upstream position servo; MuJoCo clips ctrl to the joint range
                ctrl = q_target
            data = mjx.step(self.mjx_model, data.replace(ctrl=ctrl))
            return (data, tau), (q_target, data.actuator_force)

        (data, tau), (targets, torques) = jax.lax.scan(
            substep, (data, lat.tau), jp.arange(self.n_substeps)
        )
        return data, tau, targets, torques

    def _policy_obs(
        self, raw: dict[str, jax.Array], proprio: jax.Array, last_act: jax.Array
    ) -> dict[str, jax.Array]:
        s = raw["state"].at[PROPRIO].set(proprio).at[LAST_ACT].set(last_act)
        priv = raw["privileged_state"].at[:OBS_SIZE].set(s)
        return {"state": s, "privileged_state": priv}

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)

        # --- latency: draw this step's delays and send the action -------------------
        a, o = self._spec.action_delay, self._spec.obs_delay
        lat: LatencyState = state.info["latency"]
        rng, k_a, k_o = jax.random.split(lat.rng, 3)
        action_offset = latency.jitter_offset(k_a, lat.action_offset, a.jitter_p, a.jitter_width)
        action_delay = latency.effective(lat.action_base, action_offset, a.max)
        obs_offset = latency.jitter_offset(k_o, lat.obs_offset, o.jitter_p, o.jitter_width)
        obs_delay = latency.effective(lat.obs_base, obs_offset, o.max)
        actions = latency.push(lat.actions, action, action_delay, elapsed=self.n_substeps)

        data, tau, targets, torques = self._physics(state.data, lat, actions)

        # --- upstream Joystick.step bookkeeping (mujoco_playground 0.2.0) -----------
        contact = jp.array(
            [
                data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
                for sensorid in self._feet_floor_found_sensor
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            self.sample_command(key1, state.info["command"]),
            state.info["command"],
        )
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
            state.info["steps_until_next_cmd"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            # Deviation: explicit dtype. Upstream's termination term (bool × float) comes out
            # weak-typed, so the state's pytree type changes after reset and every jitted
            # loop retraces once (F18).
            state.metrics[f"reward/{k}"] = jp.asarray(v, dtype=jp.float32)
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        # --- end upstream bookkeeping ------------------------------------------------

        # --- latency: measure, delay and assemble the policy observation ------------
        obs_line = latency.push(lat.obs, obs["state"][PROPRIO], obs_delay, elapsed=1)
        obs = self._policy_obs(obs, latency.read(obs_line, 0), last_act=action)
        state.info["latency"] = lat.replace(
            rng=rng,
            action_offset=action_offset,
            action_delay=action_delay,
            obs_offset=obs_offset,
            obs_delay=obs_delay,
            actions=actions,
            obs=obs_line,
            tau=tau,
            applied_targets=targets,
            applied_torque=torques,
        )

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)


def make_env(cfg: Mapping[str, Any]) -> LatencyJoystick:
    """Build from an env config (configs/env/go1_flat.yaml). Domain randomization is not
    applied here: the training loop vmaps over randomized models (D15)."""
    if cfg["name"] != ENV_NAME:
        raise ValueError(f"only {ENV_NAME} is supported, got {cfg['name']!r}")
    return LatencyJoystick(
        spec=EnvSpec.from_dict(cfg),
        config_overrides={"impl": cfg["impl"], "noise_config.level": float(cfg["noise_level"])},
    )
