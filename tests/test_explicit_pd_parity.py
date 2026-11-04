"""Explicit PD (T_m = 0) vs upstream position servos (plan §8, D11, F17).

Upstream integrates joint damping explicitly too (`eulerdamp="disable"` in the Go1 XML), so
the only modelling difference is where the torque limit applies: upstream clips Kp(q* − q)
and adds the damper outside the clip; we clip the whole PD torque. Hence:
- while no torque saturates, trajectories agree to float32 rounding (tolerance 1e-4 rad
  over 1 s; measured ~1e-5);
- with saturation they may diverge, but only from the first saturated substep on.
"""

import jax
import numpy as np
from conftest import random_actions, rollout

T = 50  # 1 s
TOL = 1e-4


def _pair(make, acts, key=1):
    trajs = []
    for mode in ["position", "explicit_pd"]:
        _, reset, step, with_latency = make(mode=mode)
        s0 = with_latency(reset(jax.random.PRNGKey(key)), action_delay=0, t_m=0.0)
        trajs.append(rollout(step, s0, acts))
    return trajs


def _saturated(states, tau_max):
    """Per control step: did any actuator hit its torque limit at any substep?"""
    return np.array(
        [
            np.any(np.abs(np.asarray(s.info["latency"].applied_torque)) >= tau_max - 1e-3)
            for s in states
        ]
    )


def test_parity_without_saturation(make):
    t = np.arange(T)[:, None] * 0.02
    sine = (0.3 * np.sin(2 * np.pi * 2 * t + np.arange(12))).astype(np.float32)
    for acts in [np.zeros((T, 12), np.float32), sine]:
        pos, exp = _pair(make, acts)
        env = make(mode="position")[0]
        assert not _saturated(pos, np.asarray(env._tau_max)).any()
        for p, e in zip(pos, exp, strict=True):
            np.testing.assert_allclose(e.data.qpos, p.data.qpos, atol=TOL)
            np.testing.assert_allclose(e.data.qvel, p.data.qvel, atol=100 * TOL)


def test_divergence_only_after_saturation(make):
    acts = random_actions(T, seed=0, scale=1.0)
    pos, exp = _pair(make, acts)
    env = make(mode="position")[0]
    sat = _saturated(pos, np.asarray(env._tau_max)) | _saturated(exp, np.asarray(env._tau_max))
    assert sat.any(), "sequence should saturate some torque"
    first = int(np.argmax(sat))
    for p, e in zip(pos[:first], exp[:first], strict=True):
        np.testing.assert_allclose(e.data.qpos, p.data.qpos, atol=TOL)
