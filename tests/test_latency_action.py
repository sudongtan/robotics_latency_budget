"""Env-level action delay: an impulse reaches the joint target exactly d_a substeps late
(plan §8; q = 0, q > 0, r ≠ 0 all covered). Also R2: never earlier."""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from conftest import rollout

T0, T = 3, 9  # impulse at control step T0


@pytest.mark.parametrize("mode", ["explicit_pd", "position"])
@pytest.mark.parametrize("d_a", [0, 2, 5, 7, 13, 20, 30])
def test_impulse_reaches_target_at_d_a(make, mode, d_a):
    env, reset, step, with_latency = make(mode=mode)
    state = with_latency(reset(jax.random.PRNGKey(0)), action_delay=d_a)
    impulse = np.zeros(12, np.float32)
    impulse[4] = 1.0  # FL thigh
    actions = [impulse if t == T0 else np.zeros(12, np.float32) for t in range(T)]
    states = rollout(step, state, actions)
    targets = np.concatenate([np.asarray(s.info["latency"].applied_targets) for s in states])
    default = np.asarray(env._default_pose)
    off = targets[:, 4] - default[4]
    hit = np.flatnonzero(np.abs(off) > 1e-6)
    n = env.n_substeps
    expected = np.arange(T0 * n + d_a, (T0 + 1) * n + d_a)
    expected = expected[expected < T * n]
    np.testing.assert_array_equal(hit, expected)
    np.testing.assert_allclose(off[hit], env._config.action_scale, rtol=1e-6)
    # Other joints never move off the default target.
    np.testing.assert_allclose(
        np.delete(targets, 4, axis=1),
        np.delete(np.broadcast_to(default, targets.shape), 4, axis=1),
        atol=0,
    )


def test_position_mode_ctrl_is_delayed_target(make):
    """In position mode data.ctrl after the last substep equals the delayed target."""
    env, reset, step, with_latency = make(mode="position")
    state = with_latency(reset(jax.random.PRNGKey(0)), action_delay=7)
    s = step(state, jp.ones(12))
    np.testing.assert_allclose(s.data.ctrl, s.info["latency"].applied_targets[-1])
