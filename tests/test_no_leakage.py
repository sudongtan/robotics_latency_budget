"""R2 property test: perturbing the robot's state after t − d_o never changes the policy
input at t; perturbing it at or before t − d_o does (so the test has teeth)."""

import jax
import numpy as np
import pytest
from conftest import random_actions

T = 8


def _obs_at_T(step, state, acts, kick_step):
    """Roll out; just before step `kick_step` add a velocity kick to the trunk. Returns the
    policy obs after step T (the input for the action at time T)."""
    for t, a in enumerate(acts):
        if t == kick_step:
            qvel = state.data.qvel.at[0:3].add(0.5).at[6:].add(2.0)
            state = state.replace(data=state.data.replace(qvel=qvel))
        state = step(state, a)
    return np.asarray(state.obs["state"])


@pytest.mark.parametrize("d_o", [0, 1, 3, 5])
@pytest.mark.parametrize("d_a", [0, 8])
def test_no_future_state_in_policy_input(make, d_o, d_a):
    env, reset, step, with_latency = make()
    s0 = with_latency(reset(jax.random.PRNGKey(5)), action_delay=d_a, obs_delay=d_o)
    acts = random_actions(T, seed=5)
    clean = _obs_at_T(step, s0, acts, kick_step=None)
    # The obs after step T measures the state at time T. A kick applied just before step k
    # (after the time-k measurement) changes the state from time k + 1 on; the policy at T
    # sees time T − d_o, so the kick must be invisible iff k + 1 > T − d_o.
    for k in range(T):
        kicked = _obs_at_T(step, s0, acts, kick_step=k)
        if k >= T - d_o:
            np.testing.assert_array_equal(kicked, clean, err_msg=f"leak: kick at {k}")
        else:
            assert not np.array_equal(kicked, clean), f"kick at {k} should be visible"
