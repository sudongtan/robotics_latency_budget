"""Env-level observation delay: the policy sees proprio from t − d_o; last action and
command are undelayed (plan §6, §8; D10)."""

import jax
import numpy as np
import pytest
from conftest import random_actions, rollout

from latrl.envs.specs import COMMAND, LAST_ACT, PROPRIO

T = 12


@pytest.fixture(scope="module")
def reference(make):
    """Undelayed rollout with noise on (noise is drawn at measurement time, so delayed
    obs must carry exactly the same noisy values)."""
    env, reset, step, with_latency = make(noise=1.0)
    s0 = with_latency(reset(jax.random.PRNGKey(3)), obs_delay=0)
    acts = random_actions(T, seed=3)
    return s0, acts, rollout(step, s0, acts)


@pytest.mark.parametrize("d_o", [1, 2, 4, 6])
def test_policy_sees_obs_from_t_minus_d_o(make, reference, d_o):
    s0_ref, acts, ref = reference
    env, reset, step, with_latency = make(noise=1.0)
    s0 = with_latency(reset(jax.random.PRNGKey(3)), obs_delay=d_o)
    out = rollout(step, s0, acts)
    ref_obs = [np.asarray(s0_ref.obs["state"])] + [np.asarray(s.obs["state"]) for s in ref]
    for t, s in enumerate(out, start=1):
        o = np.asarray(s.obs["state"])
        np.testing.assert_array_equal(o[PROPRIO], ref_obs[max(t - d_o, 0)][PROPRIO])
        np.testing.assert_array_equal(o[COMMAND], ref_obs[t][COMMAND])
        np.testing.assert_array_equal(o[LAST_ACT], acts[t - 1])  # D10/D16: action just sent
        # Physics is unaffected by observation delay under a fixed action sequence.
        np.testing.assert_array_equal(np.asarray(s.data.qpos), np.asarray(ref[t - 1].data.qpos))
        # Critic input starts with the (delayed) actor obs.
        np.testing.assert_array_equal(np.asarray(s.obs["privileged_state"][:48]), o)
