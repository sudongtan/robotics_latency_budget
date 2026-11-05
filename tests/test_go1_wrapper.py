"""Our wrapper in `position` mode at zero latency reproduces upstream Playground (physics,
rewards, done, metrics, command sampling, noise), except the `last_act` obs slot (D16).

Not bitwise: the two jitted graphs fuse differently (~1 ulp at step 1), and contact
dynamics amplify rounding to ~1e-4 rad after 1 s of free running (F18). So the test uses
teacher forcing: before each step our state is re-synced to upstream's, and one step is
compared at float32-level tolerance. A logic error (wrong RNG order, wrong target, wrong
bookkeeping) shows up as an O(1) mismatch.
"""

import jax
import numpy as np
from conftest import random_actions
from mujoco_playground import registry

from latrl.envs.go1 import ENV_NAME
from latrl.envs.specs import LAST_ACT

T = 50
TOL = dict(rtol=1e-4, atol=1e-4)


def _sync(ours, up):
    """Copy upstream data/info/metrics into our state, keeping our latency state."""
    info = dict(up.info)
    info["latency"] = ours.info["latency"]
    return ours.replace(data=up.data, info=info, metrics=dict(up.metrics))


def test_position_mode_matches_upstream(make):
    upstream = registry.load(ENV_NAME, config_overrides={"impl": "jax"})  # noise on
    up_step = jax.jit(upstream.step)
    _, reset, step, with_latency = make(mode="position", noise=1.0)
    key = jax.random.PRNGKey(7)
    acts = random_actions(T, seed=7, scale=1.0)
    o = with_latency(reset(key), action_delay=0, obs_delay=0)
    u = jax.jit(upstream.reset)(key)
    np.testing.assert_array_equal(o.obs["state"], u.obs["state"])
    np.testing.assert_array_equal(o.data.qpos, u.data.qpos)
    mask = np.ones(48, bool)
    mask[LAST_ACT] = False
    for t, a in enumerate(acts):
        o = step(_sync(o, u), a)
        u = up_step(u, a)
        np.testing.assert_allclose(o.data.qpos, u.data.qpos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(o.data.qvel, u.data.qvel, **TOL)
        np.testing.assert_allclose(o.reward, u.reward, **TOL)
        np.testing.assert_array_equal(o.done, u.done)
        np.testing.assert_array_equal(o.info["command"], u.info["command"])
        np.testing.assert_array_equal(o.info["rng"], u.info["rng"])
        for k in u.metrics:
            np.testing.assert_allclose(o.metrics[k], u.metrics[k], **TOL, err_msg=k)
        np.testing.assert_allclose(
            np.asarray(o.obs["state"])[mask], np.asarray(u.obs["state"])[mask], **TOL
        )
        np.testing.assert_allclose(
            o.obs["privileged_state"][48:], u.obs["privileged_state"][48:], rtol=1e-3, atol=1e-3
        )
        # D16: upstream's `last_act` is the action from two steps back (obs is built before
        # info["last_act"] is updated); ours is the action just sent.
        np.testing.assert_array_equal(o.obs["state"][LAST_ACT], a)
        prev = acts[t - 1] if t > 0 else np.zeros(12)
        np.testing.assert_array_equal(u.obs["state"][LAST_ACT], prev)
