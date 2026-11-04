"""R3: the env jits and vmaps over envs and latencies, compiling once."""

import jax
import jax.numpy as jp
import numpy as np


def test_vmap_over_latencies_single_trace(make):
    env, _, _, _ = make(a_jit=0.3, o_jit=0.3)
    traces = {"reset": 0, "step": 0}

    def reset(key, d_a, d_o, t_m):
        traces["reset"] += 1
        return env.with_latency(env.reset(key), action_delay=d_a, obs_delay=d_o, t_m=t_m)

    def step(state, action):
        traces["step"] += 1
        return env.step(state, action)

    reset_v = jax.jit(jax.vmap(reset))
    step_v = jax.jit(jax.vmap(step))
    n = 4
    for trial in range(2):  # different latency values, same shapes -> no retrace
        keys = jax.random.split(jax.random.PRNGKey(trial), n)
        d_a = jp.array([0, 5, 13, 27]) + trial
        d_o = jp.array([0, 1, 3, 5]) + trial
        t_m = jp.array([0.0, 0.01, 0.02, 0.04]) + 0.005 * trial
        state = reset_v(keys, d_a, d_o, t_m)
        for _ in range(3):
            state = step_v(state, jp.zeros((n, 12)))
        lat = state.info["latency"]
        np.testing.assert_array_equal(lat.action_base, d_a)
        np.testing.assert_array_equal(lat.obs_base, d_o)
        assert bool(jp.all(jp.isfinite(state.obs["state"])))
        assert bool(jp.all(lat.action_delay <= env.spec.action_delay.max))
    assert traces == {"reset": 1, "step": 1}


def test_reset_samples_within_ranges_and_is_deterministic(make):
    env, reset, _, _ = make(a_hi=20, o_hi=4)
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    lat = jax.vmap(reset)(keys).info["latency"]
    assert int(lat.action_base.min()) >= 0 and int(lat.action_base.max()) <= 20
    assert int(lat.obs_base.min()) >= 0 and int(lat.obs_base.max()) <= 4
    assert len(np.unique(lat.action_base)) > 10  # actually random across envs
    lat2 = jax.vmap(reset)(keys).info["latency"]  # R9: same keys, same latencies
    np.testing.assert_array_equal(lat.action_base, lat2.action_base)
