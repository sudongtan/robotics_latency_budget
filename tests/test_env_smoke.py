"""Phase 0 smoke test: the Playground Go1 env loads, resets and steps on CPU.

Also pins the env facts the latency design depends on (n_sub, obs layout), so a
Playground upgrade that changes them fails loudly instead of silently.
"""

import jax
import jax.numpy as jp
import pytest
from mujoco_playground import registry

ENV_NAME = "Go1JoystickFlatTerrain"


@pytest.fixture(scope="module")
def env():
    return registry.load(ENV_NAME, config_overrides={"impl": "jax"})


def test_timing(env):
    assert env.dt == pytest.approx(0.02)
    assert env.sim_dt == pytest.approx(0.004)
    assert env.n_substeps == 5


def test_reset_and_step(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert env.action_size == 12
    assert state.obs["state"].shape == (48,)
    assert state.obs["privileged_state"].shape == (123,)

    step = jax.jit(env.step)
    for _ in range(10):
        state = step(state, jp.zeros(env.action_size))
    assert bool(jp.all(jp.isfinite(state.obs["state"])))
    # Zero action = hold the default pose; the robot must not fall in 0.2 s.
    assert float(state.done) == 0.0
