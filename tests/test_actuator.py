"""Actuator lag (plan §6, step 8; D33): first-order step response with time constant T_m,
applied to the joint target; the joint loop stays stable for any T_m (F32)."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from latrl.envs import actuator

DT = 0.004  # physics substep


@pytest.mark.parametrize("t_m", [0.004, 0.01, 0.02, 0.04, 0.06])
def test_step_response_time_constant(t_m):
    """Unit step in the command. The filter is the backward-Euler discretization of
    dτ/dt = (τ_cmd − τ)/T_m, so after k substeps τ = 1 − (T_m/(T_m+Δp))^k exactly, and the
    63.2 % point (one time constant) is reached within one substep of T_m."""
    alpha = actuator.lag_alpha(jp.float32(t_m), DT)
    tau, resp = jp.float32(0.0), []
    for _ in range(int(5 * t_m / DT) + 2):
        tau = actuator.lag_step(tau, jp.float32(1.0), alpha)
        resp.append(float(tau))
    resp = np.array(resp)
    k = np.arange(1, len(resp) + 1)
    np.testing.assert_allclose(resp, 1 - (t_m / (t_m + DT)) ** k, rtol=1e-5)
    t_cross = DT * (np.argmax(resp >= 1 - np.exp(-1)) + 1)
    assert abs(t_cross - t_m) <= DT + 1e-9


def test_no_lag_is_identity():
    assert float(actuator.lag_alpha(jp.float32(0.0), DT)) == 1.0
    assert float(actuator.lag_step(jp.float32(3.0), jp.float32(-2.0), 1.0)) == -2.0


def test_env_torque_lags_with_t_m(make):
    """In the env, a large target step produces a torque that rises more slowly with larger
    T_m (the lagged target moves toward the command more slowly); with T_m = 0 the lagged
    target is the commanded one."""
    env, reset, step, with_latency = make()
    first_torque = {}
    for t_m in [0.0, 0.02, 0.06]:
        s = with_latency(reset(jax.random.PRNGKey(0)), action_delay=0, obs_delay=0, t_m=t_m)
        s = step(s, jp.full(12, 0.5))
        lat = s.info["latency"]
        first_torque[t_m] = float(jp.abs(lat.applied_torque).mean())  # over the step's substeps
        if t_m == 0.0:
            np.testing.assert_allclose(np.asarray(lat.q_lag), np.asarray(lat.applied_targets[-1]))
    assert first_torque[0.0] > first_torque[0.02] > first_torque[0.06] > 0.0


@pytest.mark.parametrize("t_m", [0.0, 0.015, 0.03, 0.06])
def test_standing_is_stable_for_any_t_m(make, t_m):
    """F32 regression: with lag on the whole PD torque the robot flipped over within 3 s at
    zero action for T_m >= 14 ms. With lag on the target it must keep standing at every T_m
    (zero action, noise off, 3 s)."""
    env, reset, step, with_latency = make()
    s = with_latency(reset(jax.random.PRNGKey(0)), t_m=t_m)
    min_z, min_up = 1.0, 1.0
    for _ in range(150):
        s = step(s, jp.zeros(12))
        min_z = min(min_z, float(s.data.qpos[2]))
        min_up = min(min_up, float(env.get_upvector(s.data)[-1]))
    assert min_z > 0.2 and min_up > 0.9, (t_m, min_z, min_up)
