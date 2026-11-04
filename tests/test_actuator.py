"""Motor lag (plan §6, step 8): first-order step response with time constant T_m."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from latrl.envs import actuator

DT = 0.004  # physics substep


@pytest.mark.parametrize("t_m", [0.004, 0.01, 0.02, 0.04, 0.06])
def test_step_response_time_constant(t_m):
    """Unit step in commanded torque. The filter is the backward-Euler discretization of
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
    """In the env, a large target step produces a delivered torque that rises more slowly
    with larger T_m; T_m = 0 reproduces the PD torque immediately."""
    env, reset, step, with_latency = make()
    first_torque = {}
    for t_m in [0.0, 0.02, 0.06]:
        s = with_latency(reset(jax.random.PRNGKey(0)), t_m=t_m)
        s = step(s, jp.full(12, 0.5))
        first_torque[t_m] = float(jp.abs(s.info["latency"].tau).mean())
    assert first_torque[0.0] > first_torque[0.02] > first_torque[0.06] > 0.0
