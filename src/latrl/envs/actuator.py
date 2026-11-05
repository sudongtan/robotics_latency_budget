"""Joint-level actuation: explicit PD torque and first-order motor lag (plan §6, D11).

On a real Go1 the motor driver closes the PD loop at high rate next to the motor, so this
runs every physics substep and is never delayed. Only the *target* q* comes from the
(possibly delayed) policy.

Motor lag. A motor does not produce the commanded torque instantly: current in the
windings builds up with a time constant T_m. We model the delivered torque τ as a
first-order low-pass filter of the commanded torque, discretized with backward Euler:

    dτ/dt = (τ_cmd − τ) / T_m   →   τ ← τ + α (τ_cmd − τ),   α = Δp / (T_m + Δp)

T_m = 0 gives α = 1 (no lag). The torque limit is applied after the filter, as the motor
driver saturates what it actually delivers.
"""

from __future__ import annotations

import jax


def pd_torque(q_target: jax.Array, q: jax.Array, qd: jax.Array, kp: float, kd: float) -> jax.Array:
    """τ_cmd = Kp (q* − q) − Kd q̇: a spring toward the target plus a damper on velocity."""
    return kp * (q_target - q) - kd * qd


def lag_alpha(t_m: jax.Array, dt: float) -> jax.Array:
    return dt / (t_m + dt)


def lag_step(tau: jax.Array, tau_cmd: jax.Array, alpha: jax.Array) -> jax.Array:
    return tau + alpha * (tau_cmd - tau)
