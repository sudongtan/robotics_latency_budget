"""Joint-level actuation: explicit PD torque and first-order actuator lag (plan §6, D11, D33).

On a real Go1 the motor driver closes the PD loop at high rate next to the motor, so this
runs every physics substep and is never delayed. Only the *target* q* comes from the
(possibly delayed) policy.

Actuator lag (D33). The joint does not follow a new command instantly. We model this as a
first-order low-pass filter on the joint *target* the PD loop tracks, with time constant T_m,
discretized with backward Euler:

    dq_lag/dt = (q* − q_lag) / T_m   →   q_lag ← q_lag + α (q* − q_lag),   α = Δp / (T_m + Δp)

then τ = clip(Kp (q_lag − q) − Kd q̇). T_m = 0 gives α = 1 (no lag).

Why the target and not the torque: filtering the whole PD torque (the first version, D11)
puts the lag inside the joint's feedback loop. For one joint, I·T_m·s³ + I·s² + Kd·s + Kp
is stable only if T_m < Kd/Kp = 0.5/35 ≈ 14 ms (Routh), so above that the joints oscillate
and the robot falls before the policy does anything (F32). Real drivers close the current
loop at kHz rates, so the loop itself is fast; what is slow is the response to new commands.
Filtering the target leaves the PD loop untouched (stable for any T_m) and slows only the
response to commands. Lagging only the P term is not enough either (bound
I·Kd / (I·Kp − Kd²): 21 ms hip, 43 ms knee, → 14 ms as the load grows; F35).
"""

from __future__ import annotations

import jax


def pd_torque(q_target: jax.Array, q: jax.Array, qd: jax.Array, kp: float, kd: float) -> jax.Array:
    """τ_cmd = Kp (q* − q) − Kd q̇: a spring toward the target plus a damper on velocity."""
    return kp * (q_target - q) - kd * qd


def lag_alpha(t_m: jax.Array, dt: float) -> jax.Array:
    return dt / (t_m + dt)


def lag_step(x: jax.Array, x_cmd: jax.Array, alpha: jax.Array) -> jax.Array:
    """One backward-Euler step of a first-order low-pass filter toward x_cmd."""
    return x + alpha * (x_cmd - x)
