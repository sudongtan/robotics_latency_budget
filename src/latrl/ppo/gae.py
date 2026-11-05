"""Value targets and policy advantages, with separate termination and truncation.

Time-major inputs [T, ...]. For transition t (s_t, a_t, r_t, s'_t):

    δ_t  = r_t + γ (1 − term_t) V(s'_t) − V(s_t)
    A_t  = δ_t + γ λ (1 − done_t) A_{t+1},      done_t = term_t ∨ trunc_t   (GAE)
    vs_t = A_t + V(s_t)                                              (λ-return, value target)

`next_values` is V of the *true* next state s'_t: the state the env reached, before any
auto-reset. On a time-out (truncation) the episode was cut, not ended, so V(s'_t) is kept
as bootstrap; on a termination it is dropped. The recursion stops at done either way, so
nothing leaks across episodes. The last rollout step bootstraps from its own V(s'_{T−1}).

Policy advantage, two estimators (`ppo.advantage_estimator`, one global choice; D20):

- `gae`: A_t above (Schulman et al. 2016; CleanRL, rsl_rl).
- `brax`: r_t + γ (1 − term_t) vs'_t − V(s_t), where vs'_t = vs_{t+1} inside an episode
  and V(s'_t) at done / at the rollout end. This is what Brax PPO feeds the surrogate
  (`losses.compute_gae`, brax 0.14.2) and what the Playground hyperparameters were tuned
  with. It equals δ_t + γ A_{t+1}: one undiscounted-by-λ step more than GAE.

Brax additionally masks truncated transitions out of the loss (bootstrap_on_timeout=False);
we bootstrap them instead. With 1000-step episodes this touches ≈0.1 % of transitions.
"""

from __future__ import annotations

import jax
import jax.numpy as jp

ESTIMATORS = ("brax", "gae")


def gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    termination: jax.Array,
    done: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Return (GAE advantages, λ-return value targets), both [T, ...], gradient-stopped."""
    deltas = rewards + gamma * (1.0 - termination) * next_values - values

    def f(acc, inp):
        delta, d = inp
        acc = delta + gamma * lam * (1.0 - d) * acc
        return acc, acc

    _, adv = jax.lax.scan(f, jp.zeros_like(deltas[0]), (deltas, done), reverse=True)
    adv = jax.lax.stop_gradient(adv)
    return adv, jax.lax.stop_gradient(adv + values)


def advantages(
    estimator: str,
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    termination: jax.Array,
    done: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Return (policy advantages, value targets) for the chosen estimator."""
    adv, vs = gae(rewards, values, next_values, termination, done, gamma, lam)
    if estimator == "gae":
        return adv, vs
    if estimator != "brax":
        raise ValueError(f"advantage_estimator must be one of {ESTIMATORS}, got {estimator!r}")
    vs_next = jp.concatenate([vs[1:], next_values[-1:]], axis=0)
    vs_next = jp.where(done > 0, next_values, vs_next)
    adv_b = rewards + gamma * (1.0 - termination) * vs_next - values
    return jax.lax.stop_gradient(adv_b), vs
