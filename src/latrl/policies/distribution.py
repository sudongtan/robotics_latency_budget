"""Tanh-squashed diagonal Gaussian, as in Brax PPO (`brax/training/distribution.py`,
`NormalTanhDistribution`, brax 0.14.2), which the Playground Go1 reference uses.

The network outputs `2·A` numbers: `loc` and a raw scale; `scale = softplus(raw) + min_std`.
Samples are drawn in pre-tanh space (`raw_action`) and stored that way, so log-probs are
exact; the env receives `tanh(raw_action)` ∈ [−1, 1]. The deterministic action is
`tanh(loc)`.

Entropy matches Brax: Gaussian entropy plus the tanh log-det-Jacobian at one sample (the
squashed distribution has no closed-form entropy).
"""

from __future__ import annotations

import jax
import jax.numpy as jp

_LOG2 = jp.log(2.0)


def split(params: jax.Array, min_std: float) -> tuple[jax.Array, jax.Array]:
    loc, raw = jp.split(params, 2, axis=-1)
    return loc, jax.nn.softplus(raw) + min_std


def _tanh_log_det(u: jax.Array) -> jax.Array:
    # log(1 − tanh(u)²) in a numerically stable form (Brax TanhBijector).
    return 2.0 * (_LOG2 - u - jax.nn.softplus(-2.0 * u))


def sample_raw(key: jax.Array, params: jax.Array, min_std: float) -> jax.Array:
    loc, scale = split(params, min_std)
    return loc + scale * jax.random.normal(key, loc.shape)


def log_prob(params: jax.Array, raw_action: jax.Array, min_std: float) -> jax.Array:
    """Log-density of the squashed action, evaluated via its pre-tanh value."""
    loc, scale = split(params, min_std)
    z = (raw_action - loc) / scale
    normal = -0.5 * z**2 - jp.log(scale) - 0.5 * jp.log(2.0 * jp.pi)
    return jp.sum(normal - _tanh_log_det(raw_action), axis=-1)


def entropy(key: jax.Array, params: jax.Array, min_std: float) -> jax.Array:
    loc, scale = split(params, min_std)
    normal = 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(scale)
    u = loc + scale * jax.random.normal(key, loc.shape)
    return jp.sum(normal + _tanh_log_det(u), axis=-1)


def mode(params: jax.Array, min_std: float) -> jax.Array:
    loc, _ = split(params, min_std)
    return jp.tanh(loc)


def postprocess(raw_action: jax.Array) -> jax.Array:
    return jp.tanh(raw_action)
