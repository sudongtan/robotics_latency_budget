"""Running mean / variance observation normalizer (same estimator as Brax
`acme/running_statistics.py`: batched Welford update, std clipped to [1e-6, 1e6]).

Stats are updated from each rollout before the SGD epochs, as in Brax PPO, and saved in
every checkpoint so evaluation normalizes exactly as training did.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from flax import struct

_STD_MIN, _STD_MAX = 1e-6, 1e6


@struct.dataclass
class Stats:
    count: jax.Array  # scalar float32
    mean: jax.Array
    summed_var: jax.Array  # sum of squared deviations (M2)
    std: jax.Array


def init(size: int) -> Stats:
    z = jp.zeros(size, jp.float32)
    return Stats(count=jp.zeros((), jp.float32), mean=z, summed_var=z, std=jp.ones(size))


def update(stats: Stats, batch: jax.Array) -> Stats:
    """Merge a batch [..., size] into the running stats (Chan et al. parallel update)."""
    x = batch.reshape(-1, batch.shape[-1]).astype(jp.float32)
    n = jp.asarray(x.shape[0], jp.float32)
    count = stats.count + n
    delta = x - stats.mean
    mean = stats.mean + jp.sum(delta, axis=0) / count
    summed_var = stats.summed_var + jp.sum(delta * (x - mean), axis=0)
    std = jp.clip(jp.sqrt(summed_var / count), _STD_MIN, _STD_MAX)
    return Stats(count=count, mean=mean, summed_var=summed_var, std=std)


def normalize(stats: Stats, x: jax.Array) -> jax.Array:
    return (x - stats.mean) / stats.std
