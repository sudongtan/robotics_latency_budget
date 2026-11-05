"""Delay lines for action and observation latency. Pure JAX, lives in env state (R3).

A `DelayLine` is a fixed-size shift register holding the K most recent items (newest at
index 0). Each item carries its arrival time at the receiver, relative to the receiver's
current clock. The receiver always uses the **newest item that has arrived** and never
goes back to an older one (D14), which is what a real motor driver or state estimator
does when messages have variable delay:

- constant delay d: the item sent at time s is used from s + d until s + 1 + d;
- delay grows: the receiver holds its current item longer;
- delay shrinks: it skips ahead; it never reverts to an older item.

Clock units are the caller's: substeps for actions, control steps for observations.

Buffer length. With delays ≤ D, the item in use is at most ⌊D / step⌋ + 1 sends old, so
K = ⌊D / step⌋ + 2 always contains it (for observations, step = 1 and the item in use is at
most D old, so K = D + 1 suffices). `read` falls back to the oldest slot if nothing has
arrived, which cannot happen within these bounds; the fallback is never the newest slot,
so a bug here could only add delay, never leak the future (R2).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from flax import struct

# Arrival time of the slots that pre-fill the buffer at reset: "arrived long ago".
_LONG_AGO = -(2**20)


@struct.dataclass
class DelayLine:
    values: jax.Array  # [K, ...] newest first
    arrival: jax.Array  # [K] int32, arrival time relative to the current clock


def action_buffer_len(max_delay_substeps: int, n_substeps: int) -> int:
    return max_delay_substeps // n_substeps + 2


def obs_buffer_len(max_delay_steps: int) -> int:
    return max_delay_steps + 1


def init(value: jax.Array, length: int) -> DelayLine:
    """Buffer pre-filled with `value`, all slots already arrived (plan §6: initial obs /
    default-pose action during the first steps of an episode)."""
    values = jp.broadcast_to(value, (length, *value.shape))
    return DelayLine(values=values, arrival=jp.full((length,), _LONG_AGO, dtype=jp.int32))


def push(line: DelayLine, value: jax.Array, delay: jax.Array, elapsed: int) -> DelayLine:
    """Advance the clock by `elapsed`, then send `value` now; it arrives after `delay`."""
    arrival = jp.maximum(line.arrival - elapsed, _LONG_AGO)
    arrival = jp.roll(arrival, 1).at[0].set(jp.asarray(delay, dtype=jp.int32))
    values = jp.roll(line.values, 1, axis=0).at[0].set(value)
    return DelayLine(values=values, arrival=arrival)


def newest_arrived(line: DelayLine, now: jax.Array | int) -> jax.Array:
    """Index (= age in sends) of the newest item with arrival ≤ now."""
    arrived = line.arrival <= now
    first = jp.argmax(arrived)  # first True = newest arrived
    return jp.where(jp.any(arrived), first, line.arrival.shape[0] - 1)


def read(line: DelayLine, now: jax.Array | int) -> jax.Array:
    return line.values[newest_arrived(line, now)]


def sample_base(key: jax.Array, lo: int, hi: int) -> jax.Array:
    """Per-episode delay ~ U{lo..hi} (inclusive)."""
    return jax.random.randint(key, (), lo, hi + 1, dtype=jp.int32)


def jitter_offset(key: jax.Array, offset: jax.Array, p: float, width: int) -> jax.Array:
    """With probability p, resample the jitter offset from U{-width..width}."""
    if p == 0.0 or width == 0:  # static config: no jitter, no extra RNG ops
        return offset
    k_resample, k_value = jax.random.split(key)
    new = jax.random.randint(k_value, (), -width, width + 1, dtype=jp.int32)
    return jp.where(jax.random.bernoulli(k_resample, p), new, offset)


def effective(base: jax.Array, offset: jax.Array, max_delay: int) -> jax.Array:
    return jp.clip(base + offset, 0, max_delay).astype(jp.int32)
