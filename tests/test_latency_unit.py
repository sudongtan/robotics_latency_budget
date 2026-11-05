"""Delay-line semantics, independent of the robot (plan §6, D14)."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from latrl.envs import latency

N_SUB = 5


def _run_action_line(delays, n_sub=N_SUB, max_delay=None):
    """Send action t (value = t) with delay delays[t]; return, per global substep, the
    send index of the item in use (-1 = initial fill)."""
    max_delay = max(delays) if max_delay is None else max_delay
    k = latency.action_buffer_len(max_delay, n_sub)
    line = latency.init(jp.array(-1.0), k)
    used = []
    for t, d in enumerate(delays):
        line = latency.push(line, jp.array(float(t)), d, elapsed=n_sub)
        for i in range(n_sub):
            used.append(int(latency.read(line, i)))
    return np.array(used)


@pytest.mark.parametrize("d", [0, 1, 4, 5, 6, 10, 13, 20, 30])
def test_constant_action_delay_matches_definition(d):
    """§6: with d = q·n_sub + r, at substep i the command is from q + (i < r) steps ago."""
    T = 12
    used = _run_action_line([d] * T)
    q, r = divmod(d, N_SUB)
    for t in range(T):
        for i in range(N_SUB):
            src = t - q - (1 if i < r else 0)
            assert used[t * N_SUB + i] == (src if src >= 0 else -1), (t, i)


@pytest.mark.parametrize("d", [0, 3, 7, 12])
def test_impulse_first_appears_at_d(d):
    """An action sent at substep 0 of step t0 is first used at global substep t0·n + d."""
    t0, T = 2, 8
    used = _run_action_line([d] * T)
    first = int(np.argmax(used == t0))
    assert first == t0 * N_SUB + d


def test_jitter_newest_arrived_never_reverts():
    """D14: under random per-action delays, the item in use is the newest arrived one and
    its send index never decreases; nothing is used before it arrives (R2)."""
    rng = np.random.default_rng(0)
    max_delay = 30
    for _ in range(20):
        delays = list(rng.integers(0, max_delay + 1, size=40))
        used = _run_action_line(delays, max_delay=max_delay)
        assert np.all(np.diff(used) >= 0)
        for g, s in enumerate(used):
            arrived = [t for t, d in enumerate(delays) if t * N_SUB + d <= g]
            assert s == (max(arrived) if arrived else -1)


def test_obs_line_constant_and_jitter():
    rng = np.random.default_rng(1)
    max_delay = 6
    for constant in [0, 1, 3, 6, None]:
        delays = [constant] * 30 if constant is not None else list(rng.integers(0, 7, 30))
        line = latency.init(jp.array(-1.0), latency.obs_buffer_len(max_delay))
        seen = []
        for t, d in enumerate(delays):
            line = latency.push(line, jp.array(float(t)), d, elapsed=1)
            seen.append(int(latency.read(line, 0)))
        assert np.all(np.diff(seen) >= 0)
        for t, s in enumerate(seen):
            arrived = [u for u, d in enumerate(delays[: t + 1]) if u + d <= t]
            assert s == (max(arrived) if arrived else -1)
            if constant is not None:
                assert s == (t - constant if t >= constant else -1)


def test_buffer_never_falls_back():
    """Within the configured max delay the newest-arrived slot always exists, so the
    fallback in `newest_arrived` is never hit (it would silently add delay)."""
    rng = np.random.default_rng(2)
    max_delay = 30
    k = latency.action_buffer_len(max_delay, N_SUB)
    line = latency.init(jp.zeros(()), k)
    line = line.replace(arrival=jp.full((k,), 10**6, jp.int32))  # nothing pre-arrived
    for t, d in enumerate(rng.integers(0, max_delay + 1, 200)):
        line = latency.push(line, jp.zeros(()), int(d), elapsed=N_SUB)
        if t * N_SUB >= max_delay:  # past the start-up window
            for i in range(N_SUB):
                assert bool(jp.any(line.arrival <= i))


def test_jitter_offset_bounds_and_static_off():
    key = jax.random.PRNGKey(0)
    off = jp.zeros((), jp.int32)
    assert latency.jitter_offset(key, off, 0.0, 3) is off
    vals = []
    for k in jax.random.split(key, 300):
        off = latency.jitter_offset(k, off, 0.5, 3)
        vals.append(int(off))
    assert min(vals) == -3 and max(vals) == 3
    assert int(latency.effective(jp.array(1), jp.array(-3), 30)) == 0
    assert int(latency.effective(jp.array(29), jp.array(3), 30)) == 30
