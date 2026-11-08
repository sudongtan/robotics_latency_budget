"""Statistics for the pre-registered analysis (prereg §4–§6). Pure numpy, no JAX.

Conventions
- A *curve* is success rate (0–1) per test setting of one run, on an increasing grid `x`.
- Crossing latencies (L50, L90, L10): the smallest test latency at which success drops below
  the threshold, linearly interpolated between the last grid point at or above it and the
  first one below it. If success is already below at x[0], the crossing is x[0]. If it never
  drops below, the value is right-censored at the top of the grid (post-hoc clarification 1:
  enters statistics as that value, and the censoring is counted).
- Across seeds: interquartile mean (IQM = 25 % trimmed mean, as rliable) with a stratified
  bootstrap over seeds (one task, so "stratified" = resample the seeds of each condition with
  replacement, independently per condition), percentile CIs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as _st


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile mean: mean of the middle 50 % (scipy trim_mean 0.25, as rliable)."""
    return _st.trim_mean(np.asarray(x, dtype=float), 0.25, axis=axis)


@dataclass(frozen=True)
class Crossing:
    value: float
    censored: bool


def crossing(x: Sequence[float], s: Sequence[float], thr: float) -> Crossing:
    """Smallest x at which s < thr, linearly interpolated; censored at x[-1] if never."""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    below = np.flatnonzero(s < thr)
    if below.size == 0:
        return Crossing(float(x[-1]), True)
    i = int(below[0])
    if i == 0:
        return Crossing(float(x[0]), False)
    x0, x1, s0, s1 = x[i - 1], x[i], s[i - 1], s[i]
    # s0 >= thr > s1: point where the segment meets thr.
    return Crossing(float(x0 + (s0 - thr) / (s0 - s1) * (x1 - x0)), False)


def delay_margin(x: Sequence[float], s: Sequence[float], thr: float = 0.5) -> float:
    """Li & Sun: largest grid x with s >= thr there and at every smaller grid x.

    NaN if success is below thr already at x[0].
    """
    x = np.asarray(x, dtype=float)
    ok = np.asarray(s, dtype=float) >= thr
    if not ok[0]:
        return float("nan")
    n = int(np.argmin(ok)) if not ok.all() else len(ok)
    return float(x[n - 1])


def boot_stat(
    groups: Sequence[np.ndarray],
    fn: Callable[..., np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of fn(*resampled groups).

    Each group is an array with seeds on axis 0, resampled with replacement independently
    (stratified over conditions). fn receives arrays of shape (n_boot, n_seeds, ...).
    """
    res = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        res.append(g[idx])
    return np.asarray(fn(*res))


def percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(samples, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def iqm_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator, level: float) -> Estimate:
    """IQM over seeds (axis 0) with a bootstrap percentile CI."""
    x = np.asarray(x, dtype=float)
    b = boot_stat([x], lambda a: iqm(a, axis=1), n_boot, rng)
    return Estimate(float(iqm(x, axis=0)), *percentile_ci(b, level))


def iqm_diff_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator, level: float
) -> Estimate:
    """IQM(x) − IQM(y), independent conditions, bootstrap percentile CI."""
    b = boot_stat([x, y], lambda a, c: iqm(a, axis=1) - iqm(c, axis=1), n_boot, rng)
    return Estimate(float(iqm(x, axis=0) - iqm(y, axis=0)), *percentile_ci(b, level))


def iqm_ratio_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator, level: float
) -> Estimate:
    """IQM(x) / IQM(y), bootstrap percentile CI."""
    b = boot_stat([x, y], lambda a, c: iqm(a, axis=1) / iqm(c, axis=1), n_boot, rng)
    return Estimate(float(iqm(x, axis=0) / iqm(y, axis=0)), *percentile_ci(b, level))


def recovery_ci(
    err_a: np.ndarray,
    err_c: np.ndarray,
    err_d: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    level: float,
) -> Estimate:
    """H3: R = (err_C − err_D) / (err_C − err_A) on IQMs over seeds, bootstrap CI."""

    def r(a, c, d):
        ia, ic, id_ = iqm(a, axis=-1), iqm(c, axis=-1), iqm(d, axis=-1)
        return (ic - id_) / (ic - ia)

    b = boot_stat([err_a, err_c, err_d], r, n_boot, rng)
    return Estimate(
        float(r(np.asarray(err_a), np.asarray(err_c), np.asarray(err_d))), *percentile_ci(b, level)
    )


# ---------------------------------------------------------------------------- verdicts
SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
INCONCLUSIVE = "inconclusive"


def verdict(supported: bool, excluded: bool) -> str:
    """Prereg §6: "supported" if the rule holds; "not supported" if it fails with CIs that
    exclude the supporting region; otherwise "inconclusive"."""
    if supported:
        return SUPPORTED
    return NOT_SUPPORTED if excluded else INCONCLUSIVE
