"""Linear (and small-MLP) probes of the true action latency from policy features (H6).

Rows are (episode, step) pairs; every split is by episode, so a probe never sees a test
episode's other steps (the latency is constant within an episode in the `lat` set, so a
step-level split would leak the target).

    rows = design(trace, rep, start_step, stride)         # X [n, f], y [n], groups [n]
    r2, pred = ridge_cv(rows, folds, inner_folds, alphas, rng)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import optax


@dataclasses.dataclass(frozen=True)
class Rows:
    x: np.ndarray  # [n, f] float64
    y: np.ndarray  # [n]
    groups: np.ndarray  # [n] episode index


def design(
    rep: np.ndarray, target: np.ndarray, alive: np.ndarray, start_step: int, stride: int
) -> Rows:
    """rep [E, T, f], target [E, T], alive [E, T] -> rows from alive steps t >= start_step,
    every `stride`-th step. Fallen steps are dropped: after a fall the state no longer
    reflects the controller, and "has fallen" alone would reveal high latency."""
    e, t = target.shape
    keep = np.zeros((e, t), bool)
    keep[:, start_step::stride] = True
    keep &= alive.astype(bool)
    ep = np.broadcast_to(np.arange(e)[:, None], (e, t))
    return Rows(
        x=np.asarray(rep, np.float64)[keep],
        y=np.asarray(target, np.float64)[keep],
        groups=ep[keep],
    )


def episode_folds(groups: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fold id per row; episodes are dealt round-robin to folds in a random order."""
    eps = np.unique(groups)
    fold_of = np.empty(eps.max() + 1, int)
    fold_of[rng.permutation(eps)] = np.arange(len(eps)) % k
    return fold_of[groups]


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst == 0 else 1.0 - float(np.sum((y - pred) ** 2)) / sst


def _standardize(xtr: np.ndarray, *xs: np.ndarray) -> list[np.ndarray]:
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)  # constant (e.g. dead) units -> 0 after centering
    return [(x - mu) / sd for x in (xtr, *xs)]


def _ridge_fit_predict(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alphas: Sequence[float]
) -> np.ndarray:
    """Predictions [len(alphas), n_te] from standardized-feature ridge (intercept unpenalized)."""
    xtr, xte = _standardize(xtr, xte)
    ym = ytr.mean()
    u, s, vt = np.linalg.svd(xtr, full_matrices=False)
    uty = u.T @ (ytr - ym)
    out = []
    for a in alphas:
        w = vt.T @ (s / (s**2 + a) * uty)
        out.append(xte @ w + ym)
    return np.stack(out)


def ridge_cv(
    rows: Rows, folds: int, inner_folds: int, alphas: Sequence[float], rng: np.random.Generator
) -> tuple[float, np.ndarray, list[float]]:
    """Nested episode-level CV. Returns (out-of-fold R², out-of-fold predictions, chosen
    alpha per outer fold). Alpha is chosen on the training episodes only."""
    fold = episode_folds(rows.groups, folds, rng)
    pred = np.empty_like(rows.y)
    chosen = []
    for k in range(folds):
        tr, te = fold != k, fold == k
        xtr, ytr, gtr = rows.x[tr], rows.y[tr], rows.groups[tr]
        inner = episode_folds(gtr, inner_folds, rng)
        sse = np.zeros(len(alphas))
        for j in range(inner_folds):
            a, b = inner != j, inner == j
            p = _ridge_fit_predict(xtr[a], ytr[a], xtr[b], alphas)
            sse += np.sum((p - ytr[b]) ** 2, axis=1)
        best = float(alphas[int(np.argmin(sse))])
        chosen.append(best)
        pred[te] = _ridge_fit_predict(xtr, ytr, rows.x[te], [best])[0]
    return r2(rows.y, pred), pred, chosen


def mlp_cv(rows: Rows, folds: int, cfg: dict[str, Any], rng: np.random.Generator) -> float:
    """Secondary probe (D6): 1 hidden layer (ReLU), full-batch AdamW, fixed settings,
    same kind of episode-level folds. Returns out-of-fold R²."""
    fold = episode_folds(rows.groups, folds, rng)
    pred = np.empty_like(rows.y)
    h, steps = int(cfg["hidden"]), int(cfg["steps"])
    opt = optax.adamw(float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    def init(key, f):
        k1, k2 = jax.random.split(key)
        return {
            "w1": jax.random.normal(k1, (f, h)) * jp.sqrt(2.0 / f),
            "b1": jp.zeros(h),
            "w2": jax.random.normal(k2, (h,)) * jp.sqrt(1.0 / h),
            "b2": jp.zeros(()),
        }

    def apply(p, x):
        return jax.nn.relu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]

    @jax.jit
    def train(key, x, y):
        p = init(key, x.shape[1])
        s = opt.init(p)

        def body(c, _):
            p, s = c
            g = jax.grad(lambda q: jp.mean((apply(q, x) - y) ** 2))(p)
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), None

        (p, _), _ = jax.lax.scan(body, (p, s), None, length=steps)
        return p

    for k in range(folds):
        tr, te = fold != k, fold == k
        xtr, xte = _standardize(rows.x[tr], rows.x[te])
        ym, ys = rows.y[tr].mean(), max(rows.y[tr].std(), 1e-8)
        key = jax.random.PRNGKey(int(rng.integers(2**31)))
        p = train(key, jp.asarray(xtr, jp.float32), jp.asarray((rows.y[tr] - ym) / ys, jp.float32))
        pred[te] = np.asarray(apply(p, jp.asarray(xte, jp.float32))) * ys + ym
    return r2(rows.y, pred)


def recompute(tr: Any, params: Any, norm: Any, obs: np.ndarray, rep: str) -> np.ndarray:
    """Re-run an actor (any params, e.g. a random init) over recorded observations.

    obs [E, T, 48] are the `obs["state"]` frames the policy saw; the history buffer and
    GRU state are rebuilt exactly as in the trace loop (`mjx_eval._trace_episodes`):
    hist_0 = init(obs_0), hist_t = push(hist_{t-1}, obs_t), h reset at t = 0.
    Not valid for condition F (its input needs the privileged latency, R4).
    Returns [E, T, f] for rep in {"features", "hidden"}.
    """
    e = obs.shape[0]
    o = jp.swapaxes(jp.asarray(obs, jp.float32), 0, 1)  # [T, E, 48]
    hist0 = jax.vmap(tr.inp.init)(o[0])

    def body(c, ot):
        hist, h, first = c
        hist = jnp_where_first(first, hist, jax.vmap(tr.inp.push)(hist, ot))
        x = jax.vmap(tr.inp.features)(hist, {})
        _, h, feats = tr.actor.step_features(params, tr._norm_policy(norm, x), h, first)
        out = h if rep == "hidden" else feats
        return (hist, h, jp.zeros(e, bool)), out

    @jax.jit
    def run(o):
        init = (hist0, tr.actor.initial_carry(e), jp.ones(e, bool))
        _, out = jax.lax.scan(body, init, o)
        return out

    return np.swapaxes(np.asarray(run(o)), 0, 1)


def jnp_where_first(first: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-env select along the leading axis: a where first, else b."""
    return jp.where(first.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
