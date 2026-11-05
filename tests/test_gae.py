"""GAE against a plain NumPy loop, incl. terminations and truncations inside a rollout."""

import jax.numpy as jp
import numpy as np
import pytest

from latrl.ppo.gae import advantages, gae


def gae_reference(r, v, nv, term, trunc, gamma, lam):
    """Textbook GAE, one env, explicit loop. Truncation keeps the bootstrap V(s')."""
    T = len(r)
    adv = np.zeros(T)
    acc = 0.0
    for t in reversed(range(T)):
        done = term[t] or trunc[t]
        delta = r[t] + gamma * (0.0 if term[t] else nv[t]) - v[t]
        acc = delta + gamma * lam * (0.0 if done else acc)
        adv[t] = acc
    return adv, adv + v


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    T, B, gamma, lam = 20, 7, 0.97, 0.95
    r = rng.normal(size=(T, B))
    v = rng.normal(size=(T, B))
    nv = np.concatenate([v[1:], rng.normal(size=(1, B))])  # V(s_{t+1}) if no reset
    term = rng.random((T, B)) < 0.1
    trunc = (rng.random((T, B)) < 0.1) & ~term
    done = term | trunc
    # After a reset the true next state differs from the next row's state.
    nv = np.where(done, rng.normal(size=(T, B)), nv)
    adv, vs = gae(*map(jp.asarray, (r, v, nv, term * 1.0, done * 1.0)), gamma, lam)
    for b in range(B):
        ra, rv = gae_reference(r[:, b], v[:, b], nv[:, b], term[:, b], trunc[:, b], gamma, lam)
        np.testing.assert_allclose(np.asarray(adv[:, b]), ra, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(vs[:, b]), rv, rtol=1e-5, atol=1e-5)


def test_termination_vs_truncation():
    """Same final transition: termination drops V(s'), truncation keeps it."""
    r, v, nv = jp.array([1.0]), jp.array([0.5]), jp.array([10.0])
    adv_term, _ = gae(r, v, nv, jp.array([1.0]), jp.array([1.0]), 0.9, 0.95)
    adv_trunc, _ = gae(r, v, nv, jp.array([0.0]), jp.array([1.0]), 0.9, 0.95)
    assert float(adv_term[0]) == pytest.approx(1.0 - 0.5)
    assert float(adv_trunc[0]) == pytest.approx(1.0 + 0.9 * 10.0 - 0.5)


def test_no_flow_across_done():
    """Rewards after a done never change advantages before it."""
    T = 10
    v = jp.zeros(T)
    term = jp.zeros(T).at[4].set(1.0)
    r1 = jp.zeros(T)
    r2 = r1.at[5:].set(100.0)
    a1, _ = gae(r1, v, v, term, term, 0.97, 0.95)
    a2, _ = gae(r2, v, v, term, term, 0.97, 0.95)
    np.testing.assert_array_equal(np.asarray(a1[:5]), np.asarray(a2[:5]))


def test_matches_brax_without_resets():
    """With no dones, our `brax` estimator equals Brax's compute_gae (brax 0.14.2): same value
    targets, same policy advantages."""
    from brax.training.agents.ppo.losses import compute_gae

    rng = np.random.default_rng(0)
    T, B = 20, 5
    r, v = rng.normal(size=(T, B)), rng.normal(size=(T, B))
    boot = rng.normal(size=B)
    nv = np.concatenate([v[1:], boot[None]])
    z = np.zeros((T, B))
    vs_b, adv_b = compute_gae(z, z, r, v, boot, lambda_=0.95, discount=0.97)
    adv, vs = advantages("brax", r, v, nv, z, z, 0.97, 0.95)
    np.testing.assert_allclose(np.asarray(vs), np.asarray(vs_b), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(adv), np.asarray(adv_b), rtol=1e-5, atol=1e-5)


def brax_adv_reference(r, v, nv, term, trunc, gamma, lam):
    """Brax policy advantage, explicit loop: r + γ·(next λ-return) − V; the next λ-return
    is V(s') at a done or at the rollout end."""
    _, vs = gae_reference(r, v, nv, term, trunc, gamma, lam)
    T = len(r)
    adv = np.zeros(T)
    for t in range(T):
        done = term[t] or trunc[t]
        nxt = nv[t] if (done or t == T - 1) else vs[t + 1]
        adv[t] = r[t] + gamma * (0.0 if term[t] else nxt) - v[t]
    return adv, vs


@pytest.mark.parametrize("seed", range(3))
def test_brax_estimator_with_resets(seed):
    rng = np.random.default_rng(100 + seed)
    T, B, gamma, lam = 20, 6, 0.97, 0.95
    r, v, nv = (rng.normal(size=(T, B)) for _ in range(3))
    term = rng.random((T, B)) < 0.1
    trunc = (rng.random((T, B)) < 0.1) & ~term
    done = term | trunc
    adv, vs = advantages("brax", *map(jp.asarray, (r, v, nv, term * 1.0, done * 1.0)), gamma, lam)
    for b in range(B):
        ra, rv = brax_adv_reference(r[:, b], v[:, b], nv[:, b], term[:, b], trunc[:, b], gamma, lam)
        np.testing.assert_allclose(np.asarray(adv[:, b]), ra, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(vs[:, b]), rv, rtol=1e-5, atol=1e-5)
