"""Latency probes (H6): episode-level splits, ridge/MLP probes on synthetic data, and the
feature recomputation used for the random-network control (checked against saved traces)."""

import pathlib

import numpy as np
import pytest

from latrl.analysis import probing as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALPHAS = [1e-3, 1e-1, 1e1, 1e3]


def _synthetic(signal: float, e=60, t=40, f=16, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.integers(0, 21, size=e).astype(float)
    target = np.broadcast_to(lat[:, None], (e, t))
    rep = rng.normal(size=(e, t, f))
    rep[..., 0] += signal * lat[:, None]
    return rep, target, np.ones((e, t), bool)


def test_folds_never_split_an_episode():
    groups = np.repeat(np.arange(37), 11)
    fold = P.episode_folds(groups, 5, np.random.default_rng(0))
    for g in range(37):
        assert len(np.unique(fold[groups == g])) == 1
    sizes = np.bincount([fold[groups == g][0] for g in range(37)])
    assert sizes.max() - sizes.min() <= 1


def test_design_drops_fallen_and_early_steps():
    rep, target, alive = _synthetic(1.0, e=3, t=20, f=2)
    alive = alive.copy()
    alive[1, 10:] = False
    rows = P.design(rep, target, alive, start_step=5, stride=5)
    # episodes 0 and 2: t = 5, 10, 15; episode 1: t = 5 only
    assert len(rows.y) == 7 and np.sum(rows.groups == 1) == 1


def test_ridge_recovers_linear_code_and_not_noise():
    rows = P.design(*_synthetic(1.0), start_step=0, stride=1)
    r2, _, _ = P.ridge_cv(rows, 5, 4, ALPHAS, np.random.default_rng(0))
    assert r2 > 0.95
    rows = P.design(*_synthetic(0.0), start_step=0, stride=1)
    r2, _, _ = P.ridge_cv(rows, 5, 4, ALPHAS, np.random.default_rng(0))
    assert r2 < 0.05


def test_episode_split_blocks_per_episode_memorisation():
    """A feature that is a random per-episode constant (an episode 'ID') predicts the
    target perfectly under a step-level split but not under the episode-level one."""
    e, t = 60, 30
    rng = np.random.default_rng(1)
    target = np.broadcast_to(rng.integers(0, 21, size=e)[:, None], (e, t)).astype(float)
    rep = np.broadcast_to(np.eye(e)[:, None, :], (e, t, e)).copy()
    rows = P.design(rep, target, np.ones((e, t), bool), 0, 1)
    r2, _, _ = P.ridge_cv(rows, 5, 4, ALPHAS, np.random.default_rng(0))
    assert r2 < 0.1


def test_mlp_probe_finds_nonlinear_code():
    rep, target, alive = _synthetic(0.0, f=4)
    theta = target / 21 * 2 * np.pi  # latency on a full circle: invertible, not linear
    rep[..., 0], rep[..., 1] = np.cos(theta), np.sin(theta)
    rows = P.design(rep, target, alive, 0, 2)
    cfg = {"hidden": 64, "steps": 1500, "lr": 3e-3, "weight_decay": 0.0}
    r2_mlp = P.mlp_cv(rows, 5, cfg, np.random.default_rng(0))
    r2_lin, _, _ = P.ridge_cv(rows, 5, 4, ALPHAS, np.random.default_rng(0))
    assert r2_mlp > r2_lin + 0.3


@pytest.mark.parametrize(
    "run,rep", [("E_gru_s0_4a4a467", "hidden"), ("D_hist4_s0_4a4a467", "features")]
)
def test_recompute_reproduces_saved_traces(run, rep):
    """The random-network control is only fair if `recompute` rebuilds the actor input
    exactly as the trace loop did: with the trained params it must reproduce the saved
    representation (float16 storage of obs and outputs bounds the agreement)."""
    run_dir = ROOT / "runs" / run
    if not (run_dir / "traces" / "lat.npz").exists():
        pytest.skip(f"{run} traces not on this machine")
    from latrl.eval.load import load_run

    _, _, tr, ts = load_run(run_dir)
    z = np.load(run_dir / "traces" / "lat.npz")
    e, t = 4, 60
    got = P.recompute(tr, ts.params["actor"], ts.norm, z["obs"][:e, :t], rep)
    want = z[rep][:e, :t].astype(np.float64)
    alive = z["alive"][:e, :t]
    err = np.abs(got - want)[alive]
    scale = np.abs(want[alive]).max()
    assert err.max() < 0.02 * scale, (err.max(), scale)
