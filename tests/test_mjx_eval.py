"""In-job MJX eval on the tiny smoke grid: structure, paired cells, latency applied, falls,
traces."""

import functools

import jax
import numpy as np
from conftest import ENV_CFG
from omegaconf import OmegaConf

from latrl.envs.go1 import make_env
from latrl.eval.mjx_eval import Evaluator, grid
from latrl.ppo.trainer import Trainer

CFG = ENV_CFG.parents[1]


def _ecfg(**over):
    cfg = OmegaConf.merge(
        OmegaConf.load(CFG / "mjx_eval" / "full.yaml"),
        OmegaConf.load(CFG / "mjx_eval" / "smoke.yaml"),
    )
    cfg.pop("defaults", None)
    for k, v in over.items():
        OmegaConf.update(cfg, k, v)
    return cfg


@functools.cache
def _setup(policy="gru"):
    env_cfg = OmegaConf.load(ENV_CFG)
    env_cfg.latency.action.hi = 20
    ppo = OmegaConf.load(CFG / "ppo" / "default.yaml")
    ppo.num_envs, ppo.num_minibatches = 4, 2
    pol = OmegaConf.load(CFG / "policy" / f"{policy}.yaml")
    tr = Trainer(make_env(env_cfg), env_cfg, ppo, pol)
    models, in_axes = tr.models(jax.random.PRNGKey(1), 4)
    ts, _ = jax.jit(lambda k: tr.init(k, models, in_axes))(jax.random.PRNGKey(0))
    return tr, env_cfg, ts


def test_grid_counts():
    full = OmegaConf.load(CFG / "mjx_eval" / "full.yaml")
    cells = grid(full)
    axes = [c.axis for c in cells]
    assert axes.count("action") == axes.count("action_jitter") == 31  # 0–120 ms every 4 ms
    assert axes.count("obs") == 7 and axes.count("tm") == 13


@functools.cache
def _rows():
    tr, env_cfg, ts = _setup()
    return Evaluator(tr, env_cfg, _ecfg()).run(ts, jax.random.PRNGKey(5), run_id="t")


def test_rows_and_columns():
    df = _rows()
    cells = grid(_ecfg())
    assert len(df) == len(cells) * 4
    for col in (
        "fell",
        "return",
        "lin_err",
        "yaw_err",
        "cot",
        "jerk",
        "duty_factor",
        "step_freq_hz",
        "apex_m",
        "stance_width_m",
        "action_delay_ms",
    ):
        assert col in df and not np.isinf(df[col].astype(float)).any(), col
    ok = df.metric_steps > 0
    assert np.isfinite(df.loc[ok, ["lin_err", "jerk", "duty_factor"]].to_numpy()).all()
    assert ((df.duty_factor >= 0) & (df.duty_factor <= 1)).all()


def test_identical_settings_give_identical_episodes():
    """action d=0, obs d=0 and T_m=0 cells are the same setting: same keys, same models →
    identical per-episode results (common random numbers across cells)."""
    df = _rows()
    cols = ["return", "lin_err", "jerk", "duty_factor", "fall_step"]
    a = df[(df.axis == "action") & (df.action_delay == 0)][cols].to_numpy()
    o = df[(df.axis == "obs") & (df.obs_delay == 0)][cols].to_numpy()
    t = df[(df.axis == "tm") & (df.t_m_s == 0)][cols].to_numpy()
    np.testing.assert_array_equal(a, o)
    np.testing.assert_array_equal(a, t)


def test_latency_is_applied():
    df = _rows()
    r0 = df[(df.axis == "action") & (df.action_delay == 0)]["jerk"].to_numpy()
    r2 = df[(df.axis == "action") & (df.action_delay == 2)]["jerk"].to_numpy()
    assert not np.allclose(r0, r2)


def test_fall_detection_threshold():
    tr, env_cfg, ts = _setup()
    df = Evaluator(tr, env_cfg, _ecfg(**{"fall.min_height": 0.5})).run(ts, jax.random.PRNGKey(5))
    assert df.fell.all() and (df.fall_step == 0).all() and (df.metric_steps == 0).all()
    assert df[["lin_err", "jerk", "cot", "apex_m"]].isna().all().all()


def test_traces_shapes_and_latency():
    tr, env_cfg, ts = _setup()
    ev = Evaluator(tr, env_cfg, _ecfg())
    rec = ev.traces(jax.random.PRNGKey(7))
    assert set(rec) == {"lat", "lat_dyn", "lat_jitter"}
    t = rec["lat"](ts)
    n, T = 2, 40
    assert t["obs"].shape == (n, T, 48) and t["features"].shape == (n, T, 128)
    assert t["hidden"].shape == (n, T, tr.actor.carry_size)
    assert t["qpos"].shape == (n, T, 19) and t["contact"].shape == (n, T, 4)
    d = t["action_delay"]
    assert ((d >= 0) & (d <= 20)).all() and (d == d[:, :1]).all()  # no jitter: constant per ep
    tj = rec["lat_jitter"](ts)
    np.testing.assert_array_equal(tj["action_delay"][:, 0] >= 0, True)
