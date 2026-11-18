"""Follow-up F1 mechanisms: the critic's optional action history (C_hcrit) and fine-tuning
from a finished run (C_ft). The pre-registered critic (action_history = 0) is unchanged."""

import functools
import importlib.util
import json
import pathlib

import jax
import jax.numpy as jp
import numpy as np
import pytest
from conftest import ENV_CFG
from omegaconf import OmegaConf

from latrl.envs.go1 import make_env
from latrl.envs.specs import NUM_ACTIONS, PRIVILEGED_OBS_SIZE
from latrl.ppo import checkpoint
from latrl.ppo.trainer import Trainer

ROOT = ENV_CFG.parents[2]
CFG = ROOT / "configs"
N = 4


@functools.cache
def _env():
    env_cfg = OmegaConf.load(ENV_CFG)
    env_cfg.latency.action.hi = 20  # condition C range; buffer sized by latency.action.max
    env_cfg.noise_level = 0.0
    return env_cfg, make_env(env_cfg)


def _trainer(k: int) -> Trainer:
    env_cfg, env = _env()
    ppo = OmegaConf.load(CFG / "ppo" / "default.yaml")
    ppo.num_envs, ppo.num_minibatches = N, 2
    pol = OmegaConf.load(CFG / "policy" / "mlp.yaml")
    return Trainer(env, env_cfg, ppo, pol, {"action_history": k})


def test_default_critic_is_unchanged():
    tr = _trainer(0)
    assert tr.critic_size == PRIVILEGED_OBS_SIZE
    models, in_axes = tr.models(jax.random.PRNGKey(1), N)
    s = tr.env_reset(models, in_axes, jax.random.split(jax.random.PRNGKey(2), N))
    np.testing.assert_array_equal(tr.critic_input(s), s.obs["privileged_state"])


def test_critic_history_holds_the_previous_actions():
    """After step t the critic sees privileged_state (incl. a_t as last action) followed by
    a_{t-1}, ..., a_{t-5}; slots before the episode start hold the reset value (zeros)."""
    k = 5
    tr = _trainer(k)
    assert tr.critic_size == PRIVILEGED_OBS_SIZE + k * NUM_ACTIONS
    assert tr.inp.size == _trainer(0).inp.size  # the actor input is untouched (R4)
    models, in_axes = tr.models(jax.random.PRNGKey(1), N)
    s = tr.env_reset(models, in_axes, jax.random.split(jax.random.PRNGKey(2), N))
    step = jax.jit(lambda s, a: tr.env_step(models, in_axes, s, a))
    rng = np.random.default_rng(0)
    sent = []
    for t in range(8):
        a = rng.uniform(-0.3, 0.3, (N, NUM_ACTIONS)).astype(np.float32)
        s = step(s, jp.asarray(a))
        sent.append(a)
        x = np.asarray(tr.critic_input(s))
        np.testing.assert_array_equal(x[:, :PRIVILEGED_OBS_SIZE], s.obs["privileged_state"])
        past = x[:, PRIVILEGED_OBS_SIZE:].reshape(N, k, NUM_ACTIONS)
        for j in range(k):
            want = sent[t - 1 - j] if t - 1 - j >= 0 else np.zeros((N, NUM_ACTIONS))
            np.testing.assert_allclose(past[:, j], want, atol=1e-6)


def test_critic_history_longer_than_the_delay_line_is_rejected():
    env_cfg, env = _env()
    with pytest.raises(ValueError, match="action_history"):
        _trainer(env._k_act)


def _train_module():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_run(tmp: pathlib.Path, tr: Trainer, key: int, cfg) -> tuple[pathlib.Path, object]:
    models, in_axes = tr.models(jax.random.PRNGKey(1), N)
    ts, _ = tr.init(jax.random.PRNGKey(key), models, in_axes)
    run = tmp / f"src{key}"
    checkpoint.save(run / "checkpoints" / "it000001.msgpack", ts)
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (run / "meta.json").write_text(json.dumps({"commit": "abc1234"}))
    (run / "DONE").write_text("")
    return run, ts


def test_init_from_loads_params_and_normalizer(tmp_path):
    train = _train_module()
    tr = _trainer(0)
    cfg = OmegaConf.create(
        {"policy": OmegaConf.load(CFG / "policy" / "mlp.yaml"), "critic": {"action_history": 0}}
    )
    src, src_ts = _fake_run(tmp_path, tr, 7, cfg)
    models, in_axes = tr.models(jax.random.PRNGKey(1), N)
    fresh, _ = tr.init(jax.random.PRNGKey(8), models, in_axes)
    cfg.init_from = str(src)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    ts = train._init_from(fresh, tr, cfg, out_dir)
    for a, b in zip(jax.tree.leaves(ts.params), jax.tree.leaves(src_ts.params), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(ts.norm), jax.tree.leaves(src_ts.norm), strict=True):
        np.testing.assert_array_equal(a, b)
    assert ts.key is fresh.key  # the run's own RNG stream (R9)
    rec = json.loads((out_dir / "init_from.json").read_text())
    assert rec["checkpoint"] == "it000001.msgpack" and len(rec["sha256"]) == 64


def test_init_from_rejects_mismatched_source(tmp_path):
    train = _train_module()
    tr = _trainer(0)
    hist = OmegaConf.load(CFG / "policy" / "history.yaml")
    cfg_src = OmegaConf.create({"policy": hist, "critic": {"action_history": 0}})
    src, _ = _fake_run(tmp_path, tr, 7, cfg_src)
    cfg = OmegaConf.create(
        {"policy": OmegaConf.load(CFG / "policy" / "mlp.yaml"), "init_from": str(src)}
    )
    models, in_axes = tr.models(jax.random.PRNGKey(1), N)
    fresh, _ = tr.init(jax.random.PRNGKey(8), models, in_axes)
    with pytest.raises(ValueError, match="policy"):
        train._init_from(fresh, tr, cfg, tmp_path)
    (src / "DONE").unlink()
    with pytest.raises(FileNotFoundError):
        train._init_from(fresh, tr, cfg, tmp_path)
