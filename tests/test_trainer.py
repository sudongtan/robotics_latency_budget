"""Training-loop tests on a tiny config: latency resampled on reset (F16), the loss replays
the rollout exactly (incl. GRU hidden-state resets at real episode ends), one trace."""

import functools

import jax
import jax.numpy as jp
import numpy as np
import pytest
from conftest import ENV_CFG
from omegaconf import OmegaConf

from latrl.envs.go1 import make_env
from latrl.policies import distribution as dist
from latrl.ppo.trainer import Trainer

CFG = ENV_CFG.parents[1]
N, T, EP = 4, 10, 7  # episodes of 7 steps force truncations inside each 10-step unroll


@functools.cache
def _trainer(policy: str):
    env_cfg = OmegaConf.load(ENV_CFG)
    env_cfg.latency.action.hi = 20  # condition C range
    env_cfg.noise_level = 0.0
    ppo = OmegaConf.load(CFG / "ppo" / "default.yaml")
    ppo.num_envs, ppo.unroll_length, ppo.num_minibatches, ppo.episode_length = N, T, 2, EP
    pol = OmegaConf.load(CFG / "policy" / f"{policy}.yaml")
    env = _env(OmegaConf.to_yaml(env_cfg))
    tr = Trainer(env, env_cfg, ppo, pol)
    models, in_axes = tr.models(jax.random.PRNGKey(10), N)
    ts, carry = jax.jit(lambda k: tr.init(k, models, in_axes))(jax.random.PRNGKey(0))
    return tr, models, in_axes, ts, carry


@functools.cache
def _env(cfg_yaml: str):
    return make_env(OmegaConf.create(cfg_yaml))


def _rollout(tr, models, in_axes, ts, carry, steps, key):
    f = jax.jit(lambda c, k: tr.rollout_step(ts, models, in_axes, c, k))
    out = []
    for k in jax.random.split(key, steps):
        carry, (trans, _) = f(carry, k)
        out.append((carry, trans))
    return out


def test_latency_resampled_on_reset():
    """F16: after every episode end the env is fully reset, so per-episode latency changes."""
    tr, models, in_axes, ts, carry = _trainer("mlp")
    seen = [np.asarray(carry.env_state.info["latency"].action_base)]
    dones = []
    for c, trans in _rollout(tr, models, in_axes, ts, carry, 3 * EP, jax.random.PRNGKey(1)):
        seen.append(np.asarray(c.env_state.info["latency"].action_base))
        dones.append(np.asarray(trans.done) > 0)
    seen, dones = np.stack(seen), np.stack(dones)
    assert dones.sum() >= 2 * N  # every env finished at least twice
    for t in range(len(dones)):
        same = seen[t + 1] == seen[t]
        assert same[~dones[t]].all(), "latency must stay fixed within an episode"
    # Across episodes it is resampled from U{0..20}: with 3+ episodes per env, some change.
    assert (seen[1:][dones] != seen[:-1][dones]).any()
    assert seen.min() >= 0 and seen.max() <= 20


def test_reset_clears_history_hidden_and_counters():
    tr, models, in_axes, ts, carry = _trainer("history")
    steps = _rollout(tr, models, in_axes, ts, carry, EP, jax.random.PRNGKey(2))
    c, trans = steps[EP - 1]
    assert (np.asarray(trans.done) > 0).all()  # all truncated at EP
    assert c.is_first.all() and (np.asarray(c.ep_steps) == 0).all()
    hist = np.asarray(c.hist)
    np.testing.assert_array_equal(hist, np.repeat(hist[:, :1], hist.shape[1], axis=1))
    np.testing.assert_array_equal(hist[:, 0], np.asarray(c.env_state.obs["state"]))


@pytest.mark.parametrize("policy", ["mlp", "gru"])
def test_loss_replays_rollout_log_probs(policy):
    """With unchanged params/normalizer, the loss's sequence replay (from the stored initial
    hidden state, resetting at is_first) reproduces the behaviour log-probs exactly, across
    real episode boundaries inside the unroll."""
    tr, models, in_axes, ts, carry = _trainer(policy)
    h0 = carry.h
    steps = _rollout(tr, models, in_axes, ts, carry, T, jax.random.PRNGKey(3))
    data = jax.tree.map(lambda *x: jp.stack(x, axis=1), *[s[1] for s in steps])  # [N, T]
    assert (np.asarray(data.is_first)[:, 1:]).any()  # a boundary inside the sequence
    dparams = tr.actor.sequence(
        ts.params["actor"], tr._norm_policy(ts.norm, data.x), h0, data.is_first
    )
    logp = dist.log_prob(dparams, data.raw_action, tr.cfg.min_std)
    np.testing.assert_allclose(np.asarray(logp), np.asarray(data.log_prob), rtol=1e-4, atol=1e-4)


def test_iteration_traces_once_and_learns_something():
    tr, models, in_axes, ts, carry = _trainer("gru")
    traces = 0

    def it(ts, carry):
        nonlocal traces
        traces += 1
        return tr.iteration(ts, carry, models, in_axes)

    f = jax.jit(it)
    p0 = ts.params
    for _ in range(3):
        ts, carry, m = f(ts, carry)
    assert traces == 1
    assert all(np.isfinite(float(v)) for k, v in m.items() if not k.startswith("rollout/ep"))
    moved = jax.tree.map(lambda a, b: float(jp.max(jp.abs(a - b))), p0, ts.params)
    assert max(jax.tree.leaves(moved)) > 0.0
    assert float(ts.norm.policy.count) == 3 * N * T
