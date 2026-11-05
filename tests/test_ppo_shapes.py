"""Actor/critic shapes, tanh-normal distribution, normalizer, and GRU episode boundaries."""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from conftest import ENV_CFG
from omegaconf import OmegaConf

from latrl.envs.specs import NUM_ACTIONS, PRIVILEGED_OBS_SIZE
from latrl.policies import distribution as dist
from latrl.policies.inputs import PolicyInput
from latrl.policies.networks import Critic, num_params
from latrl.ppo import normalizer
from latrl.ppo.trainer import make_actor

POLICY_DIR = ENV_CFG.parents[1] / "policy"
POLICIES = ["mlp", "history", "gru", "privileged"]


def _actor(name):
    pcfg = OmegaConf.load(POLICY_DIR / f"{name}.yaml")
    inp = PolicyInput.from_config(pcfg, OmegaConf.load(ENV_CFG))
    return make_actor(pcfg, inp.size), inp


@pytest.mark.parametrize("name", POLICIES)
def test_actor_shapes(name):
    actor, inp = _actor(name)
    params = actor.init(jax.random.PRNGKey(0))
    B, T = 3, 5
    x = jp.ones((B, inp.size))
    out, h = actor.step(params, x, actor.initial_carry(B), jp.ones(B, bool))
    assert out.shape == (B, 2 * NUM_ACTIONS)
    assert h.shape == (B, actor.carry_size)
    seq = actor.sequence(params, jp.ones((B, T, inp.size)), h, jp.zeros((B, T), bool))
    assert seq.shape == (B, T, 2 * NUM_ACTIONS)


def test_param_counts_matched():
    """Plan §7: MLP, history and GRU actors within ±10 % of each other."""
    counts = {n: num_params(_actor(n)[0].init(jax.random.PRNGKey(0))) for n in POLICIES}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo <= 1.10, counts


def test_critic_shape():
    c = Critic(PRIVILEGED_OBS_SIZE, (512, 256, 128))
    p = c.init(jax.random.PRNGKey(0))
    assert c(p, jp.ones((4, 6, PRIVILEGED_OBS_SIZE))).shape == (4, 6)


def _gru():
    actor, inp = _actor("gru")
    return actor, inp, actor.init(jax.random.PRNGKey(1))


def test_gru_sequence_equals_stepping():
    """The loss's sequence replay reproduces the rollout's step-by-step outputs."""
    actor, inp, params = _gru()
    B, T = 4, 12
    x = jax.random.normal(jax.random.PRNGKey(2), (B, T, inp.size))
    first = jax.random.uniform(jax.random.PRNGKey(3), (B, T)) < 0.2
    h0 = jax.random.normal(jax.random.PRNGKey(4), (B, actor.carry_size))
    seq = actor.sequence(params, x, h0, first)
    h = h0
    for t in range(T):
        out, h = actor.step(params, x[:, t], h, first[:, t])
        np.testing.assert_allclose(np.asarray(out), np.asarray(seq[:, t]), atol=1e-5)


def test_gru_resets_exactly_at_episode_start():
    """Outputs from an episode-start step onward do not depend on anything before it:
    not on h0, not on earlier inputs. Before it, they do."""
    actor, inp, params = _gru()
    T, k = 10, 6  # episode boundary: step k is the first step of a new episode
    first = jp.zeros((1, T), bool).at[0, k].set(True)
    x = jax.random.normal(jax.random.PRNGKey(5), (1, T, inp.size))
    x2 = x.at[:, :k].set(jax.random.normal(jax.random.PRNGKey(6), (1, k, inp.size)))
    h0, h0b = jp.zeros((1, actor.carry_size)), jp.ones((1, actor.carry_size))
    a = actor.sequence(params, x, h0, first)
    b = actor.sequence(params, x2, h0b, first)
    np.testing.assert_array_equal(np.asarray(a[:, k:]), np.asarray(b[:, k:]))
    assert not np.allclose(np.asarray(a[:, k - 1]), np.asarray(b[:, k - 1]))
    # A fresh sequence started at k gives the same outputs.
    c = actor.sequence(params, x[:, k:], h0b, first[:, k:])
    np.testing.assert_allclose(np.asarray(a[:, k:]), np.asarray(c), atol=1e-6)


def test_gru_no_gradient_across_episode_boundary():
    actor, inp, params = _gru()
    T, k = 8, 4
    first = jp.zeros((1, T), bool).at[0, k].set(True)
    x = jax.random.normal(jax.random.PRNGKey(7), (1, T, inp.size))

    def later_loss(x, h0):
        return jp.sum(actor.sequence(params, x, h0, first)[:, k:] ** 2)

    gx, gh = jax.grad(later_loss, argnums=(0, 1))(x, jp.ones((1, actor.carry_size)))
    assert float(jp.max(jp.abs(gx[:, :k]))) == 0.0
    assert float(jp.max(jp.abs(gh))) == 0.0
    assert float(jp.max(jp.abs(gx[:, k:]))) > 0.0


def test_distribution_matches_brax():
    from brax.training.distribution import NormalTanhDistribution

    ref = NormalTanhDistribution(event_size=NUM_ACTIONS)
    p = jax.random.normal(jax.random.PRNGKey(0), (16, 2 * NUM_ACTIONS))
    raw = dist.sample_raw(jax.random.PRNGKey(1), p, 0.001)
    np.testing.assert_allclose(
        np.asarray(dist.log_prob(p, raw, 0.001)), np.asarray(ref.log_prob(p, raw)), rtol=1e-5
    )
    k = jax.random.PRNGKey(2)
    np.testing.assert_allclose(
        np.asarray(dist.entropy(k, p, 0.001)), np.asarray(ref.entropy(p, k)), rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(dist.mode(p, 0.001)), np.asarray(ref.mode(p)), rtol=1e-6)


def test_normalizer_matches_numpy():
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=(n, 5)) for n in (7, 50, 3)]
    s = normalizer.init(5)
    for b in batches:
        s = normalizer.update(s, jp.asarray(b))
    allx = np.concatenate(batches)
    np.testing.assert_allclose(np.asarray(s.mean), allx.mean(0), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(s.std), allx.std(0), rtol=1e-4)


@pytest.mark.parametrize("name", POLICIES)
def test_step_features_matches_step(name):
    actor, inp = _actor(name)
    params = actor.init(jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, inp.size))
    h, first = actor.initial_carry(3), jp.ones(3, bool)
    out, h1 = actor.step(params, x, h, first)
    out2, h2, feats = actor.step_features(params, x, h, first)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(out2))
    np.testing.assert_array_equal(np.asarray(h1), np.asarray(h2))
    assert feats.shape == (3, actor.hidden[-1])
