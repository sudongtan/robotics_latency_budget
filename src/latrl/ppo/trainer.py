"""PPO for the latency env: feedforward and recurrent actors, one code path.

Structure mirrors Brax PPO (brax 0.14.2, `agents/ppo/train.py`) as used by the Playground
Go1 joystick reference, so S1 can compare the two on condition A:

- `num_envs` envs step in parallel under `jax.vmap`; one iteration collects an unroll of
  `unroll_length` steps per env, updates the observation normalizer with it, then runs
  `num_updates_per_batch` epochs of `num_minibatches` minibatches. Minibatches are whole
  per-env unrolls (Brax permutes the env axis and keeps time intact), which is exactly
  what the GRU needs: each minibatch item is a sequence with its stored initial hidden
  state (truncated BPTT over the unroll, no burn-in; D22).
- Losses (D20): clipped surrogate (ε = 0.3), squared-error value loss × 0.5·vf_coef,
  entropy bonus, Brax's advantage estimator (`latrl.ppo.gae`), advantages normalized per
  minibatch and recomputed per minibatch with the current critic. Adam with global-norm clipping.
- Asymmetric actor-critic: the actor sees its policy input (`latrl.policies.inputs`), the
  critic sees `privileged_state` for every condition (D23).
- Domain randomization (D15): each env gets one randomized model for the whole run, from
  Playground's `domain_randomize`, as in `wrap_for_brax_training`.

Differences from Brax that matter:
- **Resets (F16).** Finished envs are reset with a full `env.reset`, so the per-episode
  latency in `info["latency"]` is resampled. Brax/Playground's default auto-reset restores
  only data and obs.
- **Truncation** bootstraps from V(true final state) instead of dropping the transition
  (`latrl.ppo.gae`).
- Episode length is counted here, not by an EpisodeWrapper.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp
import optax
from flax import struct
from mujoco_playground._src.locomotion.go1 import randomize

from latrl.envs.go1 import LatencyJoystick
from latrl.envs.specs import NUM_ACTIONS, PRIVILEGED_OBS_SIZE
from latrl.policies import distribution as dist
from latrl.policies.inputs import PolicyInput
from latrl.policies.networks import Actor, Critic
from latrl.ppo import normalizer
from latrl.ppo.gae import ESTIMATORS, advantages


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    num_envs: int
    unroll_length: int
    num_minibatches: int
    num_updates_per_batch: int
    discounting: float
    gae_lambda: float
    learning_rate: float
    entropy_cost: float
    clipping_epsilon: float
    vf_coefficient: float
    max_grad_norm: float
    reward_scaling: float
    normalize_advantage: bool
    normalize_observations: bool
    min_std: float
    advantage_estimator: str
    value_hidden: tuple[int, ...]
    episode_length: int

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> PPOConfig:
        d = {f.name: cfg[f.name] for f in dataclasses.fields(cls)}
        d["value_hidden"] = tuple(d["value_hidden"])
        out = cls(**d)
        if out.num_envs % out.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if out.advantage_estimator not in ESTIMATORS:
            raise ValueError(f"advantage_estimator must be one of {ESTIMATORS}")
        return out

    @property
    def steps_per_iteration(self) -> int:
        return self.num_envs * self.unroll_length


def make_actor(policy_cfg: Mapping[str, Any], input_size: int) -> Actor:
    return Actor(
        arch=policy_cfg["arch"],
        input_size=input_size,
        out=2 * NUM_ACTIONS,
        hidden=tuple(policy_cfg["hidden"]),
        gru_size=int(policy_cfg.get("gru_size", 0)),
    )


@struct.dataclass
class Norm:
    policy: normalizer.Stats
    critic: normalizer.Stats


@struct.dataclass
class TrainState:
    params: dict[str, Any]  # {"actor", "critic"}
    opt_state: Any
    norm: Norm
    key: jax.Array


@struct.dataclass
class Carry:
    """Per-env rollout state, carried across iterations (all leading dim N)."""

    env_state: Any
    hist: jax.Array  # policy-input history [N, H, 48]
    h: jax.Array  # actor hidden state [N, carry_size]
    is_first: jax.Array  # bool [N]: the next step is the first of an episode
    ep_steps: jax.Array  # int32 [N]
    ep_return: jax.Array  # float32 [N]


@struct.dataclass
class Transition:
    x: jax.Array  # raw (unnormalized) actor input
    crit_x: jax.Array
    next_crit_x: jax.Array  # critic input at the true next state (before reset)
    raw_action: jax.Array  # pre-tanh sample
    log_prob: jax.Array
    reward: jax.Array
    termination: jax.Array
    done: jax.Array  # termination or truncation
    is_first: jax.Array


def _tree_where(mask: jax.Array, a: Any, b: Any) -> Any:
    def f(x, y):
        m = mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))
        return jp.where(m, x, y)

    return jax.tree.map(f, a, b)


class Trainer:
    def __init__(
        self,
        env: LatencyJoystick,
        env_cfg: Mapping[str, Any],
        ppo_cfg: Mapping[str, Any],
        policy_cfg: Mapping[str, Any],
    ) -> None:
        self.env = env
        self.cfg = PPOConfig.from_dict(ppo_cfg)
        self.inp = PolicyInput.from_config(policy_cfg, env_cfg)
        self.actor = make_actor(policy_cfg, self.inp.size)
        self.critic = Critic(PRIVILEGED_OBS_SIZE, self.cfg.value_hidden)
        self.domain_randomization = bool(env_cfg["domain_randomization"])
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.cfg.max_grad_norm),
            optax.adam(self.cfg.learning_rate),
        )

    # ------------------------------------------------------------------ models

    def models(self, key: jax.Array, n: int) -> tuple[Any, Any]:
        """(batched mjx models, vmap in_axes): one randomized model per env (D15)."""
        if not self.domain_randomization:
            return self.env.mjx_model, None
        return randomize.domain_randomize(self.env.mjx_model, jax.random.split(key, n))

    @contextlib.contextmanager
    def _with_model(self, model: Any):
        # Same trick as Playground's BraxDomainRandomizationVmapWrapper: swap the model the
        # env reads while tracing the vmapped body.
        old = self.env._mjx_model
        try:
            self.env._mjx_model = model
            yield
        finally:
            self.env._mjx_model = old

    def env_reset(self, models: Any, in_axes: Any, keys: jax.Array) -> Any:
        def f(m, k):
            with self._with_model(m):
                return self.env.reset(k)

        return jax.vmap(f, in_axes=(in_axes, 0))(models, keys)

    def env_step(self, models: Any, in_axes: Any, state: Any, action: jax.Array) -> Any:
        def f(m, s, a):
            with self._with_model(m):
                return self.env.step(s, a)

        return jax.vmap(f, in_axes=(in_axes, 0, 0))(models, state, action)

    # --------------------------------------------------------------- policy I/O

    def features(self, hist: jax.Array, env_state: Any) -> jax.Array:
        return jax.vmap(self.inp.features)(hist, env_state.info)

    def init_hist(self, env_state: Any) -> jax.Array:
        return jax.vmap(self.inp.init)(env_state.obs["state"])

    def act(
        self, ts: TrainState, x: jax.Array, h: jax.Array, is_first: jax.Array, key: jax.Array | None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """(action, raw_action, log_prob, h_next). key=None → deterministic tanh(loc)."""
        xn = self._norm_policy(ts.norm, x)
        params, h = self.actor.step(ts.params["actor"], xn, h, is_first)
        if key is None:
            a = dist.mode(params, self.cfg.min_std)
            return a, a, jp.zeros(a.shape[:-1]), h
        raw = dist.sample_raw(key, params, self.cfg.min_std)
        return dist.postprocess(raw), raw, dist.log_prob(params, raw, self.cfg.min_std), h

    def _norm_policy(self, norm: Norm, x: jax.Array) -> jax.Array:
        if not self.cfg.normalize_observations:
            return x
        return normalizer.normalize(norm.policy, x)

    def _norm_critic(self, norm: Norm, x: jax.Array) -> jax.Array:
        if not self.cfg.normalize_observations:
            return x
        return normalizer.normalize(norm.critic, x)

    # -------------------------------------------------------------------- init

    def init(self, key: jax.Array, models: Any, in_axes: Any) -> tuple[TrainState, Carry]:
        k_actor, k_critic, k_reset, k_train = jax.random.split(key, 4)
        params = {"actor": self.actor.init(k_actor), "critic": self.critic.init(k_critic)}
        n = self.cfg.num_envs
        env_state = self.env_reset(models, in_axes, jax.random.split(k_reset, n))
        ts = TrainState(
            params=params,
            opt_state=self.optimizer.init(params),
            norm=Norm(normalizer.init(self.inp.size), normalizer.init(PRIVILEGED_OBS_SIZE)),
            key=k_train,
        )
        carry = Carry(
            env_state=env_state,
            hist=self.init_hist(env_state),
            h=self.actor.initial_carry(n),
            is_first=jp.ones(n, bool),
            ep_steps=jp.zeros(n, jp.int32),
            ep_return=jp.zeros(n, jp.float32),
        )
        return ts, carry

    # ----------------------------------------------------------------- rollout

    def rollout_step(
        self, ts: TrainState, models: Any, in_axes: Any, carry: Carry, key: jax.Array
    ) -> tuple[Carry, tuple[Transition, dict[str, jax.Array]]]:
        k_act, k_reset = jax.random.split(key)
        x = self.features(carry.hist, carry.env_state)
        action, raw, logp, h = self.act(ts, x, carry.h, carry.is_first, k_act)
        nxt = self.env_step(models, in_axes, carry.env_state, action)

        term = nxt.done > 0
        ep_steps = carry.ep_steps + 1
        trunc = (ep_steps >= self.cfg.episode_length) & ~term
        done = term | trunc
        ep_return = carry.ep_return + nxt.reward

        # F16: a full env.reset for finished envs, so latency is resampled each episode.
        # Skipped (lax.cond, outside vmap) on steps where no env finished.
        def reset_done(_):
            keys = jax.random.split(k_reset, self.cfg.num_envs)
            return _tree_where(done, self.env_reset(models, in_axes, keys), nxt)

        env_state = jax.lax.cond(jp.any(done), reset_done, lambda _: nxt, None)
        frames = env_state.obs["state"]
        hist = jp.where(
            done[:, None, None],
            self.init_hist(env_state),
            jax.vmap(self.inp.push)(carry.hist, frames),
        )
        tr = Transition(
            x=x,
            crit_x=carry.env_state.obs["privileged_state"],
            next_crit_x=nxt.obs["privileged_state"],
            raw_action=raw,
            log_prob=logp,
            reward=nxt.reward,
            termination=term.astype(jp.float32),
            done=done.astype(jp.float32),
            is_first=carry.is_first,
        )
        stats = {
            "completed_return": jp.sum(jp.where(done, ep_return, 0.0)),
            "completed_length": jp.sum(jp.where(done, ep_steps, 0)).astype(jp.float32),
            "completed": jp.sum(done).astype(jp.float32),
            "terminated": jp.sum(term).astype(jp.float32),
            "reward": jp.mean(nxt.reward),
        }
        carry = Carry(
            env_state=env_state,
            hist=hist,
            h=h,
            is_first=done,
            ep_steps=jp.where(done, 0, ep_steps),
            ep_return=jp.where(done, 0.0, ep_return),
        )
        return carry, (tr, stats)

    # -------------------------------------------------------------------- loss

    def loss(
        self, params: dict[str, Any], norm: Norm, mb: Transition, h0: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """PPO loss on a minibatch of sequences: every leaf [B, T, ...], h0 [B, carry]."""
        c = self.cfg
        dparams = self.actor.sequence(
            params["actor"], self._norm_policy(norm, mb.x), h0, mb.is_first
        )
        values = self.critic(params["critic"], self._norm_critic(norm, mb.crit_x))
        next_values = self.critic(params["critic"], self._norm_critic(norm, mb.next_crit_x))

        def tm(x):  # [B, T] -> [T, B]
            return jp.swapaxes(x, 0, 1)

        adv, vs = advantages(
            c.advantage_estimator,
            tm(mb.reward * c.reward_scaling),
            tm(values),
            tm(next_values),
            tm(mb.termination),
            tm(mb.done),
            c.discounting,
            c.gae_lambda,
        )
        adv, vs = tm(adv), tm(vs)
        if c.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logp = dist.log_prob(dparams, mb.raw_action, c.min_std)
        log_ratio = logp - mb.log_prob
        ratio = jp.exp(log_ratio)
        surr = jp.minimum(
            ratio * adv,
            jp.clip(ratio, 1 - c.clipping_epsilon, 1 + c.clipping_epsilon) * adv,
        )
        policy_loss = -jp.mean(surr)
        v_loss = 0.5 * c.vf_coefficient * jp.mean((vs - values) ** 2)
        entropy = jp.mean(dist.entropy(key, dparams, c.min_std))
        total = policy_loss + v_loss - c.entropy_cost * entropy
        _, scale = dist.split(dparams, c.min_std)
        return total, {
            "loss/total": total,
            "loss/policy": policy_loss,
            "loss/value": v_loss,
            "loss/entropy": entropy,
            "loss/approx_kl": jp.mean((ratio - 1.0) - log_ratio),
            "loss/clip_frac": jp.mean(jp.abs(ratio - 1.0) > c.clipping_epsilon),
            "policy/std_mean": jp.mean(scale),
        }

    # --------------------------------------------------------------- iteration

    def iteration(
        self, ts: TrainState, carry: Carry, models: Any, in_axes: Any
    ) -> tuple[TrainState, Carry, dict[str, jax.Array]]:
        """One rollout + PPO update. Jit with `models`/`in_axes` closed over or static."""
        c = self.cfg
        key, k_roll, k_sgd = jax.random.split(ts.key, 3)
        h0 = carry.h
        carry, (data, stats) = jax.lax.scan(
            functools.partial(self.rollout_step, ts, models, in_axes),
            carry,
            jax.random.split(k_roll, c.unroll_length),
        )
        # data leaves [T, N, ...] -> [N, T, ...] (sequences per env).
        data = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data)

        norm = ts.norm
        if c.normalize_observations:  # before SGD, as Brax
            norm = Norm(
                normalizer.update(norm.policy, data.x),
                normalizer.update(norm.critic, data.crit_x),
            )

        grad_fn = jax.value_and_grad(self.loss, has_aux=True)

        def minibatch(carry_, mb):
            params, opt_state, k = carry_
            k, k_ent = jax.random.split(k)
            batch, mb_h0 = mb
            (_, m), g = grad_fn(params, norm, batch, mb_h0, k_ent)
            updates, opt_state = self.optimizer.update(g, opt_state, params)
            m["grad_norm"] = optax.tree.norm(g)
            return (optax.apply_updates(params, updates), opt_state, k), m

        def epoch(carry_, k_perm):
            params, opt_state, k = carry_
            perm = jax.random.permutation(k_perm, c.num_envs)

            def shard(x):
                return x[perm].reshape((c.num_minibatches, -1) + x.shape[1:])

            mbs = (jax.tree.map(shard, data), shard(h0))
            return jax.lax.scan(minibatch, (params, opt_state, k), mbs)

        k_perm, k_mb = jax.random.split(k_sgd)
        (params, opt_state, _), m = jax.lax.scan(
            epoch,
            (ts.params, ts.opt_state, k_mb),
            jax.random.split(k_perm, c.num_updates_per_batch),
        )
        metrics = jax.tree.map(jp.mean, m)
        n_done = jp.sum(stats["completed"])
        metrics |= {
            "rollout/reward_per_step": jp.mean(stats["reward"]),
            "rollout/episodes": n_done,
            "rollout/terminations": jp.sum(stats["terminated"]),
            # NaN when no episode finished in this rollout.
            "rollout/episode_return": jp.sum(stats["completed_return"]) / n_done,
            "rollout/episode_length": jp.sum(stats["completed_length"]) / n_done,
        }
        ts = TrainState(params=params, opt_state=opt_state, norm=norm, key=key)
        return ts, carry, metrics

    # -------------------------------------------------------------------- eval

    def evaluate(
        self, ts: TrainState, models: Any, in_axes: Any, key: jax.Array, num_envs: int
    ) -> dict[str, jax.Array]:
        """Deterministic policy, one episode per env, latency from the training
        distribution (in-distribution check, as Brax's eval). Reward accumulates until the
        first termination."""
        env_state = self.env_reset(models, in_axes, jax.random.split(key, num_envs))
        init = (
            env_state,
            self.init_hist(env_state),
            self.actor.initial_carry(num_envs),
            jp.ones(num_envs, bool),
            jp.ones(num_envs, bool),  # alive
            jp.zeros(num_envs),
            jp.zeros(num_envs),
        )

        def f(carry_, _):
            s, hist, h, is_first, alive, ret, length = carry_
            x = self.features(hist, s)
            a, _, _, h = self.act(ts, x, h, is_first, None)
            s = self.env_step(models, in_axes, s, a)
            ret = ret + alive * s.reward
            length = length + alive
            alive = alive & (s.done == 0)
            hist = jax.vmap(self.inp.push)(hist, s.obs["state"])
            return (s, hist, h, jp.zeros_like(is_first), alive, ret, length), None

        out, _ = jax.lax.scan(f, init, None, length=self.cfg.episode_length)
        *_, alive, ret, length = out
        return {
            "eval/episode_return": jp.mean(ret),
            "eval/episode_return_std": jp.std(ret),
            "eval/episode_length": jp.mean(length),
            "eval/survived": jp.mean(alive),
        }
