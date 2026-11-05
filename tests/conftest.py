"""Shared fixtures: env construction is slow (~5 s jit per variant), so cache variants."""

import functools
import pathlib

import jax
import jax.numpy as jp
import numpy as np
import pytest
from omegaconf import OmegaConf

from latrl.envs.go1 import make_env

ENV_CFG = pathlib.Path(__file__).parents[1] / "configs" / "env" / "go1_flat.yaml"


@functools.cache
def _env(mode: str, noise: float, a_hi: int, o_hi: int, a_jit: float, o_jit: float):
    cfg = OmegaConf.load(ENV_CFG)
    cfg.noise_level = noise
    cfg.actuator.mode = mode
    cfg.latency.action.hi = a_hi
    cfg.latency.obs.hi = o_hi
    cfg.latency.action.jitter_p, cfg.latency.action.jitter_width = a_jit, (3 if a_jit else 0)
    cfg.latency.obs.jitter_p, cfg.latency.obs.jitter_width = o_jit, (1 if o_jit else 0)
    env = make_env(cfg)
    return env, jax.jit(env.reset), jax.jit(env.step), jax.jit(env.with_latency)


@pytest.fixture(scope="session")
def make():
    """make(mode="explicit_pd", noise=0.0, a_hi=30, o_hi=6, a_jit=0.0, o_jit=0.0)
    -> (env, reset, step, with_latency), all jitted. Ranges are wide so `with_latency`
    can set any test value within the buffer bounds."""

    def f(mode="explicit_pd", noise=0.0, a_hi=30, o_hi=6, a_jit=0.0, o_jit=0.0):
        return _env(mode, noise, a_hi, o_hi, a_jit, o_jit)

    return f


def rollout(step, state, actions):
    """Run a fixed action sequence; return the list of states after each step."""
    out = []
    for a in actions:
        state = step(state, jp.asarray(a, jp.float32))
        out.append(state)
    return out


def random_actions(n, seed=0, scale=0.3):
    return np.random.default_rng(seed).uniform(-scale, scale, (n, 12)).astype(np.float32)
