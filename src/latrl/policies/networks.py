"""Actor and critic networks.

Defaults follow the Playground Go1 joystick reference (Brax `make_ppo_networks`,
brax 0.14.2): swish activations, LeCun-uniform kernels, zero biases, policy and value MLPs
(512, 256, 128). All sizes come from `configs/policy/*.yaml` and `configs/ppo/default.yaml`.

Every actor has the same recurrent interface so the PPO loop has one code path:

- `step(params, x[N, P], h[N, H], is_first[N]) -> (dist_params[N, 2A], h_next)` (rollout)
- `sequence(params, x[B, T, P], h0[B, H], is_first[B, T]) -> dist_params[B, T, 2A]` (loss)

`h` is the hidden state carried between control steps. For the MLP it has size 0. For the
GRU it is zeroed wherever `is_first` is set, i.e. at the first step of every episode, both
in the rollout and when the loss replays a stored sequence, so neither activations nor
gradients cross an episode boundary.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jp

_KERNEL_INIT = jax.nn.initializers.lecun_uniform()


class MLP(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for w in self.hidden:
            x = nn.swish(nn.Dense(w, kernel_init=_KERNEL_INIT)(x))
        # Last hidden layer, recorded only when applied with mutable=["intermediates"]
        # (probing targets, plan step 30); no parameters, no effect otherwise.
        self.sow("intermediates", "features", x)
        return nn.Dense(self.out, kernel_init=_KERNEL_INIT)(x)


class GRUActorCore(nn.Module):
    """One GRU layer on the normalized input, then an MLP head (rsl_rl
    `ActorCriticRecurrent` layout: RNN first, MLP actor on its output)."""

    gru_size: int
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, h: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h, y = nn.GRUCell(features=self.gru_size, name="gru")(h, x)
        return h, MLP(self.hidden, self.out, name="head")(y)


@dataclasses.dataclass(frozen=True)
class Actor:
    arch: str  # "mlp" | "gru"
    input_size: int
    out: int  # 2 × action dim (loc, raw scale)
    hidden: tuple[int, ...]
    gru_size: int = 0

    @property
    def carry_size(self) -> int:
        return self.gru_size if self.arch == "gru" else 0

    def _module(self) -> nn.Module:
        if self.arch == "mlp":
            return MLP(self.hidden, self.out)
        if self.arch == "gru":
            return GRUActorCore(self.gru_size, self.hidden, self.out)
        raise ValueError(f"unknown actor arch {self.arch!r}")

    def init(self, key: jax.Array) -> Any:
        x = jp.zeros((1, self.input_size))
        if self.arch == "mlp":
            return self._module().init(key, x)
        return self._module().init(key, jp.zeros((1, self.gru_size)), x)

    def initial_carry(self, n: int) -> jax.Array:
        return jp.zeros((n, self.carry_size))

    def step(
        self, params: Any, x: jax.Array, h: jax.Array, is_first: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if self.arch == "mlp":
            return self._module().apply(params, x), h
        h = jp.where(is_first[:, None], 0.0, h)
        h, out = self._module().apply(params, h, x)
        return out, h

    def step_features(
        self, params: Any, x: jax.Array, h: jax.Array, is_first: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """`step` that also returns the last hidden layer [N, hidden[-1]] (probing)."""
        if self.arch == "mlp":
            out, st = self._module().apply(params, x, mutable=["intermediates"])
            return out, h, st["intermediates"]["features"][0]
        h = jp.where(is_first[:, None], 0.0, h)
        (h, out), st = self._module().apply(params, h, x, mutable=["intermediates"])
        return out, h, st["intermediates"]["head"]["features"][0]

    def sequence(self, params: Any, x: jax.Array, h0: jax.Array, is_first: jax.Array) -> jax.Array:
        if self.arch == "mlp":
            return self._module().apply(params, x)

        def f(h, inp):
            xt, ft = inp
            out, h = self.step(params, xt, h, ft)
            return h, out

        # Scan over time: inputs [B, T, ...] -> [T, B, ...].
        _, out = jax.lax.scan(f, h0, (jp.swapaxes(x, 0, 1), jp.swapaxes(is_first, 0, 1)))
        return jp.swapaxes(out, 0, 1)


@dataclasses.dataclass(frozen=True)
class Critic:
    input_size: int
    hidden: tuple[int, ...]

    def _module(self) -> nn.Module:
        return MLP(self.hidden, 1)

    def init(self, key: jax.Array) -> Any:
        return self._module().init(key, jp.zeros((1, self.input_size)))

    def __call__(self, params: Any, x: jax.Array) -> jax.Array:
        return self._module().apply(params, x)[..., 0]


def num_params(params: Any) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params)))
