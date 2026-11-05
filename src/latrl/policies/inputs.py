"""What the actor sees, per condition (plan §7). Set by `configs/policy/*.yaml`.

- `obs` (A, B, C, E): the env's 48-dim policy observation `obs["state"]` (delayed proprio,
  own last action, command; plan §6).
- `history` (D): the last H observation frames, newest first, flattened. Each frame's
  `last_act` slot holds the action sent one step before it (D16), so H frames carry the H
  most recent actions a_t … a_{t−H+1} (the constant-delay Markov argument, F11) *and* the
  H most recent proprio readings, which the policy needs to infer an unknown delay from
  how the body responded to its past actions (D21).
- `privileged_latency` (F): `obs["state"]` plus the true delays, normalized to [0, 1]:
  `[action_delay / action_max, obs_delay / obs_max]`.

R4: only `privileged_latency` reads `info["latency"]`; tests/test_policy_inputs.py checks
that the other modes' features do not change when the latency does.

The history buffer is a jax array carried by the training/eval loop (R3), one per env. It
starts filled with the reset frame (plan §10: buffers start with the initial obs).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp

from latrl.envs.specs import OBS_SIZE

MODES = ("obs", "history", "privileged_latency")
NUM_LATENCY_FEATURES = 2


@dataclasses.dataclass(frozen=True)
class PolicyInput:
    mode: str
    history_len: int  # frames kept; 1 unless mode == "history"
    action_delay_max: int  # substeps, for normalizing the privileged feature
    obs_delay_max: int  # control steps

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"policy input mode must be one of {MODES}, got {self.mode!r}")
        if self.history_len < 1 or (self.mode != "history" and self.history_len != 1):
            raise ValueError(f"history_len must be >= 1, and 1 unless mode='history': {self}")

    @classmethod
    def from_config(cls, policy_cfg: Mapping[str, Any], env_cfg: Mapping[str, Any]) -> PolicyInput:
        return cls(
            mode=policy_cfg["input"],
            history_len=int(policy_cfg.get("history_len", 1)),
            action_delay_max=int(env_cfg["latency"]["action"]["max"]),
            obs_delay_max=int(env_cfg["latency"]["obs"]["max"]),
        )

    @property
    def size(self) -> int:
        if self.mode == "history":
            return OBS_SIZE * self.history_len
        if self.mode == "privileged_latency":
            return OBS_SIZE + NUM_LATENCY_FEATURES
        return OBS_SIZE

    # Per-env functions; the loops vmap them.

    def init(self, frame: jax.Array) -> jax.Array:
        """History buffer [H, 48] filled with the reset frame."""
        return jp.broadcast_to(frame, (self.history_len, OBS_SIZE))

    def push(self, hist: jax.Array, frame: jax.Array) -> jax.Array:
        return jp.concatenate([frame[None], hist[:-1]], axis=0)

    def features(self, hist: jax.Array, info: Mapping[str, Any]) -> jax.Array:
        """Actor input from the history buffer (hist[0] is the current obs["state"])."""
        if self.mode == "history":
            return hist.reshape(-1)
        if self.mode == "privileged_latency":
            lat = info["latency"]
            feats = jp.stack(
                [
                    lat.action_delay / max(self.action_delay_max, 1),
                    lat.obs_delay / max(self.obs_delay_max, 1),
                ]
            ).astype(jp.float32)
            return jp.concatenate([hist[0], feats])
        return hist[0]
