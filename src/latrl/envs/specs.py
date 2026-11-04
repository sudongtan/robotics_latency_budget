"""Shared obs / action / latency specs (R10: one module for the MJX and native envs).

Units (CLAUDE.md conventions): action delay in physics substeps (Δp = 4 ms), observation
delay in control steps (Δc = 20 ms), motor time constant in seconds. Milliseconds are for
display only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

# Policy observation layout of Playground's Go1 joystick `state` obs (48), verified against
# mujoco_playground 0.2.0 (docs/env_notes.md). Only the proprioceptive part is sensor data
# and goes through observation delay. `last_act` is the controller's own memory (D10) and
# `command` is the operator's joystick input; neither is delayed (plan §6).
PROPRIO = slice(0, 33)  # local linvel 3, gyro 3, gravity 3, joint pos 12, joint vel 12
LAST_ACT = slice(33, 45)
COMMAND = slice(45, 48)
OBS_SIZE = 48
PRIVILEGED_OBS_SIZE = 123  # [state (48), true signals for the critic (75)]
NUM_ACTIONS = 12

ACTUATOR_MODES = ("explicit_pd", "position")


@dataclasses.dataclass(frozen=True)
class DelaySpec:
    """Delay distribution for one channel (action or observation).

    Per episode the base delay is drawn from U{lo..hi}. With jitter, each control step
    the offset is resampled with probability `jitter_p` from U{-jitter_width..jitter_width};
    the effective delay is clip(base + offset, 0, max). `max` also sizes the buffer.
    """

    lo: int
    hi: int
    max: int
    jitter_p: float
    jitter_width: int

    def __post_init__(self) -> None:
        if not (0 <= self.lo <= self.hi <= self.max):
            raise ValueError(f"need 0 <= lo <= hi <= max, got {self}")
        if not (0.0 <= self.jitter_p <= 1.0) or self.jitter_width < 0:
            raise ValueError(f"bad jitter settings: {self}")


@dataclasses.dataclass(frozen=True)
class ActuatorSpec:
    """How policy targets become joint torque.

    `explicit_pd` (D11): our wrapper computes τ = Kp(q* − q) − Kd·q̇, low-pass filters it
    with motor time constant T_m ~ U[tm_lo, tm_hi] (seconds), and clips to the torque limit.
    `position`: Playground's original MuJoCo position actuators (Kd as passive joint
    damping). Kept only to check that our wrapper reproduces the upstream env exactly; it
    cannot model motor lag.
    """

    mode: str
    tm_lo: float
    tm_hi: float

    def __post_init__(self) -> None:
        if self.mode not in ACTUATOR_MODES:
            raise ValueError(f"mode must be one of {ACTUATOR_MODES}, got {self.mode!r}")
        if not (0.0 <= self.tm_lo <= self.tm_hi):
            raise ValueError(f"need 0 <= tm_lo <= tm_hi, got {self}")
        if self.mode == "position" and self.tm_hi > 0.0:
            raise ValueError("motor lag requires mode='explicit_pd'")


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    action_delay: DelaySpec  # substeps
    obs_delay: DelaySpec  # control steps
    actuator: ActuatorSpec

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> EnvSpec:
        """Build from the `latency` and `actuator` blocks of configs/env/*.yaml."""
        lat = cfg["latency"]
        return cls(
            action_delay=DelaySpec(**dict(lat["action"])),
            obs_delay=DelaySpec(**dict(lat["obs"])),
            actuator=ActuatorSpec(**dict(cfg["actuator"])),
        )
