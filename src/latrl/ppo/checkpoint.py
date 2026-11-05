"""Checkpoints: actor/critic params and normalizer stats (flax msgpack).

The optimizer state is not saved: checkpoints are for evaluation, not for resuming.
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax
from flax import serialization

from latrl.ppo.trainer import TrainState


def payload(ts: TrainState) -> dict[str, Any]:
    return {"params": ts.params, "norm": ts.norm}


def save(path: pathlib.Path, ts: TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(serialization.to_bytes(jax.device_get(payload(ts))))
    tmp.replace(path)  # atomic: rsync never sees a half-written checkpoint


def load(path: pathlib.Path, template: TrainState) -> dict[str, Any]:
    """Restore into the structure of `template` (e.g. `Trainer.init(...)[0]`)."""
    return serialization.from_bytes(payload(template), path.read_bytes())
