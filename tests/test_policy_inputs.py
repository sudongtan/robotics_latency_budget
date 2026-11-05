"""Policy-side inputs: history layout and R4 (latency reaches only condition F's actor)."""

import jax
import jax.numpy as jp
import numpy as np
from conftest import ENV_CFG
from omegaconf import OmegaConf

from latrl.envs.specs import LAST_ACT, OBS_SIZE
from latrl.policies.inputs import PolicyInput

ENV = OmegaConf.load(ENV_CFG)


def _inp(mode, h=1):
    return PolicyInput.from_config({"input": mode, "history_len": h}, ENV)


def test_history_newest_first_and_filled_at_reset():
    inp = _inp("history", 4)
    f0, f1 = jp.full(OBS_SIZE, 0.0), jp.full(OBS_SIZE, 1.0)
    hist = inp.init(f0)
    assert hist.shape == (4, OBS_SIZE) and float(jp.max(jp.abs(hist))) == 0.0
    hist = inp.push(hist, f1)
    x = inp.features(hist, {})
    assert x.shape == (inp.size,) == (4 * OBS_SIZE,)
    np.testing.assert_array_equal(np.asarray(x[:OBS_SIZE]), 1.0)
    np.testing.assert_array_equal(np.asarray(x[OBS_SIZE:]), 0.0)


def test_history_carries_last_four_actions(make):
    """With D16's last_act, 4 frames hold a_t, a_{t-1}, a_{t-2}, a_{t-3}."""
    env, reset, step, _ = make()
    inp = _inp("history", 4)
    s = reset(jax.random.PRNGKey(0))
    hist = inp.init(s.obs["state"])
    acts = [jp.full(12, 0.01 * (i + 1)) for i in range(6)]
    for a in acts:
        s = step(s, a)
        hist = inp.push(hist, s.obs["state"])
    x = np.asarray(inp.features(hist, s.info)).reshape(4, OBS_SIZE)
    for j in range(4):
        np.testing.assert_allclose(x[j, LAST_ACT], np.asarray(acts[-1 - j]), atol=1e-7)


def test_r4_latency_only_in_privileged_features(make):
    """Changing the true latency changes F's actor input and nobody else's."""
    env, reset, _, with_latency = make()
    s = reset(jax.random.PRNGKey(0))
    s_lo = with_latency(s, action_delay=0, obs_delay=0)
    s_hi = with_latency(s, action_delay=20, obs_delay=3)
    for mode, h in [("obs", 1), ("history", 4)]:
        inp = _inp(mode, h)
        hist = inp.init(s.obs["state"])
        np.testing.assert_array_equal(
            np.asarray(inp.features(hist, s_lo.info)), np.asarray(inp.features(hist, s_hi.info))
        )
    inp = _inp("privileged_latency")
    hist = inp.init(s.obs["state"])
    lo, hi = inp.features(hist, s_lo.info), inp.features(hist, s_hi.info)
    np.testing.assert_array_equal(np.asarray(lo[:OBS_SIZE]), np.asarray(hi[:OBS_SIZE]))
    np.testing.assert_allclose(np.asarray(lo[OBS_SIZE:]), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(hi[OBS_SIZE:]), [20 / 30, 3 / 6], rtol=1e-6)


def test_only_condition_f_uses_privileged_input():
    exp_dir = ENV_CFG.parents[1] / "experiment"
    pol_dir = ENV_CFG.parents[1] / "policy"
    for path in sorted(exp_dir.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        policy = [d["override /policy"] for d in cfg.defaults if "override /policy" in d][0]
        mode = OmegaConf.load(pol_dir / f"{policy}.yaml").input
        assert (mode == "privileged_latency") == (path.stem == "F"), path.name
