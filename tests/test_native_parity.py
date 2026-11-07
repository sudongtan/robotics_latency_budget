"""Native MuJoCo env vs the MJX env (plan §8, R10).

1. The two models are configured identically (timestep, solver, floor, actuators, damping).
2. The numpy delay line equals the JAX one on random push/read sequences.
3. Same episodes: replayed commands equal MJX's; applied joint targets (delay logic + action
   mapping) are identical; observations (sensors + noise + obs delay) and state agree over a
   short horizon within a tolerance set from the measured gap (G2 in the log). Long-horizon
   agreement is not expected (plan §10).
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground._src.locomotion.go1 import randomize

from latrl.envs import latency
from latrl.envs.native.go1_native import DelayLine, NativeGo1, exogenous

N, T = 4, 25  # episodes, control steps (0.5 s)
D_A, D_O, T_M = 7, 2, 0.02  # 28 ms action delay (q = 1, r = 2), 40 ms obs delay, 20 ms lag


def test_models_configured_identically(make):
    env, *_ = make(noise=1.0)
    m, mx = env.mj_model, env.mjx_model
    assert m.opt.timestep == pytest.approx(0.004) and env.n_substeps == 5
    assert int(mx.opt.timestep * 1e6) == int(m.opt.timestep * 1e6)
    assert int(mx.opt.integrator) == int(m.opt.integrator)
    assert int(mx.opt.iterations) == m.opt.iterations
    assert int(mx.opt.ls_iterations) == m.opt.ls_iterations
    assert m.geom("floor").id >= 0 and m.ngeom == mx.ngeom
    np.testing.assert_array_equal(np.asarray(mx.geom_type), m.geom_type)
    for f in (
        "geom_friction",
        "body_mass",
        "body_ipos",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "jnt_range",
    ):
        np.testing.assert_allclose(np.asarray(getattr(mx, f)), getattr(m, f), err_msg=f)
    # explicit PD (D11): torque motors, no passive joint damping
    assert np.all(m.dof_damping[6:] == 0) and np.all(m.actuator_gainprm[:, 0] == 1)


def test_delay_line_matches_jax():
    rng = np.random.default_rng(0)
    k, n = 8, 3
    jl = [latency.init(jp.zeros(2), k) for _ in range(n)]
    nl = DelayLine.init(np.zeros((n, 2)), k)
    for _ in range(60):
        v = rng.normal(size=(n, 2))
        d = rng.integers(0, 30, size=n)
        jl = [latency.push(line, v[i], d[i], elapsed=5) for i, line in enumerate(jl)]
        nl.push(v, d, elapsed=5)
        for now in range(5):
            want = np.stack([np.asarray(latency.read(line, now)) for line in jl])
            np.testing.assert_allclose(nl.read(now), want, rtol=0, atol=1e-6)


def _actions():
    t = np.arange(T)[:, None, None]
    j = np.arange(12)[None, None, :]
    e = np.arange(N)[None, :, None]
    return (0.4 * np.sin(0.3 * t + 0.5 * j + e)).astype(np.float32)  # [T, N, 12]


@pytest.fixture(scope="module")
def rollouts(make):
    env, *_ = make(noise=1.0)
    keys = jax.random.split(jax.random.PRNGKey(3), N)
    models, in_axes = randomize.domain_randomize(env.mjx_model, jax.random.split(keys[0], N))

    def with_model(m, f):
        old = env._mjx_model
        env._mjx_model = m
        try:
            return f()
        finally:
            env._mjx_model = old

    def reset(m, k):
        s = with_model(m, lambda: env.reset(k))
        return env.with_latency(s, action_delay=D_A, obs_delay=D_O, t_m=T_M)

    step = jax.jit(
        jax.vmap(lambda m, s, a: with_model(m, lambda: env.step(s, a)), in_axes=(in_axes, 0, 0))
    )
    s = jax.jit(jax.vmap(reset, in_axes=(in_axes, 0)))(models, keys)
    acts = _actions()
    mjx = {"obs": [np.asarray(s.obs["state"])], "qpos": [], "cmd": [], "targets": []}
    for t in range(T):
        s = step(models, s, jp.asarray(acts[t]))
        mjx["obs"].append(np.asarray(s.obs["state"]))
        mjx["qpos"].append(np.asarray(s.data.qpos))
        mjx["cmd"].append(np.asarray(s.info["command"]))
        mjx["targets"].append(np.asarray(s.info["latency"].applied_targets))

    exo = exogenous(env, models, in_axes, keys, T)
    nat_env = NativeGo1(
        env, exo, np.full(N, D_A), np.full(N, D_O), np.full(N, T_M), jitter=False, action_max=30
    )
    nat = {"obs": [nat_env.obs.copy()], "qpos": [], "targets": []}
    for t in range(T):
        nat_env.step(acts[t].astype(np.float64))
        nat["obs"].append(nat_env.obs.copy())
        nat["qpos"].append(np.stack([d.qpos.copy() for d in nat_env.datas]))
        nat["targets"].append(nat_env.targets.copy())
    return exo, {k: np.stack(v) for k, v in mjx.items()}, {k: np.stack(v) for k, v in nat.items()}


def test_replayed_commands_equal_mjx(rollouts):
    exo, mjx, _ = rollouts
    np.testing.assert_allclose(exo.command[:, 1:].swapaxes(0, 1), mjx["cmd"], atol=1e-6)


def test_applied_targets_identical(rollouts):
    _, mjx, nat = rollouts
    np.testing.assert_allclose(nat["targets"], mjx["targets"], rtol=0, atol=1e-5)


def test_reset_observation_identical(rollouts):
    _, mjx, nat = rollouts
    np.testing.assert_allclose(nat["obs"][0], mjx["obs"][0], rtol=0, atol=1e-4)


def test_short_horizon_agreement(rollouts):
    """Same inputs, different simulators (float32 MJX vs float64 C, different contact
    solvers). Tolerances from the measured gap (G2): ≤ 6e-4 m / 1.2e-2 rad over the first
    0.12 s; under this open-loop forcing the feet impacts then amplify it to ≈ 1.4e-2 m /
    0.15 rad by 0.5 s. Bounds are ≈ 2–3× the measurement."""
    _, mjx, nat = rollouts
    obs_gap = np.abs(nat["obs"] - mjx["obs"])
    assert obs_gap[:, :, 45:].max() < 1e-6  # command: exact
    assert obs_gap[:, :, 33:45].max() < 1e-6  # last action: exact
    # obs delay 2: the first two policy observations are the reset reading in both sims
    assert obs_gap[1:3, :, :33].max() < 1e-4
    q = np.abs(nat["qpos"] - mjx["qpos"])
    assert q[:6, :, :3].max() < 2e-3 and q[:6, :, 7:].max() < 3e-2  # first 0.12 s
    assert q[:, :, :3].max() < 5e-2 and q[:, :, 7:].max() < 0.4  # 0.5 s
