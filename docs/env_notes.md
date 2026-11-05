# Environment notes

Verified 2026-11-03 against the installed versions below by reading
`mujoco_playground/_src/locomotion/go1/{joystick,base,randomize}.py`, the model XML,
and stepping the env on the Mac.

## Versions (from `uv.lock`)

| Package | Version |
|---|---|
| mujoco / mujoco-mjx | 3.17.0 |
| playground (`mujoco_playground`) | 0.2.0 |
| jax / jaxlib | 0.12.0.1 (CPU on Mac) |
| flax | 0.12.12 |
| optax | 0.2.8 |
| brax (pulled in by Playground; reference PPO only) | 0.14.2 |
| rliable | 1.2.0 |
| Python | 3.12 |

## Environment

- **Registry name:** `Go1JoystickFlatTerrain` (`mujoco_playground.registry.load`).
  Class `Joystick` in `locomotion/go1/joystick.py`, task `flat_terrain`.
- **Model XML:** `scene_mjx_feetonly_flat_terrain.xml` → `go1_mjx_feetonly.xml`
  (only the feet collide with the floor; body-ground contact is not simulated).
- **MJX backend:** env default is `impl="warp"`. On the Mac CPU both `warp` and `jax`
  run; `jax` is ~2.4× faster (200 steps: 2.8 s vs 6.7 s, single env, after compile) and
  `warp` prints "solver iterations limit reached" warnings because the model uses
  `iterations=1`. **Mac uses `impl="jax"`.** GPU backend is decided in S1 (see `notes/log_decision.md` D3).

## Timing

| Quantity | Value |
|---|---|
| Control step `Δc` (`ctrl_dt`) | 0.02 s (50 Hz) |
| Physics step `Δp` (`sim_dt`) | 0.004 s |
| `n_sub` = Δc/Δp | **5** |
| Episode length | 1000 control steps = 20 s |
| Integrator | Euler (`opt.integrator = 0`), `iterations=1`, `ls_iterations=5` |

So action latency in substeps has 4 ms resolution, matching the 4 ms eval grid.

## Actuation: how a policy output becomes torque

1. Policy outputs `action ∈ R^12` (one per joint; 3 per leg: hip abduction, hip/thigh, knee/calf).
   Nothing clips it inside the env.
2. `motor_targets = default_pose + 0.5 * action` (`action_scale = 0.5` rad per unit).
   `default_pose` is the `home` keyframe's joint angles.
3. `motor_targets` are written to `data.ctrl` and held for all 5 substeps (zero-order hold).
4. The XML actuators are MuJoCo **position actuators**: force = `Kp·(ctrl − q)` with
   `Kp = 35` (set in `base.py` via `actuator_gainprm/biasprm`). `inheritrange=1` makes
   the ctrl range equal the joint range, so targets are clipped to joint limits.
5. **Kd is not on the actuator.** `base.py` sets `dof_damping[6:] = Kd = 0.5`, i.e. passive
   joint damping. MuJoCo's Euler integrator would normally integrate joint damping
   implicitly, but the Go1 XMLs set `<flag eulerdamp="disable"/>`, so here damping is
   applied **explicitly** (`−Kd·q̇` from the start of the substep), as a plain torque would
   be (F17).
6. Actuator force is clipped by `forcerange`: ±23.7 N·m (hip abduction, thigh),
   ±35.55 N·m (knee).

Implications for Phase 1 (explicit PD wrapper):
- An explicit `τ = Kp(q* − q) − Kd·q̇` puts the damping inside the force clip, whereas the
  original applies damping outside it (passive, unclipped). Both integrate damping
  explicitly (F17), so trajectories match to float32 rounding (~1e-6 rad) while no torque
  saturates, and diverge only once one does (measured: F17).
- For the actuator-lag model, whether Kd is lagged too (it is in a real motor driver) is a
  choice to record in `notes/log_decision.md` (Q2).

## Observations

`obs` is a dict.

`state` (48) — actor input:

| Slice | Content | Noise (uniform ±, × level 1.0) |
|---|---|---|
| 0:3 | local linear velocity (trunk frame) | 0.1 |
| 3:6 | gyro (angular velocity) | 0.2 |
| 6:9 | gravity vector in IMU frame | 0.05 |
| 9:21 | joint angles − default pose | 0.03 |
| 21:33 | joint velocities | 1.5 |
| 33:45 | **last action** (`info["last_act"]`, the raw policy output from the previous step) | none |
| 45:48 | command (vx, vy, yaw rate) | none |

`privileged_state` (123) — critic input (asymmetric actor-critic): `state` + noiseless
gyro, accelerometer, gravity, linvel, global angvel, joint pos/vel, actuator forces,
foot contacts, foot velocities, feet air time, perturbation force + flag.

Implications:
- **Condition A already sees one step of action history** (slice 33:45). Under action
  delay this is the policy's own output, not what the motors currently apply. Needs a
  decision before Phase 1 step 7 and before defining condition D.
- The critic sees privileged information. With randomized latency, whether the critic
  gets the true latency is a separate choice from R4 (which is about the actor).

## Commands

Sampled uniformly in ±[1.5, 0.8, 1.2] (vx m/s, vy m/s, yaw rad/s) at reset; resampled at
exponentially distributed intervals (mean 5 s), each component zeroed with some probability.

## Reward terms (scales; total is `sum × dt`, then clipped to ≥ 0)

| Term | Scale | Why it exists |
|---|---|---|
| tracking_lin_vel | +1.0 | Main task: match commanded vx, vy (`exp(−err²/0.25)`) |
| tracking_ang_vel | +0.5 | Main task: match commanded yaw rate |
| lin_vel_z | −0.5 | No bouncing of the trunk |
| ang_vel_xy | −0.05 | No rolling/pitching of the trunk |
| orientation | −5.0 | Keep trunk level |
| dof_pos_limits | −1.0 | Stay inside 95% of joint range |
| pose | +0.5 | Stay near the default stance (cheap prior for a natural posture) |
| termination | −1.0 | Falling is bad |
| stand_still | −1.0 | Hold the default pose when the command is zero |
| torques | −0.0002 | Smaller torques: efficiency, less motor heating |
| action_rate | −0.01 | Smooth actions (`‖a_t − a_{t−1}‖²`) — will interact with delay |
| energy | −0.001 | Mechanical power `|q̇·τ|` |
| feet_clearance | −2.0 | Lift swinging feet to ~0.1 m (avoid scuffing) |
| feet_height | −0.2 | Swing peak should hit 0.1 m |
| feet_slip | −0.1 | Stance feet should not slide |
| feet_air_time | +0.1 | Encourage real steps (air time > 0.1 s) instead of shuffling |

## Termination

Only `upvector.z < 0` (trunk upside-down past horizontal). **No trunk-height termination.**
Our "fall" metric for episode success therefore needs its own definition (e.g. trunk
height or tilt threshold) on top of the env's termination; decide in Phase 2.

## Noise, perturbations, domain randomization

- Sensor noise: on by default (`noise_config.level = 1.0`), uniform, scales above.
- Velocity-kick perturbations: implemented but **off** by default (`pert_config.enable = False`).
- Domain randomization (`randomize.py`, applied by the training wrapper, not the env):
  floor friction U[0.4, 1.0], joint frictionloss ×U[0.9, 1.1], armature ×U[1.0, 1.05],
  trunk CoM offset ±5 cm, all body masses ×U[0.9, 1.1], trunk mass ±1 kg, default joint
  offset ±0.05 rad.

## Playground reference PPO (Brax) config for this env

200M env steps; policy and value MLPs (512, 256, 128); policy reads `state`, value reads
`privileged_state`; learning rate 3e-4, fixed schedule.

## Mac observations

- Model renders with `mujoco.Renderer`; robot holds the home pose (trunk height 0.26 m after 2 s).
- Interactive viewer confirmed by owner 2026-11-04 (via `scripts/mjpython.sh`).
- Viewer: `scripts/mjpython.sh scripts/view_env.py`. Bare `uv run mjpython` fails with `Library not loaded: @rpath/libpython3.12.dylib` (uv's standalone Python keeps libpython in its own LIBDIR, which mjpython does not search); the wrapper sets `DYLD_FALLBACK_LIBRARY_PATH` to that LIBDIR.
- Random actions ±1 make the robot fall within 200 steps (`done = 1`), as expected.

## Our wrapper: `src/latrl/envs/go1.py` (Phase 1)

`LatencyJoystick` subclasses Playground's `Joystick`; reward, termination, commands and
noise are upstream's. Per control step (20 ms):

1. The policy action `a_t` (12 numbers in [−1, 1]) is sent into the **action delay line**
   with this step's delay `d_a` (substeps; D14 newest-arrived rule under jitter).
2. For each of the 5 physics substeps `i`: take the newest arrived action, map it to a joint
   target `q* = default_pose + 0.5·a`, clip to the joint range, compute
   `τ_cmd = 35·(q* − q) − 0.5·q̇` from the current joint angle and velocity, low-pass it
   with motor time constant `T_m`, clip to ±23.7 / ±35.55 N·m, and step MuJoCo with that
   torque. (Model actuators are plain motors, gain 1; `dof_damping` = 0.)
3. Upstream bookkeeping: contacts, obs (with noise), termination, reward, command resampling.
4. The proprioceptive obs (0:33) goes into the **obs delay line** with delay `d_o` (control
   steps); the policy gets the newest arrived one. `last_act` (33:45) = `a_t`; command
   (45:48) undelayed. The critic's `privileged_state` starts with this delayed policy obs.

Latency values are sampled in `reset` (`info["latency"]`); `with_latency` overrides them
for eval sweeps. Config: `configs/env/go1_flat.yaml`; specs: `src/latrl/envs/specs.py`.

Upstream behaviors found while building it (verified on mujoco_playground 0.2.0):
- **Damping is explicit.** The Go1 XMLs set `<flag eulerdamp="disable"/>`, so MuJoCo
  applies `dof_damping` as an explicit torque. Our explicit PD equals upstream to float
  rounding unless a torque saturates (F17).
- **`last_act` lags by one step upstream.** The obs is built before `info["last_act"]` is
  updated, so the policy choosing `a_{t+1}` sees `a_{t−1}`. We put `a_t` there (D16).
- **Auto-reset does not reset `info`.** `BraxAutoResetWrapper(full_reset=False)` restores
  only `data` and `obs` from a cached first state; per-episode latency in `info` would
  never be resampled (F16).
- **Weak-typed metric.** `reward/termination` becomes weak-typed after the first step,
  forcing one retrace of any jitted loop; our step casts metrics to float32 (F18).
- **Reward clipped at 0.** Per-step reward is `clip(sum·dt, 0, ∞)`: a robot doing badly
  gets 0, not a negative reward.
