"""Open the Go1 joystick model in the native MuJoCo viewer (Phase 0 sanity check).

Run with `scripts/mjpython.sh scripts/view_env.py` on macOS: the passive viewer needs
`mjpython` because macOS only lets the main thread own the GUI.

The robot holds its "home" keyframe pose: the position actuators track the default
joint angles, so it should stand still. No policy is involved.
"""

import time

import mujoco
import mujoco.viewer
from mujoco_playground import registry


def main() -> None:
    env = registry.load("Go1JoystickFlatTerrain", config_overrides={"impl": "jax"})
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    data.ctrl[:] = model.keyframe("home").qpos[7:]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
