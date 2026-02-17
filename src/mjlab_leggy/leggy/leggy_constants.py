import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
import numpy as np
from mjlab_leggy.leggy.leggy_actions import motor_to_knee, knee_to_motor

LEGGY_XML: Path = Path(os.path.dirname(__file__)) / "robot.xml"
assert LEGGY_XML.exists(), f"XML not found: {LEGGY_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LEGGY_XML))


# NOTE: Motor-to-knee conversion is done via the LeggyJointAction action term.
# Passive joints are handled automatically by MuJoCo's constraint solver.
# Observations compute motor space from knee angles using joint_pos_motor/joint_vel_motor.
#
# For usage in tasks, see: from mjlab_leggy.leggy.leggy_actions import LeggyJointActionCfg


stand_pose = {
    "hipY": 13 * np.pi / 180.0, # 0.2269 rad
    "hipX": -35 * np.pi / 180.0, # -0.4363 rad
    "kneeMotor": -38 * np.pi / 180.0, # -0.9250 rad
    "knee": (-38+35) * np.pi / 180.0 # -0.4887 rad (-28 deg)
}

HOME_FRAME = EntityCfg.InitialStateCfg(
    # Starting position (robot body center height)
    pos=(0.0, 0.0, 0.175),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        ".*hipY.*": stand_pose["hipY"],
        ".*hipX.*": stand_pose["hipX"], # -25 deg
        ".*knee.*": stand_pose["knee"], # -28 deg (motor_to_knee(-53, -25) = -53+25 = -28)
        "LpassiveMotor": knee_to_motor(stand_pose["knee"], stand_pose["hipX"]), # -53 deg (motor)
        "RpassiveMotor": knee_to_motor(stand_pose["knee"], stand_pose["hipX"]), # -53 deg (motor)
        "Lpassive2": stand_pose["knee"], # -28 deg
        "Rpassive2": stand_pose["knee"] # -28 deg
    },
    joint_vel={".*": 0.0},
    # Explicitly set root (freejoint) velocities to zero
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
)


# Use existing position actuators defined in robot.xml
# (LhipY, LhipX, Lknee, RhipY, RhipX, Rknee with kp=10, forcerange=0.236)
LEGGY_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=("LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"),
)

LEGGY_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(),  # Empty - rely on config.json for all collision properties
    articulation=EntityArticulationInfoCfg(
        actuators=(LEGGY_ACTUATORS,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": LEGGY_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device=device)
    model = scene.compile()

    # NOTE: For proper simulation with motor conversion and passive joints,
    # use the full environment with LeggyJointActionCfg instead of this viewer.
    print("Launching basic viewer. Passive joints will not be automatically updated.")
    print("For full functionality, run the environment with LeggyJointActionCfg.")

    with viewer.launch_passive(model, scene.data) as v:
        while v.is_running():
            mujoco.mj_step(model, scene.data)
            v.sync()
