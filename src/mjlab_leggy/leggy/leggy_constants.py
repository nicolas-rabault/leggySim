import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
import numpy as np
from mjlab_leggy.leggy.leggy_actions import motor_to_knee, knee_to_motor

LEGGY_XML: Path = Path(os.path.dirname(__file__)) / "robot.xml"
assert LEGGY_XML.exists(), f"XML not found: {LEGGY_XML}"

NUM_STEPS_PER_ENV = 48


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LEGGY_XML))


stand_pose = {
    "hipY": 13 * np.pi / 180.0,  # 13 deg
    "hipX": -35 * np.pi / 180.0,  # -35 deg
    "kneeMotor": -38 * np.pi / 180.0,  # -38 deg
    "knee": (-38 + 35) * np.pi / 180.0,  # -3 deg = motor_to_knee(-38, -35)
}

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.175),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        # Equivalent motor-space formulation (uses kneeMotor + motor_to_knee):
        # ".*hipY.*": stand_pose["hipY"],
        # ".*hipX.*": stand_pose["hipX"],
        # ".*knee.*": motor_to_knee(stand_pose["kneeMotor"], stand_pose["hipX"]),
        # "LpassiveMotor": stand_pose["kneeMotor"],
        # "RpassiveMotor": stand_pose["kneeMotor"],
        # "Lpassive2": motor_to_knee(stand_pose["kneeMotor"], stand_pose["hipX"]),
        # "Rpassive2": motor_to_knee(stand_pose["kneeMotor"], stand_pose["hipX"]),

        ".*hipY.*": stand_pose["hipY"],
        ".*hipX.*": stand_pose["hipX"],
        ".*knee.*": stand_pose["knee"],
        "LpassiveMotor": knee_to_motor(stand_pose["knee"], stand_pose["hipX"]),
        "RpassiveMotor": knee_to_motor(stand_pose["knee"], stand_pose["hipX"]),
        "Lpassive2": stand_pose["knee"],
        "Rpassive2": stand_pose["knee"],
    },
    joint_vel={".*": 0.0},
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
)

LEGGY_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=("LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"),
)

LEGGY_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(LEGGY_ACTUATORS,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
