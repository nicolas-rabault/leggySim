import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
import numpy as np

LEGGY_XML: Path = Path(os.path.dirname(__file__)) / "robot.xml"
assert LEGGY_XML.exists(), f"XML not found: {LEGGY_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LEGGY_XML))


def update_passive_joints(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Convert motor→knee for actuators, then set passive joint angles."""
    # Actuator order: LhipY(0), LhipX(1), Lknee(2), RhipY(3), RhipX(4), Rknee(5)
    # Convert motor commands to knee angles
    data.ctrl[2] = MotorToKnee(data.ctrl[2], data.ctrl[1])
    data.ctrl[5] = MotorToKnee(data.ctrl[5], data.ctrl[4])
    
    # Set passive joints based on knee angles
    lknee = data.qpos[model.joint("Lknee").qposadr[0]]
    rknee = data.qpos[model.joint("Rknee").qposadr[0]]
    data.qpos[model.joint("Lpassive1").qposadr[0]] = lknee
    data.qpos[model.joint("Lpassive2").qposadr[0]] = lknee
    data.qpos[model.joint("Rpassive1").qposadr[0]] = rknee
    data.qpos[model.joint("Rpassive2").qposadr[0]] = rknee


def KneeToMotor(knee: float, hipX: float) -> float:
    return knee - hipX

def MotorToKnee(motor: float, hipX: float) -> float:
    return motor + hipX

def enable_passive_joint_callback() -> None:
    """Register MuJoCo control callback. Call once before training/running."""
    mujoco.set_mjcb_control(update_passive_joints)


stand_pose = {
    "hipY": 6 * np.pi / 180.0,
    "hipX": 25 * np.pi / 180.0,
    "knee": 45 * np.pi / 180.0
}

HOME_FRAME = EntityCfg.InitialStateCfg(
    # Starting position (robot body center height)
    pos=(0.0, 0.0, 0.18),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        ".*hipY.*": stand_pose["hipY"],
        ".*hipX.*": stand_pose["hipX"],#25
        ".*knee.*": stand_pose["knee"],#45 so delta is 45+25=70
        "Lpassive1": stand_pose["hipX"] + stand_pose["knee"], #70
        "Rpassive1": stand_pose["hipX"] + stand_pose["knee"],#70
        "Lpassive2": stand_pose["hipX"] + stand_pose["knee"], #70
        "Rpassive2": stand_pose["hipX"] + stand_pose["knee"] #70
    },
    joint_vel={".*": 0.0},
    # Explicitly set root (freejoint) velocities to zero
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
)

# Use existing position actuators defined in robot.xml
# (LhipY, LhipX, Lknee, RhipY, RhipX, Rknee with kp=10, forcerange=0.236)
LEGGY_ACTUATORS = XmlPositionActuatorCfg(
    joint_names_expr=("LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"),
)

LEGGY_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
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
    enable_passive_joint_callback()

    with viewer.launch_passive(model, scene.data) as v:
        while v.is_running():
            mujoco.mj_step(model, scene.data)
            v.sync()
