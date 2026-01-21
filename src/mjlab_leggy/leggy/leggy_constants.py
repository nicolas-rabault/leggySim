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


HOME_FRAME = EntityCfg.InitialStateCfg(
    # Starting position (robot body center height)
    pos=(0.0, 0.0, 0.18),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        ".*hipY.*": 6 * np.pi / 180.0,
        ".*hipX.*": 25 * np.pi / 180.0,
        ".*knee.*": 45 * np.pi / 180.0
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

    # Use CPU on Mac (no CUDA support)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": LEGGY_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device=device)
    print(f"Using device: {device}")
    viewer.launch(scene.compile())
