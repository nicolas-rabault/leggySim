import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

LEGGY_XML: Path = Path(os.path.dirname(__file__)) / "robot.xml"
assert LEGGY_XML.exists(), f"XML not found: {LEGGY_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LEGGY_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Left leg
        ".*LhipY.*": 0.10472,
        ".*LhipX.*": 0.523599,
        ".*Lknee.*": 0.523599,
        # Right leg
        ".*RhipY.*": 0.10472,
        ".*RhipX.*": 0.523599,
        ".*Rknee.*": 0.523599,
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

# Actuator configuration - using values from config.json
# kp=10, forcerange=0.236, armature=0.00006499, frictionloss=0.01
actuators = ActuatorCfg(
    joint_names_expr=[r".*"],
    stiffness=10.0,
    damping=1.0,  # Default damping ratio
    effort_limit=0.236,
    armature=0.00006499,
    frictionloss=0.01,
)

LEGGY_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    # Actuators are already defined in robot.xml, so we don't need to add them here
    # articulation=EntityArticulationInfoCfg(
    #     actuators=(actuators,),
    #     soft_joint_pos_limit_factor=0.9,
    # ),
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
