"""Reusable configuration functions for Leggy robot tasks.

Provides standard configuration for sensors, rewards, terrain, events, and terminations
that are common across all Leggy tasks.
"""

from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from .leggy_rewards import joint_pos_limits_motor, leg_collision_penalty


# -------------------------------------------------------------------------
# Geometry References (from robot.xml)
# -------------------------------------------------------------------------

# Foot geometry and site names
FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
FOOT_SITE_NAMES = ("left_foot", "right_foot")
BODY_NAME = "boddy"  # Main body name in robot.xml


# -------------------------------------------------------------------------
# Contact Sensors Configuration
# -------------------------------------------------------------------------

def configure_contact_sensors(cfg):
    """Configure foot and leg collision contact sensors.

    Sets up:
    - Foot-ground contact sensors with air time tracking
    - Leg self-collision sensors (9 sensor pairs for complete coverage)

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.scene.sensors - Adds all contact sensors
    """
    # Foot-ground contact sensor
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Leg self-collision sensors
    # Each sensor monitors one specific geom pair
    # Left leg: left_femur_collision, left_tibia_collision, left_rod_collision
    # Right leg: right_femur_collision, right_tibia_collision, right_rod_collision

    leg_collision_sensors = []

    # Left tibia vs right leg parts
    for right_part in ["tibia", "femur", "rod"]:
        leg_collision_sensors.append(
            ContactSensorCfg(
                name=f"leg_collision_ltibia_r{right_part}",
                primary=ContactMatch(mode="geom", pattern="left_tibia_collision", entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=f"right_{right_part}_collision", entity="robot"),
                fields=("found",),
                reduce="maxforce",
                num_slots=1,
            )
        )

    # Left femur vs right leg parts
    for right_part in ["tibia", "femur", "rod"]:
        leg_collision_sensors.append(
            ContactSensorCfg(
                name=f"leg_collision_lfemur_r{right_part}",
                primary=ContactMatch(mode="geom", pattern="left_femur_collision", entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=f"right_{right_part}_collision", entity="robot"),
                fields=("found",),
                reduce="maxforce",
                num_slots=1,
            )
        )

    # Left rod vs right leg parts
    for right_part in ["tibia", "femur", "rod"]:
        leg_collision_sensors.append(
            ContactSensorCfg(
                name=f"leg_collision_lrod_r{right_part}",
                primary=ContactMatch(mode="geom", pattern="left_rod_collision", entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=f"right_{right_part}_collision", entity="robot"),
                fields=("found",),
                reduce="maxforce",
                num_slots=1,
            )
        )

    cfg.scene.sensors = (feet_ground_cfg, *leg_collision_sensors)


# -------------------------------------------------------------------------
# Asset References Configuration
# -------------------------------------------------------------------------

def configure_asset_references(cfg):
    """Configure asset references for Leggy's geometry.

    Updates all reward and event configurations to use correct Leggy geometry names.

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.observations - Updates foot height observations
        cfg.events - Updates foot friction events
        cfg.rewards - Updates foot and body references
    """
    # Foot sites for critic observations
    if "foot_height" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Foot geoms for friction randomization
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
        cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)  # min, max friction

    # Foot sites for gait rewards
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Body name for orientation rewards
    for reward_name in ["upright", "body_ang_vel"]:
        if reward_name in cfg.rewards and hasattr(cfg.rewards[reward_name].params.get("asset_cfg", None), "body_names"):
            cfg.rewards[reward_name].params["asset_cfg"].body_names = (BODY_NAME,)


# -------------------------------------------------------------------------
# Pose Reward Configuration
# -------------------------------------------------------------------------

def configure_pose_reward(cfg):
    """Configure pose reward parameters for different locomotion modes.

    Sets up pose constraints for standing, walking, and running modes.

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.rewards["pose"] - Adds mode-specific std parameters
    """
    if "pose" not in cfg.rewards:
        return

    # Limit to actuated joints only (exclude passive joints)
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (
        "LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"
    )

    # Standing mode: tight constraints for balance
    cfg.rewards["pose"].params["std_standing"] = {
        ".*hipY.*": 0.15,  # HipY needs tight control
        ".*hipX.*": 0.2,  # HipX needs tight control
        ".*knee.*": 0.2,  # Knee tight to prevent crouching
    }

    # Walking mode: medium constraints for locomotion
    cfg.rewards["pose"].params["std_walking"] = {
        ".*hipY.*": 0.2,
        ".*hipX.*": 0.6,
        ".*knee.*": 0.8,
    }

    # Running mode: relaxed constraints for dynamic motion
    cfg.rewards["pose"].params["std_running"] = {
        ".*hipY.*": 0.4,
        ".*hipX.*": 1.0,
        ".*knee.*": 1.2,
    }


# -------------------------------------------------------------------------
# Motor Limit and Collision Rewards
# -------------------------------------------------------------------------

def configure_motor_limit_reward(cfg):
    """Configure motor limit penalty (checks limits in motor space).

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.rewards["dof_pos_limits"] - Replaces with motor-space version
    """
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=joint_pos_limits_motor,
        weight=-1.0
    )


def configure_leg_collision_penalty(cfg):
    """Configure leg collision penalty (soft constraint before termination).

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.rewards["leg_collision_penalty"] - Adds collision penalty
    """
    cfg.rewards["leg_collision_penalty"] = RewardTermCfg(
        func=leg_collision_penalty,
        weight=-5.0,
        params={"sensor_name": "leg_collision_ltibia_rtibia"},
    )


# -------------------------------------------------------------------------
# Terrain Configuration
# -------------------------------------------------------------------------

def configure_flat_terrain(cfg):
    """Configure flat plane terrain with high precision collision.

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.scene.terrain - Sets up flat plane with tuned physics
        cfg.curriculum - Disables terrain curriculum
    """
    assert cfg.scene.terrain is not None

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.friction = "1.5 0.005 0.0001"  # Higher friction
    cfg.scene.terrain.solref = "0.005 1"  # Stiffer contact, less bouncing
    cfg.scene.terrain.solimp = "0.995 0.9995 0.001 0.5 2"  # Less penetration, precise collision
    cfg.scene.terrain.contact = "enable"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]


# -------------------------------------------------------------------------
# Standard Terminations
# -------------------------------------------------------------------------

def configure_standard_terminations(cfg):
    """Configure standard termination conditions.

    Sets up:
    - Body below floor termination
    - Bad orientation (fall over) termination (already in default config)

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.terminations - Adds floor penetration check
    """
    cfg.terminations["body_below_floor"] = TerminationTermCfg(
        func=mdp_terminations.root_height_below_minimum,
        params={"minimum_height": 0.00},  # meters
    )


# -------------------------------------------------------------------------
# Event Randomization
# -------------------------------------------------------------------------

def configure_push_robot_event(cfg, velocity_range: dict | None = None):
    """Configure robot push event for robustness training.

    Args:
        cfg: Environment configuration
        velocity_range: Dict with 'x' and 'y' velocity ranges.
                       Defaults to (-0.8, 0.8) for both axes.

    Modifies:
        cfg.events["push_robot"] - Sets velocity range
    """
    if velocity_range is None:
        velocity_range = {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}

    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = velocity_range


# -------------------------------------------------------------------------
# Viewer Configuration
# -------------------------------------------------------------------------

def configure_viewer(cfg):
    """Configure viewer to track robot body.

    Args:
        cfg: Environment configuration

    Modifies:
        cfg.viewer.body_name - Sets to main body
    """
    cfg.viewer.body_name = BODY_NAME


# -------------------------------------------------------------------------
# All-in-One Configuration
# -------------------------------------------------------------------------

def configure_leggy_base(cfg):
    """Configure all standard Leggy base elements.

    Applies all common configurations:
    - Contact sensors
    - Asset references
    - Pose reward
    - Motor limit reward
    - Leg collision penalty
    - Flat terrain
    - Standard terminations
    - Push robot event
    - Viewer

    Args:
        cfg: Environment configuration

    This is a convenience function that calls all individual config functions.
    Use this for consistent base setup across all Leggy tasks.
    """
    configure_contact_sensors(cfg)
    configure_viewer(cfg)
    configure_asset_references(cfg)
    configure_pose_reward(cfg)
    configure_motor_limit_reward(cfg)
    configure_leg_collision_penalty(cfg)
    configure_flat_terrain(cfg)
    configure_standard_terminations(cfg)
    configure_push_robot_event(cfg)


__all__ = [
    # Geometry constants
    "FOOT_GEOM_NAMES",
    "FOOT_SITE_NAMES",
    "BODY_NAME",
    # Individual configuration functions
    "configure_contact_sensors",
    "configure_asset_references",
    "configure_pose_reward",
    "configure_motor_limit_reward",
    "configure_leg_collision_penalty",
    "configure_flat_terrain",
    "configure_standard_terminations",
    "configure_push_robot_event",
    "configure_viewer",
    # All-in-one function
    "configure_leggy_base",
]
