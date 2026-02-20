"""Reusable configuration functions for Leggy robot tasks."""

from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from .leggy_rewards import joint_pos_limits_motor, leg_collision_penalty

FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
FOOT_SITE_NAMES = ("left_foot", "right_foot")
BODY_NAME = "boddy"


def configure_contact_sensors(cfg):
    """Configure foot-ground and leg collision contact sensors."""
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    leg_collision_sensors = []
    left_parts = ["tibia", "femur", "rod"]
    right_parts = ["tibia", "femur", "rod"]

    for left_part in left_parts:
        for right_part in right_parts:
            leg_collision_sensors.append(
                ContactSensorCfg(
                    name=f"leg_collision_l{left_part}_r{right_part}",
                    primary=ContactMatch(mode="geom", pattern=f"left_{left_part}_collision", entity="robot"),
                    secondary=ContactMatch(mode="geom", pattern=f"right_{right_part}_collision", entity="robot"),
                    fields=("found",),
                    reduce="maxforce",
                    num_slots=1,
                )
            )

    cfg.scene.sensors = (feet_ground_cfg, *leg_collision_sensors)


def configure_asset_references(cfg):
    """Configure asset references for Leggy's geometry."""
    if "foot_height" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
        cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    for reward_name in ["upright", "body_ang_vel"]:
        if reward_name in cfg.rewards and hasattr(cfg.rewards[reward_name].params.get("asset_cfg", None), "body_names"):
            cfg.rewards[reward_name].params["asset_cfg"].body_names = (BODY_NAME,)


def configure_pose_reward(cfg):
    """Configure velocity-adaptive pose reward with per-joint std parameters."""
    if "pose" not in cfg.rewards:
        return

    cfg.rewards["pose"].params["asset_cfg"].joint_names = (
        "LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"
    )

    cfg.rewards["pose"].params["std_standing"] = {
        ".*hipY.*": 0.4, ".*hipX.*": 0.5, ".*knee.*": 0.6,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        ".*hipY.*": 0.5, ".*hipX.*": 0.8, ".*knee.*": 1.2,
    }
    cfg.rewards["pose"].params["std_running"] = {
        ".*hipY.*": 0.4, ".*hipX.*": 1.0, ".*knee.*": 1.2,
    }


def configure_motor_limit_reward(cfg):
    """Configure motor limit penalty (checks limits in motor space)."""
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(func=joint_pos_limits_motor, weight=-1.0)


def configure_leg_collision_penalty(cfg):
    """Configure leg collision penalty."""
    cfg.rewards["leg_collision_penalty"] = RewardTermCfg(
        func=leg_collision_penalty,
        weight=-2.0,
        params={"sensor_name": "leg_collision_ltibia_rtibia"},
    )


def configure_flat_terrain(cfg):
    """Configure flat plane terrain with tuned contact physics."""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.friction = "1.5 0.005 0.0001"
    cfg.scene.terrain.solref = "0.005 1"
    cfg.scene.terrain.solimp = "0.995 0.9995 0.001 0.5 2"
    cfg.scene.terrain.contact = "enable"
    cfg.scene.terrain.terrain_generator = None

    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]


def configure_standard_terminations(cfg):
    """Configure body-below-floor termination."""
    cfg.terminations["body_below_floor"] = TerminationTermCfg(
        func=mdp_terminations.root_height_below_minimum,
        params={"minimum_height": 0.00},
    )


def configure_push_robot_event(cfg, velocity_range=None):
    """Configure robot push event for robustness training."""
    if velocity_range is None:
        velocity_range = {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = velocity_range


def configure_viewer(cfg):
    """Configure viewer to track robot body."""
    cfg.viewer.body_name = BODY_NAME


def configure_leggy_base(cfg):
    """Configure all standard Leggy base elements."""
    configure_contact_sensors(cfg)
    configure_viewer(cfg)
    configure_asset_references(cfg)
    configure_pose_reward(cfg)
    configure_motor_limit_reward(cfg)
    configure_leg_collision_penalty(cfg)
    configure_flat_terrain(cfg)
    configure_standard_terminations(cfg)
    configure_push_robot_event(cfg)
