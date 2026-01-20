"""Leggy stand up environment"""

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


# Leggy foot collision geom names
FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
FOOT_SITE_NAMES = ("left_foot", "right_foot")


def leggy_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy stand up environment configuration."""
    cfg = make_velocity_env_cfg()

    # Set leggy robot
    cfg.scene.entities = {"robot": LEGGY_ROBOT_CFG}

    # Add contact sensors for feet
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Note: Leggy only has foot collision geoms (left_foot_collision, right_foot_collision)
    # so we skip the nonfoot ground contact sensor
    cfg.scene.sensors = (feet_ground_cfg,)

    # Configure viewer
    cfg.viewer.body_name = "trunclink"

    # Configure velocity command ranges (mostly standing for balance training)
    twist_cmd = cfg.commands["twist"]
    twist_cmd.viz.z_offset = 1.0
    twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)  # Very small rotation
    twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)  # Very small lateral
    twist_cmd.ranges.lin_vel_x = (-0.1, 0.1)  # Very small forward
    twist_cmd.rel_standing_envs = 0.8  # 80% just standing
    twist_cmd.rel_heading_envs = 0.0

    # Update observations that reference foot sites
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Update events that reference foot geoms
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES

    # Update rewards that reference foot sites
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Update rewards that reference body names
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunclink",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunclink",)

    # Walking on plane only
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum (if present)
    if cfg.curriculum is not None and "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]

    # Apply play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)

        # Disable observation corruption and push events for play mode
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def leggy_stand_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Leggy stand up task."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="leggy_stand_up",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=50_000,
    )
