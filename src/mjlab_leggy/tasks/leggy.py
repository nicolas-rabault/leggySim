"""Leggy locomotion environment.

Trains standing, walking, and running using progressive velocity curriculum:
- 8 stages over 40K iterations
- Linearly increases from 0.1 m/s (standing) to 1.0 m/s (running)
- Final ranges: lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)

Pose reward automatically adapts based on speed:
- Standing mode (< 0.5 m/s): Relaxed pose constraints
- Walking mode (0.5-1.2 m/s): Medium constraints
- Running mode (> 1.2 m/s): Relaxed constraints for dynamic motion
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG, NUM_STEPS_PER_ENV
from mjlab_leggy.leggy.leggy_actions import LeggyJointActionCfg
from mjlab_leggy.leggy.leggy_observations import configure_leggy_observations
from mjlab_leggy.leggy.leggy_config import configure_leggy_base
from mjlab_leggy.leggy.leggy_curriculums import VELOCITY_STAGES_STANDARD
from mjlab_leggy.leggy.leggy_rewards import (
    action_rate_running_adaptive,
    flight_penalty,
    forward_symmetry,
    gait_symmetry,
    mechanical_power,
    same_foot_penalty,
)


def leggy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy environment configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70
    cfg.decimation = 2  # 100Hz control frequency

    cfg.scene.entities = {"robot": LEGGY_ROBOT_CFG}

    cfg.actions = {
        "joint_pos": LeggyJointActionCfg(entity_name="robot", scale=0.5)
    }

    configure_leggy_observations(cfg, enable_corruption=True)
    configure_leggy_base(cfg)

    # -- Velocity tracking --
    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_angular_velocity"].weight = 5.0

    # -- Pose and orientation --
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["pose"].weight = 2.0
    cfg.rewards["pose"].params["walking_threshold"] = 0.5
    cfg.rewards["pose"].params["running_threshold"] = 1.2

    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=action_rate_running_adaptive,
        weight=-1.5,
        params={"command_name": "twist", "velocity_threshold": 0.5},
    )

    # -- Gait and foot behavior --
    cfg.rewards["foot_clearance"].weight = 2.5
    cfg.rewards["foot_clearance"].params["target_height"] = 0.05
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05
    cfg.rewards["foot_swing_height"].weight = 2.0
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.05
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.05

    cfg.rewards["same_foot_penalty"] = RewardTermCfg(
        func=same_foot_penalty,
        weight=-1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    cfg.rewards["foot_slip"].weight = -1.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.1

    cfg.rewards["flight_penalty"] = RewardTermCfg(
        func=flight_penalty,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "run_threshold": 0.8,
        },
    )

    # -- Energy efficiency --
    cfg.rewards["mechanical_power"] = RewardTermCfg(
        func=mechanical_power,
        weight=-0.05,
    )

    # -- Gait symmetry --
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=gait_symmetry,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0,
        },
    )

    cfg.rewards["forward_symmetry"] = RewardTermCfg(
        func=forward_symmetry,
        weight=-2.0,
        params={
            "command_name": "twist",
            "command_threshold": 0,
            "alpha": 0.01,
        },
    )

    # -- Regularization --
    cfg.rewards["body_ang_vel"].weight = -0.2
    cfg.rewards["angular_momentum"].weight = -0.02

    # Velocity curriculum
    del cfg.curriculum["command_vel"]
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": VELOCITY_STAGES_STANDARD,
        },
    )

    # -- Play mode overrides --
    if play:
        cfg.episode_length_s = int(1e9)
        configure_leggy_observations(cfg, enable_corruption=False)
        cfg.events.pop("push_robot", None)
        cfg.events.pop("foot_friction", None)
        cfg.curriculum.pop("command_vel", None)

        velocities = VELOCITY_STAGES_STANDARD[-1]
        cfg.commands["twist"].ranges.ang_vel_z = velocities["ang_vel_z"]
        cfg.commands["twist"].ranges.lin_vel_y = velocities["lin_vel_y"]
        cfg.commands["twist"].ranges.lin_vel_x = velocities["lin_vel_x"]
        cfg.commands["twist"].rel_standing_envs = 0.2
        cfg.commands["twist"].rel_heading_envs = 0.5

    return cfg


def leggy_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Leggy."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.5,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
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
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="leggy",
        save_interval=500,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=50_000,
    )
