"""Leggy stand up environment.

Trains locomotion using progressive velocity curriculum:
- Stage 0 (0-5K iters): Almost standing still (0.1 m/s)
- Stage 1 (5K-10K): Moderate walking (0.8 m/s)
- Stage 2 (10K-15K): Fast walking/jogging (1.8 m/s)
- Stage 3 (15K+): Full running speed (up to 3.0 m/s!)

Customizing Velocity Curriculum:
    To use different velocity progression, import alternatives:

    from mjlab_leggy.leggy.leggy_curriculums import (
        VELOCITY_STAGES_CONSERVATIVE,  # Slower progression
        VELOCITY_STAGES_AGGRESSIVE,    # Faster progression
    )

    Then replace VELOCITY_STAGES_STANDARD in the curriculum config.
"""

from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG
from mjlab_leggy.leggy.leggy_actions import LeggyJointActionCfg
from mjlab_leggy.leggy.leggy_observations import configure_leggy_observations
from mjlab_leggy.leggy.leggy_config import configure_leggy_base
from mjlab_leggy.leggy.leggy_rewards import (
    air_time_both_feet,
    pose_running_adaptive,
    action_rate_running_adaptive,
)


def leggy_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy stand up environment configuration."""
    cfg = make_velocity_env_cfg()

    # Set mujoco sim parameters to improve stability and collision detection
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 45

    # Set control frequency to 100Hz (was 50Hz with decimation=4)                                                                                                                                               
    cfg.decimation = 2

    # Set leggy robot
    cfg.scene.entities = {"robot": LEGGY_ROBOT_CFG}

    # -------------------------------------------------------------------------
    # Custom Actions: Motor-to-knee conversion
    # -------------------------------------------------------------------------
    # LeggyJointAction converts motor commands to knee angles.
    # Passive joints are handled automatically by MuJoCo's constraint solver.
    # Scale of 0.5 provides more conservative movements for better stability
    cfg.actions = {
        "joint_pos": LeggyJointActionCfg(entity_name="robot", scale=0.5)
    }

    # -------------------------------------------------------------------------
    # Standard Leggy Configuration
    # -------------------------------------------------------------------------
    # Configure observations (motor space, IMU, corruption)
    configure_leggy_observations(cfg, enable_corruption=True)

    # Configure all standard Leggy base elements:
    # - Contact sensors (feet + leg collisions)
    # - Asset references (foot sites/geoms, body names)
    # - Pose reward (standing/walking/running modes)
    # - Motor limit reward (motor space)
    # - Leg collision penalty
    # - Flat terrain with high precision
    # - Standard terminations
    # - Push robot event
    # - Viewer
    configure_leggy_base(cfg)

    # -------------------------------------------------------------------------
    # Reward weights
    # -------------------------------------------------------------------------
    # Each reward term is: weight * reward_fn(env) * dt
    # Positive weights encourage behavior, negative weights penalize it.

    # -- Velocity tracking --
    # Reward for matching commanded linear velocity (forward/backward, left/right)
    cfg.rewards["track_linear_velocity"].weight = 15.0
    # Reward for matching commanded angular velocity (turning rate)
    cfg.rewards["track_angular_velocity"].weight = 8.0

    # -- Pose and orientation --
    # Reward for keeping body upright (gravity aligned with body Z axis)
    cfg.rewards["upright"].weight = 1.0

    # Velocity-adaptive pose reward - relaxes at high speeds
    cfg.rewards["pose"] = RewardTermCfg(
        func=pose_running_adaptive,
        weight=2.0,
        params={
            "command_name": "twist",
            "velocity_threshold": 0.8,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]),
        },
    )

    # Velocity-adaptive action rate - reduces penalty at high speeds
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=action_rate_running_adaptive,
        weight=-1.0,
        params={
            "command_name": "twist",
            "velocity_threshold": 1.0,
        },
    )

    # -- Gait and foot behavior --
    # Foot clearance during swing phase - promotes proper stepping
    cfg.rewards["foot_clearance"].weight = 1.0
    cfg.rewards["foot_clearance"].params["target_height"] = 0.03
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    # Minimum swing height - ensures feet lift properly
    cfg.rewards["foot_swing_height"].weight = 1.0
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.05
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    # Air time tracking - encourages slower gait with feet spending time in air
    cfg.rewards["air_time"].weight = 0.5
    cfg.rewards["air_time"].params["command_threshold"] = 0.05

    # Both feet airtime - rewards running gait (flight phase) at high speeds
    cfg.rewards["air_time_both_feet_running"] = RewardTermCfg(
        func=air_time_both_feet,
        weight=12.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "mode": "velocity",
            "velocity_threshold": 0.50,
        },
    )
    # Penalty for foot slipping on ground during contact
    cfg.rewards["foot_slip"].weight = -3.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0

    # -- Regularization --
    # Penalty for body angular velocity - reduces unwanted spinning/wobbling
    cfg.rewards["body_ang_vel"].weight = -0.05
    # Penalty for angular momentum - reduces unwanted spinning/wobbling
    cfg.rewards["angular_momentum"].weight = -0.02

    # Configure custom command velocity curriculum
    # Progressively increases velocity ranges: 0.1 → 0.8 → 1.8 → 3.0 m/s
    del cfg.curriculum["command_vel"]  # Remove default curriculum
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.tasks.velocity.mdp.curriculums import commands_vel
    from mjlab_leggy.leggy.leggy_curriculums import VELOCITY_STAGES_STANDARD

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": VELOCITY_STAGES_STANDARD,  # From leggy_curriculums.py
        },
    )

    # -------------------------------------------------------------------------
    # Play mode overrides
    # -------------------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)  # Effectively infinite episode length
        configure_leggy_observations(cfg, enable_corruption=False)  # Disable corruption in play
        cfg.events.pop("push_robot", None)
        cfg.events.pop("foot_friction", None)

        # Disable curriculum in play mode
        cfg.curriculum.pop("command_vel", None)

        # Set play mode twist ranges (will persist across episodes)
        cfg.commands["twist"].ranges.ang_vel_z = (-0.2, 0.2)
        cfg.commands["twist"].ranges.lin_vel_y = (-2.0, 0.3)
        cfg.commands["twist"].ranges.lin_vel_x = (-0.2, 0.2)
        cfg.commands["twist"].rel_standing_envs = 0.2
        cfg.commands["twist"].rel_heading_envs = 0.5

    return cfg


def leggy_stand_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Leggy stand up task.

    Returns PPO training configuration with network architecture and hyperparameters.
    """
    return RslRlOnPolicyRunnerCfg(
        # ---------------------------------------------------------------------
        # Policy network architecture
        # ---------------------------------------------------------------------
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,  # Initial std for action noise during exploration
            actor_obs_normalization=True,  # Enable normalization for numerical stability
            critic_obs_normalization=True,  # Enable normalization for numerical stability
            actor_hidden_dims=(512, 256, 128),  # Actor (policy) network layers
            critic_hidden_dims=(512, 256, 128),  # Critic (value) network layers
            activation="elu",
        ),
        # ---------------------------------------------------------------------
        # PPO algorithm hyperparameters
        # ---------------------------------------------------------------------
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,  # Weight of value loss (higher = more emphasis on critic)
            use_clipped_value_loss=True,
            clip_param=0.2,  # Policy ratio clipping - limits update magnitude
            entropy_coef=0.01,  # Encourages exploration (higher = more random)
            num_learning_epochs=5,  # Epochs per batch of experience
            num_mini_batches=4,  # Mini-batches per epoch
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,  # Discount factor (higher = more farsighted)
            lam=0.95,  # GAE lambda (higher = less bias, more variance)
            desired_kl=0.01,  # Target KL for adaptive learning rate
            max_grad_norm=1.0,  # Gradient clipping (prevents exploding gradients)
        ),
        # ---------------------------------------------------------------------
        # Training parameters
        # ---------------------------------------------------------------------
        experiment_name="leggy_stand_up",
        save_interval=500,  # Checkpoint saving interval (iterations)
        num_steps_per_env=24,  # Simulation steps before each policy update
        max_iterations=50_000,
    )
