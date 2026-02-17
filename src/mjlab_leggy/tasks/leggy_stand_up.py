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
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG, NUM_STEPS_PER_ENV
from mjlab_leggy.leggy.leggy_actions import LeggyJointActionCfg
from mjlab_leggy.leggy.leggy_observations import configure_leggy_observations
from mjlab_leggy.leggy.leggy_config import configure_leggy_base
from mjlab_leggy.leggy.leggy_rewards import (
    action_rate_running_adaptive,
    flight_penalty,
    mechanical_power,
    same_foot_penalty,
)


def leggy_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy stand up environment configuration."""
    cfg = make_velocity_env_cfg()

    # Set mujoco sim parameters to improve stability and collision detection
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70

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
    cfg.rewards["track_linear_velocity"].weight = 5.0
    # Reward for matching commanded angular velocity (turning rate)
    cfg.rewards["track_angular_velocity"].weight = 5.0

    # -- Pose and orientation --
    # Reward for keeping body upright (gravity aligned with body Z axis)
    cfg.rewards["upright"].weight = 1.0

    # Velocity-adaptive pose reward - uses standard 3-phase system (standing/walking/running)
    # Already configured by configure_leggy_base() with per-joint std parameters
    cfg.rewards["pose"].weight = 2.0
    # Set speed thresholds to match velocity curriculum (max 1.0 m/s)
    cfg.rewards["pose"].params["walking_threshold"] = 0.5  # Standing -> Walking transition
    cfg.rewards["pose"].params["running_threshold"] = 1.2  # Walking -> Running transition

    # Velocity-adaptive action rate - reduces penalty at high speeds
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=action_rate_running_adaptive,
        weight=-1.5,
        params={
            "command_name": "twist",
            "velocity_threshold": 0.5,
        },
    )

    # -- Gait and foot behavior --
    # Foot clearance during swing phase - promotes proper stepping
    cfg.rewards["foot_clearance"].weight = 2.5
    cfg.rewards["foot_clearance"].params["target_height"] = 0.05
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05
    # Minimum swing height - ensures feet lift properly
    cfg.rewards["foot_swing_height"].weight = 2.0
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.05
    # Air time tracking - encourages feet spending time in air
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.05

    # Same foot penalty - counts consecutive same-foot contacts.
    # Walking alternates feet → count stays at 1 → minimal penalty.
    # One-leg hopping → count grows 1, 2, 3... → penalty grows per hop.
    cfg.rewards["same_foot_penalty"] = RewardTermCfg(
        func=same_foot_penalty,
        weight=-1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    # Penalty for foot slipping on ground during contact
    cfg.rewards["foot_slip"].weight = -1.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.1

    # Flight penalty - penalizes both feet in the air at low speed.
    # Scales linearly: full penalty at 0 m/s, no penalty above 0.8 m/s.
    # Allows running with flight phases at high speed.
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
    # Penalize mechanical power (torque * velocity) to discourage high-energy
    # gaits like hopping and encourage efficient alternating steps
    cfg.rewards["mechanical_power"] = RewardTermCfg(
        func=mechanical_power,
        weight=-0.05,
    )

    # -- Regularization --
    # Penalty for body angular velocity - reduces unwanted spinning/wobbling
    cfg.rewards["body_ang_vel"].weight = -0.2
    # Penalty for angular momentum - reduces unwanted spinning/wobbling
    cfg.rewards["angular_momentum"].weight = -0.02

    # Configure command velocity curriculum
    # 8 stages over 40K iterations, linearly increasing from 0.1 to 1.0 m/s
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
        velocities = VELOCITY_STAGES_STANDARD[-1]
        cfg.commands["twist"].ranges.ang_vel_z = velocities["ang_vel_z"]
        cfg.commands["twist"].ranges.lin_vel_y = velocities["lin_vel_y"]
        cfg.commands["twist"].ranges.lin_vel_x = velocities["lin_vel_x"]
        cfg.commands["twist"].rel_standing_envs = 0.2
        cfg.commands["twist"].rel_heading_envs = 0.5

    return cfg


def leggy_stand_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Leggy locomotion task.

    Returns PPO training configuration with network architecture and hyperparameters.
    """
    return RslRlOnPolicyRunnerCfg(
        # ---------------------------------------------------------------------
        # Policy network architecture
        # ---------------------------------------------------------------------
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.5,  # Initial std for action noise during exploration
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
            learning_rate=5.0e-4,
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
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=50_000,
    )
