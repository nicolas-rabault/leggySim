"""Leggy locomotion with Unitree G1-style rewards.

Uses standard mjlab velocity task rewards (tracking, clearance, slip, landing)
instead of custom gait rewards. Simpler reward structure that relies on
the reward functions to shape the gait naturally.
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
    dynamic_upright,
    flight_penalty,
    same_foot_penalty,
)


def leggy_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy environment with G1-style velocity rewards."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70
    cfg.decimation = 2  # 100Hz control

    cfg.scene.entities = {"robot": LEGGY_ROBOT_CFG}
    cfg.actions = {"joint_pos": LeggyJointActionCfg(entity_name="robot", scale=0.5)}

    configure_leggy_observations(cfg, enable_corruption=True)
    configure_leggy_base(cfg)

    # G1-style reward weights -- base defaults already match G1 for:
    # track_linear_velocity=2.0, track_angular_velocity=2.0, upright=1.0,
    # pose=1.0, dof_pos_limits=-1.0, action_rate_l2=-0.1, air_time=0.0,
    # foot_clearance=-2.0, foot_swing_height=-0.25, foot_slip=-0.1,
    # soft_landing=-1e-5
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.7
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.7
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["leg_collision_penalty"].weight = -1.0

    cfg.rewards["same_foot_penalty"] = RewardTermCfg(
        func=same_foot_penalty,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=action_rate_running_adaptive,
        weight=-1.0,
        params={"command_name": "twist", "velocity_threshold": 0.5},
    )

    cfg.rewards["flight_penalty"] = RewardTermCfg(
        func=flight_penalty,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "run_threshold": 0.8,
        },
    )

    cfg.rewards["upright"] = RewardTermCfg(
        func=dynamic_upright,
        weight=1.0,
        params={"command_name": "twist"},
    )

    # Leggy-specific foot params (smaller robot than G1)
    cfg.rewards["foot_clearance"].params["target_height"] = 0.05
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.08

    # Pose thresholds for speed-adaptive posture
    cfg.rewards["pose"].params["walking_threshold"] = 0.5
    cfg.rewards["pose"].params["running_threshold"] = 1.2

    # Velocity curriculum (G1-style 3 stages)
    del cfg.curriculum["command_vel"]
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": VELOCITY_STAGES_STANDARD,
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        configure_leggy_observations(cfg, enable_corruption=False)
        cfg.events.pop("push_robot", None)
        cfg.events.pop("foot_friction", None)
        cfg.events.pop("pd_gains", None)
        cfg.events.pop("joint_damping", None)
        cfg.events.pop("passive_armature", None)
        cfg.events.pop("joint_frictionloss", None)
        cfg.events.pop("body_mass", None)
        cfg.events.pop("effort_limits", None)
        cfg.curriculum.pop("command_vel", None)

        velocities = VELOCITY_STAGES_STANDARD[-1]
        cfg.commands["twist"].ranges.ang_vel_z = velocities["ang_vel_z"]
        cfg.commands["twist"].ranges.lin_vel_y = velocities["lin_vel_y"]
        cfg.commands["twist"].ranges.lin_vel_x = velocities["lin_vel_x"]
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].rel_heading_envs = 0.0

    return cfg


def leggy_run_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration (G1-style)."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
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
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.005,
            max_grad_norm=1.0,
        ),
        experiment_name="leggy_run",
        save_interval=500,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=100_000,
    )
