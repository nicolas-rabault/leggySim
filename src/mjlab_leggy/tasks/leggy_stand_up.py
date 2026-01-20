"""Leggy stand up environment"""

from dataclasses import dataclass

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from mjlab.tasks.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg


@dataclass
class LeggyStandUpEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Set leggy robot
        self.scene.entities = {"robot": LEGGY_ROBOT_CFG}

        # Configure viewer
        self.viewer.body_name = "trunclink"
        self.commands.twist.viz.z_offset = 1.0

        # Configure velocity command ranges (mostly standing for balance training)
        self.commands.twist.ranges.ang_vel_z = (-0.2, 0.2)  # Very small rotation
        self.commands.twist.ranges.lin_vel_y = (-0.1, 0.1)  # Very small lateral
        self.commands.twist.ranges.lin_vel_x = (-0.1, 0.1)  # Very small forward
        self.commands.twist.rel_standing_envs = 0.8  # 80% just standing
        self.commands.twist.rel_heading_envs = 0.0

        # Walking on plane only
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Disable curriculum
        self.curriculum.command_vel = None


@dataclass
class LeggyStandUpEnvCfg_PLAY(LeggyStandUpEnvCfg):
    """Play configuration with infinite episode length."""

    def __post_init__(self):
        super().__post_init__()
        # Infinite episode length for testing
        self.episode_length_s = int(1e9)


# RL training configuration
LeggyStandUpRlCfg = RslRlOnPolicyRunnerCfg(
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
    wandb_project="mjlab_leggy",
    experiment_name="leggy_stand_up",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
)
