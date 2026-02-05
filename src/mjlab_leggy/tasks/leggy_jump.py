"""Leggy jump + run training environment.

Combines locomotion with jumping using progressive curriculum:
1. Stage 0-3 (0-15K iters): Learn to walk/run (velocity curriculum)
2. Stage 4 (15K-20K): Introduce small jumps while running
3. Stage 5 (20K-25K): Increase jump height and frequency
4. Stage 6 (25K+): Full jumping + running capability

Customizing Curriculum:
    This task uses TWO curricula (both from leggy_curriculums):
    1. VELOCITY_STAGES_STANDARD (inherited from stand_up) - for running
    2. JUMP_STAGES_STANDARD - for jumping

    To customize, import alternatives:

    from mjlab_leggy.leggy.leggy_curriculums import (
        VELOCITY_STAGES_AGGRESSIVE,  # Faster locomotion progression
        JUMP_STAGES_AGGRESSIVE,      # Earlier jump start
        JUMP_STAGES_CONSERVATIVE,    # Later jump start
        make_jump_stages,            # Create custom jump stages
        make_reward_stages,          # Create custom reward stages
    )

    Then replace the stage constants in the curriculum config.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_leggy.leggy.leggy_curriculums import (
    jump_command_curriculum,
    reward_weight_curriculum,
    JUMP_STAGES_STANDARD,
    REWARD_STAGES_STANDARD,
)
from mjlab_leggy.leggy.leggy_rewards import (
    vertical_velocity_reward,
    joint_extension_speed,
    leg_coordination,
    air_time_both_feet,
    jump_height_reward,
    landing_stability,
    soft_landing_bonus,
)

from .leggy_stand_up import leggy_stand_up_env_cfg


# -------------------------------------------------------------------------
# Custom Command Term for Jump
# -------------------------------------------------------------------------

class JumpCommandCfg(CommandTermCfg):
    """Configuration for jump command.

    Generates jump height commands (0 = no jump, 0-1 = jump height target).
    """

    class_type: type = None  # Will be set to JumpCommand

    # Jump command ranges (will be adjusted by curriculum)
    jump_height_range: tuple[float, float] = (0.0, 1.0)  # 0-1 maps to 0-10cm
    jump_probability: float = 0.0  # Start at 0%, curriculum will increase


class JumpCommand:
    """Jump command generator.

    Generates jump height commands with specified probability.
    """

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
        self.cfg = cfg
        self._env = env

        # Initialize command buffer (num_envs, 1)
        self.command = torch.zeros(env.num_envs, 1, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset jump commands for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)

        # Sample jump commands based on jump_probability
        should_jump = torch.rand(len(env_ids), device=self._env.device) < self.cfg.jump_probability

        # Sample jump heights from range
        jump_heights = torch.rand(len(env_ids), device=self._env.device) * (
            self.cfg.jump_height_range[1] - self.cfg.jump_height_range[0]
        ) + self.cfg.jump_height_range[0]

        self.command[env_ids, 0] = torch.where(should_jump, jump_heights, 0.0)

    def compute(self) -> torch.Tensor:
        """Return current jump command."""
        return self.command


# Set the class type
JumpCommandCfg.class_type = JumpCommand


# -------------------------------------------------------------------------
# Main environment configuration
# -------------------------------------------------------------------------

def leggy_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy jump + run environment configuration.

    Combines locomotion training with jumping capability.
    Uses progressive curriculum to first teach running, then add jumping.

    Args:
        play: If True, configure for inference/visualization mode.

    Returns:
        Environment configuration for jump + run training.
    """
    # Start with standing configuration (includes standard Leggy observations)
    cfg = leggy_stand_up_env_cfg(play=play)

    # -------------------------------------------------------------------------
    # Jump Command Manager
    # -------------------------------------------------------------------------
    cfg.commands["jump_command"] = JumpCommandCfg()

    # -------------------------------------------------------------------------
    # Add Jump-Specific Observations
    # -------------------------------------------------------------------------
    # Note: Standard Leggy observations (motor space, IMU, etc.) are already
    # configured by leggy_stand_up_env_cfg via configure_leggy_observations()

    # Add vertical velocity (crucial for jump control and landing)
    cfg.observations["policy"].terms["base_lin_vel_z"] = ObservationTermCfg(
        func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2:3]
    )

    # Add jump command to policy observations
    cfg.observations["policy"].terms["jump_command"] = ObservationTermCfg(
        func=mdp_obs.generated_commands,
        params={"command_name": "jump_command"}
    )

    # -------------------------------------------------------------------------
    # Jump-Specific Rewards (initially disabled, enabled by curriculum)
    # -------------------------------------------------------------------------

    # Vertical velocity during jump
    cfg.rewards["vertical_velocity"] = RewardTermCfg(
        func=vertical_velocity_reward,
        weight=0.0,  # Curriculum will enable
        params={"command_name": "jump_command"}
    )

    # Joint extension speed
    cfg.rewards["joint_extension"] = RewardTermCfg(
        func=joint_extension_speed,
        weight=0.0,
        params={"command_name": "jump_command"}
    )

    # Leg coordination (important for jumping)
    cfg.rewards["leg_coordination"] = RewardTermCfg(
        func=leg_coordination,
        weight=0.0
    )

    # Air time with both feet (core jump reward)
    cfg.rewards["air_time_both"] = RewardTermCfg(
        func=air_time_both_feet,
        weight=0.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "jump_command"
        }
    )

    # Jump height reward
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=jump_height_reward,
        weight=0.0,
        params={
            "standing_height": 0.18,
            "command_name": "jump_command"
        }
    )

    # Landing stability
    cfg.rewards["landing_stability"] = RewardTermCfg(
        func=landing_stability,
        weight=0.0,
        params={"sensor_name": "feet_ground_contact"}
    )

    # Soft landing bonus
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=soft_landing_bonus,
        weight=0.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_force_threshold": 15.0  # Higher threshold for running jumps
        }
    )

    # -------------------------------------------------------------------------
    # Adjust Existing Rewards for Combined Locomotion + Jumping
    # -------------------------------------------------------------------------

    # Keep locomotion rewards active (velocity tracking)
    # These will remain throughout training
    cfg.rewards["track_linear_velocity"].weight = 8.0
    cfg.rewards["track_angular_velocity"].weight = 8.0

    # Slightly relax action smoothness to allow explosive jumps
    cfg.rewards["action_rate_l2"].weight = -1.5  # Was -2.0

    # Keep gait rewards for locomotion (will be adjusted by curriculum)
    cfg.rewards["foot_clearance"].weight = 1.0
    cfg.rewards["foot_swing_height"].weight = 1.0
    cfg.rewards["air_time"].weight = 1.0  # Regular gait air time
    cfg.rewards["foot_slip"].weight = -6.0

    # Keep pose reward (will be relaxed during jumping by curriculum)
    cfg.rewards["pose"].weight = 3.5

    # -------------------------------------------------------------------------
    # Curriculum Configuration
    # -------------------------------------------------------------------------
    # Keep the velocity curriculum from stand_up (teaches running)
    # Already configured in stand_up_env_cfg

    # Add jump command curriculum (introduces jumping progressively)
    # Uses standard stages: 0% → 10% → 20% → 30% jumping from 15K-25K iterations
    cfg.curriculum["jump_command"] = CurriculumTermCfg(
        func=jump_command_curriculum,
        params={
            "command_name": "jump_command",
            "jump_stages": JUMP_STAGES_STANDARD,  # From leggy_curriculums.py
        },
    )

    # Add reward weight curriculum (enables jump rewards progressively)
    # Coordinates with jump stages to enable rewards as jumping is introduced
    cfg.curriculum["reward_weights"] = CurriculumTermCfg(
        func=reward_weight_curriculum,
        params={
            "reward_stages": REWARD_STAGES_STANDARD,  # From leggy_curriculums.py
        },
    )

    # -------------------------------------------------------------------------
    # Episode Length (keep at 10s for mixed locomotion + jumping)
    # -------------------------------------------------------------------------
    cfg.episode_length_s = 10.0  # Same as stand_up

    # -------------------------------------------------------------------------
    # Play mode overrides
    # -------------------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        # Observation corruption already disabled by stand_up play mode
        cfg.events.pop("push_robot", None)
        cfg.events.pop("foot_friction", None)

        # Disable curriculum in play mode
        cfg.curriculum.pop("command_vel", None)
        cfg.curriculum.pop("jump_command", None)
        cfg.curriculum.pop("reward_weights", None)

        # Set play mode commands
        cfg.commands["twist"].ranges.ang_vel_z = (-0.2, 0.2)
        cfg.commands["twist"].ranges.lin_vel_y = (-3.0, 0.3)
        cfg.commands["twist"].ranges.lin_vel_x = (-0.2, 0.2)
        cfg.commands["twist"].rel_standing_envs = 0.3  # 30% standing

        # Enable jumping in play mode
        cfg.commands["jump_command"].jump_probability = 0.3  # 30% jump
        cfg.commands["jump_command"].jump_height_range = (0.5, 1.0)

    return cfg


# -------------------------------------------------------------------------
# RL Configuration (same as stand_up, optimized for combined task)
# -------------------------------------------------------------------------

def leggy_jump_rl_cfg():
    """Create RL runner configuration for Leggy jump + run task.

    Uses same hyperparameters as stand_up since we're learning both skills together.
    """
    from .leggy_stand_up import leggy_stand_up_rl_cfg

    cfg = leggy_stand_up_rl_cfg()

    # Update experiment name
    cfg.experiment_name = "leggy_jump_run"

    # Extend training duration for full curriculum
    cfg.max_iterations = 30_000  # Extended for jumping stages

    return cfg
