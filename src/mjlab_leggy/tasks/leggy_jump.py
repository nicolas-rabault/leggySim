"""Leggy jump + run training environment.

Combines locomotion with jumping using progressive curriculum:
1. Stage 0-3 (0-15K iters): Learn to walk/run (velocity curriculum)
2. Stage 4 (15K-20K): Introduce small jumps while running
3. Stage 5 (20K-25K): Increase jump height and frequency
4. Stage 6 (25K+): Full jumping + running capability
"""

from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

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
# Jump Command Curriculum
# -------------------------------------------------------------------------

def jump_command_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    jump_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Progressively enable jumping based on training stage.

    Args:
        env: Environment instance
        env_ids: Environment IDs (unused)
        command_name: Name of jump command
        jump_stages: List of stage configurations

    Returns:
        Dictionary of curriculum metrics
    """
    del env_ids  # Unused
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = command_term.cfg

    # Update jump parameters based on current step
    for stage in jump_stages:
        if env.common_step_counter > stage["step"]:
            if "jump_probability" in stage:
                cfg.jump_probability = stage["jump_probability"]
            if "jump_height_range" in stage:
                cfg.jump_height_range = stage["jump_height_range"]

    return {
        "jump_probability": torch.tensor(cfg.jump_probability),
        "jump_height_max": torch.tensor(cfg.jump_height_range[1]),
    }


# -------------------------------------------------------------------------
# Reward Weight Curriculum
# -------------------------------------------------------------------------

def reward_weight_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Adjust multiple reward weights based on training stage.

    Args:
        env: Environment instance
        env_ids: Environment IDs (unused)
        reward_stages: List of stage configurations with reward weights

    Returns:
        Dictionary of current reward weights
    """
    del env_ids  # Unused

    # Update reward weights based on current step
    for stage in reward_stages:
        if env.common_step_counter > stage["step"]:
            for reward_name, weight in stage.items():
                if reward_name != "step" and reward_name in env.reward_manager._terms:
                    env.reward_manager._terms[reward_name].weight = weight

    # Return current weights for logging
    metrics = {}
    for name, term in env.reward_manager._terms.items():
        if name.startswith(("jump", "vertical", "air_time", "landing")):
            metrics[f"weight_{name}"] = torch.tensor(term.weight)

    return metrics


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
    # Start with standing configuration (includes locomotion)
    cfg = leggy_stand_up_env_cfg(play=play)

    # -------------------------------------------------------------------------
    # Jump Command Manager
    # -------------------------------------------------------------------------
    cfg.commands["jump_command"] = JumpCommandCfg()

    # -------------------------------------------------------------------------
    # Observations: Add vertical velocity and jump command
    # -------------------------------------------------------------------------
    # Vertical velocity helps with jump control and landing
    cfg.observations["policy"].terms["base_lin_vel_z"] = ObservationTermCfg(
        func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2:3]
    )

    # Jump command to policy
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
    # Already configured in stand_up_env_cfg at lines 361-402

    # Add jump command curriculum (introduces jumping progressively)
    cfg.curriculum["jump_command"] = CurriculumTermCfg(
        func=jump_command_curriculum,
        params={
            "command_name": "jump_command",
            "jump_stages": [
                # Stage 0-3: No jumping, focus on locomotion (0-15K iterations)
                {
                    "step": 0,
                    "jump_probability": 0.0,
                    "jump_height_range": (0.0, 0.0),
                },
                # Stage 4: Introduce small jumps (15K-20K iterations = 360K-480K steps)
                {
                    "step": 15000 * 24,
                    "jump_probability": 0.1,  # 10% of environments jump
                    "jump_height_range": (0.3, 0.5),  # Small jumps (3-5cm)
                },
                # Stage 5: Increase jump frequency and height (20K-25K iterations)
                {
                    "step": 20000 * 24,
                    "jump_probability": 0.2,  # 20% jump
                    "jump_height_range": (0.4, 0.7),  # Medium jumps (4-7cm)
                },
                # Stage 6: Full jumping capability (25K+ iterations)
                {
                    "step": 25000 * 24,
                    "jump_probability": 0.3,  # 30% jump
                    "jump_height_range": (0.5, 1.0),  # Full range (5-10cm)
                },
            ],
        },
    )

    # Add reward weight curriculum (enables jump rewards progressively)
    cfg.curriculum["reward_weights"] = CurriculumTermCfg(
        func=reward_weight_curriculum,
        params={
            "reward_stages": [
                # Stage 0-3: Locomotion only (0-15K iterations)
                {
                    "step": 0,
                    "vertical_velocity": 0.0,
                    "joint_extension": 0.0,
                    "leg_coordination": 0.0,
                    "air_time_both": 0.0,
                    "jump_height": 0.0,
                    "landing_stability": 0.0,
                    "soft_landing": 0.0,
                    # Locomotion rewards stay at default
                    "pose": 3.5,
                    "action_rate_l2": -2.0,
                },
                # Stage 4: Enable basic jump rewards (15K-20K iterations)
                {
                    "step": 15000 * 24,
                    "vertical_velocity": 2.0,
                    "joint_extension": 0.5,
                    "leg_coordination": 1.5,
                    "air_time_both": 5.0,  # Main focus
                    "jump_height": 3.0,
                    "landing_stability": 3.0,
                    "soft_landing": 2.0,
                    # Relax constraints for jumping
                    "pose": 2.5,
                    "action_rate_l2": -1.0,
                },
                # Stage 5: Increase jump rewards (20K-25K iterations)
                {
                    "step": 20000 * 24,
                    "vertical_velocity": 3.0,
                    "joint_extension": 1.0,
                    "leg_coordination": 2.0,
                    "air_time_both": 8.0,
                    "jump_height": 6.0,  # Encourage higher jumps
                    "landing_stability": 4.0,
                    "soft_landing": 3.0,
                    "pose": 2.0,
                    "action_rate_l2": -0.5,
                },
                # Stage 6: Optimize jumping (25K+ iterations)
                {
                    "step": 25000 * 24,
                    "vertical_velocity": 2.5,
                    "joint_extension": 1.0,
                    "leg_coordination": 2.5,
                    "air_time_both": 10.0,
                    "jump_height": 8.0,  # Maximum jump height emphasis
                    "landing_stability": 5.0,  # Must maintain landing
                    "soft_landing": 3.0,
                    "pose": 2.0,
                    "action_rate_l2": -0.5,
                },
            ],
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
        cfg.observations["policy"].enable_corruption = False
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
