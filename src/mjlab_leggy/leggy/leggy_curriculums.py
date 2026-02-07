"""Curriculum functions and stage definitions for Leggy training.

Provides reusable curriculum implementations that can be configured
in task files with simple stage definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# -------------------------------------------------------------------------
# Curriculum Functions
# -------------------------------------------------------------------------

def jump_command_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    jump_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Progressively enable jumping based on training stage.

    Updates jump command configuration dynamically during training.

    Args:
        env: Environment instance
        env_ids: Environment IDs (unused)
        command_name: Name of jump command term
        jump_stages: List of stage dicts with 'step', 'jump_probability', 'jump_height_range'

    Returns:
        Dictionary of curriculum metrics for logging
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


def reward_weight_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Adjust multiple reward weights based on training stage.

    Updates reward term weights dynamically during training.

    Args:
        env: Environment instance
        env_ids: Environment IDs (unused)
        reward_stages: List of stage dicts with 'step' and reward_name: weight pairs

    Returns:
        Dictionary of current reward weights for logging
    """
    del env_ids  # Unused

    # Update reward weights based on current step
    for stage in reward_stages:
        if env.common_step_counter > stage["step"]:
            for reward_name, weight in stage.items():
                if reward_name != "step" and reward_name in env.reward_manager._terms:
                    env.reward_manager._terms[reward_name].weight = weight

    # Return current weights for logging (jump-related only)
    metrics = {}
    for name, term in env.reward_manager._terms.items():
        if name.startswith(("jump", "vertical", "air_time_both", "landing", "leg_coordination")):
            metrics[f"weight_{name}"] = torch.tensor(term.weight)

    return metrics


# -------------------------------------------------------------------------
# Velocity Stages (for locomotion curriculum)
# -------------------------------------------------------------------------

# Standard velocity curriculum: progressively increase running speed
# Linearly interpolates from near-zero to the final target values.
_N_STAGES = 6
_LAST_STEP = 20000 * 24
_INITIAL = {"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)}
_FINAL = {"lin_vel_x": (-0.8, 0.8), "lin_vel_y": (-2.0, 1.0), "ang_vel_z": (-1.0, 1.0)}

VELOCITY_STAGES_STANDARD = []
for _i in range(_N_STAGES):
    _t = _i / (_N_STAGES - 1)
    _stage = {"step": round(_LAST_STEP * _t)}
    for _key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        _lo = _INITIAL[_key][0] + _t * (_FINAL[_key][0] - _INITIAL[_key][0])
        _hi = _INITIAL[_key][1] + _t * (_FINAL[_key][1] - _INITIAL[_key][1])
        _stage[_key] = (round(_lo, 4), round(_hi, 4))
    VELOCITY_STAGES_STANDARD.append(_stage)

# -------------------------------------------------------------------------
# Jump Stages (for jump command curriculum)
# -------------------------------------------------------------------------

# Standard jump command stages for locomotion + jumping
# Stage 0 is constant (no jumping), then linearly interpolates from start_step to last_step.
_JUMP_N_STAGES = 3
_JUMP_START_STEP = 15000 * 24
_JUMP_LAST_STEP = 25000 * 24
_JUMP_INITIAL = {"jump_probability": 0.0, "jump_height_range": (0.0, 0.0)}
_JUMP_FINAL = {"jump_probability": 0.3, "jump_height_range": (0.5, 1.0)}

JUMP_STAGES_STANDARD = [
    # No jumping during locomotion training
    {"step": 0, "jump_probability": 0.0, "jump_height_range": (0.0, 0.0)},
]
for _i in range(_JUMP_N_STAGES):
    _t = _i / (_JUMP_N_STAGES - 1)
    _stage = {"step": round(_JUMP_START_STEP + _t * (_JUMP_LAST_STEP - _JUMP_START_STEP))}
    _stage["jump_probability"] = round(
        _JUMP_INITIAL["jump_probability"] + _t * (_JUMP_FINAL["jump_probability"] - _JUMP_INITIAL["jump_probability"]), 4
    )
    _lo = _JUMP_INITIAL["jump_height_range"][0] + _t * (_JUMP_FINAL["jump_height_range"][0] - _JUMP_INITIAL["jump_height_range"][0])
    _hi = _JUMP_INITIAL["jump_height_range"][1] + _t * (_JUMP_FINAL["jump_height_range"][1] - _JUMP_INITIAL["jump_height_range"][1])
    _stage["jump_height_range"] = (round(_lo, 4), round(_hi, 4))
    JUMP_STAGES_STANDARD.append(_stage)

# -------------------------------------------------------------------------
# Reward Stages (for reward weight curriculum)
# -------------------------------------------------------------------------

# Standard reward weight stages for jump training
# Progressively enables and increases jump rewards
REWARD_STAGES_STANDARD = [
    # Stage 0-3: Locomotion only (0-15K iterations)
    {
        "step": 0,
        # Jump rewards disabled
        "vertical_velocity": 0.0,
        "joint_extension": 0.0,
        "leg_coordination": 0.0,
        "air_time_both": 0.0,
        "jump_height": 0.0,
        "landing_stability": 0.0,
        "soft_landing": 0.0,
        # Locomotion rewards at default
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
]


__all__ = [
    # Curriculum functions
    "jump_command_curriculum",
    "reward_weight_curriculum",
    # Stage definitions (only what's actually used)
    "VELOCITY_STAGES_STANDARD",
    "JUMP_STAGES_STANDARD",
    "REWARD_STAGES_STANDARD",
]
