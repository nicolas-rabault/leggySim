"""Custom reward functions for Leggy robot.

This module provides robot-specific reward functions that account for Leggy's
motor-to-knee conversion mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from .leggy_actions import knee_to_motor


def joint_pos_limits_motor(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize motor positions if they cross soft limits.

    For Leggy's knee joints, checks limits in motor space (motor = knee - hipX)
    rather than knee space, since the physical motor limits are in motor space.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()

    # Get joint positions for conversion
    joint_names = ["LhipX", "Lknee", "RhipX", "Rknee"]
    joint_ids = asset.find_joints(joint_names)[0]
    lhipx_pos = asset.data.joint_pos[:, joint_ids[0]]
    lknee_pos = asset.data.joint_pos[:, joint_ids[1]]
    rhipx_pos = asset.data.joint_pos[:, joint_ids[2]]
    rknee_pos = asset.data.joint_pos[:, joint_ids[3]]

    # Convert knee to motor space and update in the joint_pos tensor
    # Joint order in asset_cfg.joint_ids is: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
    joint_pos[:, 2] = knee_to_motor(lknee_pos, lhipx_pos)
    joint_pos[:, 5] = knee_to_motor(rknee_pos, rhipx_pos)

    # Check limits
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_limits = -(joint_pos - limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[:, :, 1]).clip(min=0.0)

    return torch.sum(out_of_limits, dim=1)


def leg_collision_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize leg-leg collisions.

    Provides continuous penalty signal when legs collide with each other.
    This acts as a soft constraint before the hard termination kicks in.

    Args:
        env: The environment.
        sensor_name: Name of the contact sensor to check.

    Returns:
        Penalty value (1.0 if collision detected, 0.0 otherwise).
    """
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    # Return 1.0 for environments with collision, 0.0 otherwise
    # This will be multiplied by negative weight to create penalty
    collision_detected = torch.any(sensor.data.found, dim=-1).float()

    return collision_detected


__all__ = [
    "joint_pos_limits_motor",
    "leg_collision_penalty",
]
