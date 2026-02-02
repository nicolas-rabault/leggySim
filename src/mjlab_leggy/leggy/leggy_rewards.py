"""Custom reward functions for Leggy robot.

This module provides robot-specific reward functions that account for Leggy's
motor-to-knee conversion mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

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


__all__ = [
    "joint_pos_limits_motor",
]
