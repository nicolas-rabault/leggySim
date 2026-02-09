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


def vertical_velocity_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "jump_command"
) -> torch.Tensor:
    """Reward upward velocity when jump command is active.

    Encourages explosive extension to achieve liftoff.
    Uses exponential scaling to strongly reward higher velocities.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        command_name: Name of the command manager with jump signal.

    Returns:
        Reward based on vertical velocity (0 if not jumping).
    """
    asset = env.scene[asset_cfg.name]
    vertical_vel = asset.data.root_lin_vel_w[:, 2]  # Z component (up is positive)

    # Get jump command (expected to be 0 or >0 for jump height target)
    jump_cmd = env.command_manager.get_command(command_name)[:, 0]  # First element is jump height

    # Only reward during jump command (jump_cmd > 0)
    is_jumping = (jump_cmd > 0.01).float()

    # Exponential reward for upward velocity: exp(5 * vel_z)
    # vel_z = 0.0 -> reward = 1.0
    # vel_z = 0.2 -> reward = 2.72
    # vel_z = 0.5 -> reward = 12.18
    reward = torch.exp(5.0 * vertical_vel.clamp(min=0.0, max=1.0)) * is_jumping

    return reward


def joint_extension_speed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "jump_command"
) -> torch.Tensor:
    """Reward rapid joint extension during jump.

    Encourages coordinated explosive movement of knees and hips.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        command_name: Name of the command manager with jump signal.

    Returns:
        Reward based on extension velocity.
    """
    asset = env.scene[asset_cfg.name]

    # Get knee and hip joint velocities
    joint_names = ["LhipX", "Lknee", "RhipX", "Rknee"]
    joint_ids = asset.find_joints(joint_names)[0]
    joint_vel = asset.data.joint_vel[:, joint_ids]

    # Get jump command
    jump_cmd = env.command_manager.get_command(command_name)[:, 0]
    is_jumping = (jump_cmd > 0.01).float()

    # Extension is positive velocity (opening joints)
    # Sum positive velocities for all extension joints
    extension_vel = torch.sum(torch.clamp(joint_vel, min=0.0), dim=1)

    return extension_vel * is_jumping


def leg_coordination(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward symmetric leg movement.

    Penalizes asymmetric extension which would cause rotation.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.

    Returns:
        Penalty for asymmetric leg movement.
    """
    asset = env.scene[asset_cfg.name]

    # Get joint velocities
    joint_names = ["LhipX", "Lknee", "RhipX", "Rknee"]
    joint_ids = asset.find_joints(joint_names)[0]
    joint_vel = asset.data.joint_vel[:, joint_ids]

    # Compute extension velocity per leg
    left_extension = joint_vel[:, 0] + joint_vel[:, 1]  # LhipX + Lknee
    right_extension = joint_vel[:, 2] + joint_vel[:, 3]  # RhipX + Rknee

    # Penalize difference (want symmetric movement)
    asymmetry = torch.abs(left_extension - right_extension)

    return -asymmetry


def air_time_both_feet(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "jump_command",
    mode: str = "jump",
    velocity_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward time with both feet off ground.

    Args:
        env: The environment.
        sensor_name: Name of the contact sensor.
        command_name: Name of the command manager.
        mode: "jump" or "velocity".
        velocity_threshold: Minimum velocity for velocity mode (m/s).
        asset_cfg: Asset configuration for accessing velocity data.

    Returns:
        Reward when both feet are airborne.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    contact = sensor.data.found.squeeze(-1).bool()
    both_feet_off = (~contact[:, 0]) & (~contact[:, 1])

    if mode == "velocity":
        asset = env.scene[asset_cfg.name]
        actual_vel = asset.data.root_lin_vel_w[:, :2]
        vel_magnitude = torch.norm(actual_vel, dim=1)
        is_active = (vel_magnitude > velocity_threshold).float()
    else:  # jump mode
        jump_cmd = env.command_manager.get_command(command_name)[:, 0]
        is_active = (jump_cmd > 0.01).float()

    return both_feet_off.float() * is_active


def jump_height_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    standing_height: float = 0.18,
    command_name: str = "jump_command"
) -> torch.Tensor:
    """Reward height above standing position.

    Encourages maximizing jump height.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        standing_height: Expected standing height in meters.
        command_name: Name of the command manager with jump signal.

    Returns:
        Reward based on height gain.
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]

    # Get jump command
    jump_cmd = env.command_manager.get_command(command_name)[:, 0]
    is_jumping = (jump_cmd > 0.01).float()

    # Reward height above standing
    height_gain = torch.clamp(current_height - standing_height, min=0.0)

    return height_gain * is_jumping * 10.0  # Scale for visibility


def landing_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
    """Reward maintaining upright orientation after landing.

    Encourages stable landing without falling.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        sensor_name: Name of the contact sensor.

    Returns:
        Reward for upright orientation when in contact.
    """
    asset = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    # Get orientation (roll, pitch)
    quat = asset.data.root_quat_w
    # Convert quaternion to euler angles (simplified for roll/pitch)
    # For small angles, can use approximation
    roll = 2.0 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3])
    pitch = 2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1])

    # Get contact state
    contact = sensor.data.found.squeeze(-1).bool()
    any_foot_contact = contact[:, 0] | contact[:, 1]

    # Reward low roll/pitch when in contact (landing phase)
    orientation_reward = torch.exp(-10.0 * (torch.abs(roll) + torch.abs(pitch)))

    return orientation_reward * any_foot_contact.float()


def soft_landing_bonus(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_force_threshold: float = 10.0
) -> torch.Tensor:
    """Reward gentle landings with low impact forces.

    Penalizes hard impacts that could damage the robot.

    Args:
        env: The environment.
        sensor_name: Name of the contact sensor.
        max_force_threshold: Maximum acceptable force (N).

    Returns:
        Penalty for hard impacts.
    """
    sensor: ContactSensor = env.scene[sensor_name]

    # Get contact forces
    forces = sensor.data.force.squeeze(2)
    normal_forces = forces[:, :, 2]

    # Get peak force
    peak_force = torch.max(torch.abs(normal_forces), dim=1)[0]

    # Penalty for excessive force
    penalty = torch.clamp(peak_force - max_force_threshold, min=0.0) / max_force_threshold

    return -penalty


def pose_running_adaptive(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    velocity_threshold: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Pose reward that relaxes at high speeds.

    Args:
        env: The environment.
        command_name: Name of the velocity command.
        velocity_threshold: Velocity above which constraints relax (m/s).
        asset_cfg: Asset configuration.

    Returns:
        Pose deviation penalty (scaled by velocity).
    """
    asset = env.scene[asset_cfg.name]
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    vel_magnitude = torch.norm(vel_cmd, dim=1)

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation_sq = torch.sum((joint_pos - joint_pos_default) ** 2, dim=1)

    # Scale penalty: full below threshold, reduced above threshold
    scale = torch.where(vel_magnitude < velocity_threshold,
                       torch.ones_like(vel_magnitude),
                       torch.ones_like(vel_magnitude) * 0.3)

    return torch.exp(-deviation_sq * scale)


def action_rate_running_adaptive(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    velocity_threshold: float = 1.0
) -> torch.Tensor:
    """Action rate penalty that reduces at high speeds.

    Args:
        env: The environment.
        command_name: Name of the velocity command.
        velocity_threshold: Velocity above which penalty reduces to 30%.

    Returns:
        Action rate penalty scaled by velocity.
    """
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    vel_magnitude = torch.norm(vel_cmd, dim=1)

    action_rate = torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)

    scale = torch.where(vel_magnitude < velocity_threshold,
                       torch.ones_like(vel_magnitude),
                       torch.ones_like(vel_magnitude) * 0.3)

    return action_rate * scale


__all__ = [
    "joint_pos_limits_motor",
    "leg_collision_penalty",
    "vertical_velocity_reward",
    "joint_extension_speed",
    "leg_coordination",
    "air_time_both_feet",
    "jump_height_reward",
    "landing_stability",
    "soft_landing_bonus",
    "pose_running_adaptive",
    "action_rate_running_adaptive",
]
