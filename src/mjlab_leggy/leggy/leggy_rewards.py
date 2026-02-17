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

    For Leggy's knee joints, checks limits in motor space (motor = knee + hipX)
    rather than knee space, since the physical motor limits are in motor space.
    Uses passiveMotor joint limits for the motor-space check.
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

    # Get limits: use actuated joint limits for hips, but passiveMotor limits for knees
    # (since we converted knee values to motor space, we must check against motor limits)
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids].clone()
    motor_joint_names = ["LpassiveMotor", "RpassiveMotor"]
    motor_joint_ids = asset.find_joints(motor_joint_names)[0]
    motor_limits = asset.data.soft_joint_pos_limits[:, motor_joint_ids]
    limits[:, 2, :] = motor_limits[:, 0, :]  # Lmotor limits
    limits[:, 5, :] = motor_limits[:, 1, :]  # Rmotor limits

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


def mechanical_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical power (torque * velocity) across all actuated joints.

    Discourages high-energy gaits like hopping in favor of efficient alternating
    gaits. Mechanical power measures the actual work done by the motors.

    Args:
        env: The environment.
        asset_cfg: Asset configuration.

    Returns:
        Total absolute mechanical power across all joints.
    """
    asset = env.scene[asset_cfg.name]

    actuator_names = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
    actuator_ids = asset.find_actuators(actuator_names)[0]
    joint_names = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
    joint_ids = asset.find_joints(joint_names)[0]

    torques = asset.data.actuator_force[:, actuator_ids]
    velocities = asset.data.joint_vel[:, joint_ids]

    # Mechanical power = |torque * velocity| per joint, summed
    power = torch.sum(torch.abs(torques * velocities), dim=1)

    return power


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
    "mechanical_power",
    "action_rate_running_adaptive",
]
