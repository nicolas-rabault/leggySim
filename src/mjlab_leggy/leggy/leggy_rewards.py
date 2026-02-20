"""Custom reward functions for Leggy robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from .leggy_actions import knee_to_motor

_JOINT_NAMES = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]


def joint_pos_limits_motor(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize motor positions crossing soft limits.

    Checks knee joints in motor space (motor = knee + hipX) since
    physical motor limits are in motor space.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()

    joint_ids = asset.find_joints(["LhipX", "Lknee", "RhipX", "Rknee"])[0]
    lhipx = asset.data.joint_pos[:, joint_ids[0]]
    lknee = asset.data.joint_pos[:, joint_ids[1]]
    rhipx = asset.data.joint_pos[:, joint_ids[2]]
    rknee = asset.data.joint_pos[:, joint_ids[3]]

    # Convert knee to motor space
    joint_pos[:, 2] = knee_to_motor(lknee, lhipx)
    joint_pos[:, 5] = knee_to_motor(rknee, rhipx)

    # Use passiveMotor limits for knee channels (now in motor space)
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids].clone()
    motor_joint_ids = asset.find_joints(["LpassiveMotor", "RpassiveMotor"])[0]
    motor_limits = asset.data.soft_joint_pos_limits[:, motor_joint_ids]
    limits[:, 2, :] = motor_limits[:, 0, :]
    limits[:, 5, :] = motor_limits[:, 1, :]

    out_of_limits = -(joint_pos - limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def leg_collision_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize leg-leg collisions (1.0 if collision, 0.0 otherwise)."""
    sensor: ContactSensor = env.scene[sensor_name]
    return torch.any(sensor.data.found, dim=-1).float()


def mechanical_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical power (|torque * velocity|) across actuated joints."""
    asset = env.scene[asset_cfg.name]
    actuator_ids = asset.find_actuators(_JOINT_NAMES)[0]
    joint_ids = asset.find_joints(_JOINT_NAMES)[0]
    torques = asset.data.actuator_force[:, actuator_ids]
    velocities = asset.data.joint_vel[:, joint_ids]
    return torch.sum(torch.abs(torques * velocities), dim=1)


def action_rate_running_adaptive(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    velocity_threshold: float = 1.0,
) -> torch.Tensor:
    """Action rate penalty that reduces to 30% above velocity_threshold."""
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    vel_magnitude = torch.norm(vel_cmd, dim=1)
    action_rate = torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)
    scale = torch.where(vel_magnitude < velocity_threshold,
                        torch.ones_like(vel_magnitude),
                        torch.ones_like(vel_magnitude) * 0.3)
    return action_rate * scale


class same_foot_penalty:
    """Growing penalty for consecutive same-foot contacts.

    Counts how many times the robot enters single-support on the same foot
    without alternating. Penalty equals the count:
    - Hopping on one leg: count grows 1, 2, 3... -> growing penalty
    - Proper walking: count stays at 1 -> minimal penalty
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.last_support = torch.full((env.num_envs,), -1.0, device=env.device)
        self.was_single_support = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.same_foot_count = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self.last_support[env_ids] = -1.0
        self.was_single_support[env_ids] = False
        self.same_foot_count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "feet_ground_contact",
        command_name: str = "twist",
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        contact = sensor.data.found.squeeze(-1) > 0  # [B, 2]

        left = contact[:, 0]
        right = contact[:, 1]
        is_single_support = left ^ right

        entered_single = is_single_support & ~self.was_single_support
        self.was_single_support = is_single_support

        current_support = torch.where(
            left & ~right,
            torch.zeros_like(self.last_support),
            torch.where(
                ~left & right,
                torch.ones_like(self.last_support),
                torch.full_like(self.last_support, -1.0),
            ),
        )

        same_foot = entered_single & (current_support == self.last_support)
        diff_foot = entered_single & (self.last_support >= 0) & (current_support >= 0) & (current_support != self.last_support)

        self.same_foot_count = torch.where(same_foot, self.same_foot_count + 1.0, self.same_foot_count)
        self.same_foot_count = torch.where(diff_foot, torch.ones_like(self.same_foot_count), self.same_foot_count)

        self.last_support = torch.where(
            entered_single & (current_support >= 0), current_support, self.last_support
        )

        vel_cmd = env.command_manager.get_command(command_name)[:, :2]
        is_moving = (torch.norm(vel_cmd, dim=1) > command_threshold).float()
        return self.same_foot_count * is_moving


def flight_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    run_threshold: float = 0.8,
) -> torch.Tensor:
    """Penalize both feet in the air, scaled down at high speed.

    Full penalty at 0 m/s, zero penalty above run_threshold.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    contact = sensor.data.found.squeeze(-1) > 0
    both_in_air = (~contact[:, 0] & ~contact[:, 1]).float()

    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    scale = torch.clamp(1.0 - torch.norm(vel_cmd, dim=1) / run_threshold, min=0.0)
    return both_in_air * scale


def gait_symmetry(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize asymmetric contact durations between left and right feet."""
    sensor: ContactSensor = env.scene[sensor_name]
    last_contact = sensor.data.last_contact_time  # [B, 2]
    asymmetry = torch.abs(last_contact[:, 0] - last_contact[:, 1])

    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    is_moving = (torch.norm(vel_cmd, dim=1) > command_threshold).float()
    return asymmetry * is_moving


class forward_symmetry:
    """Penalize one foot being consistently ahead of the other.

    Uses EMA of forward foot offset — transient differences average out,
    only sustained bias triggers penalty.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset = env.scene["robot"]
        self.foot_site_ids = asset.find_sites(("left_foot", "right_foot"))[0]
        self.mean_diff = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self.mean_diff[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        command_threshold: float = 0.1,
        alpha: float = 0.01,
    ) -> torch.Tensor:
        asset = env.scene["robot"]
        foot_pos = asset.data.site_pos_w[:, self.foot_site_ids]

        _, _, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
        forward_x = torch.cos(yaw)
        forward_y = torch.sin(yaw)

        diff = foot_pos[:, 0, :2] - foot_pos[:, 1, :2]
        forward_diff = diff[:, 0] * forward_x + diff[:, 1] * forward_y

        self.mean_diff = alpha * forward_diff + (1.0 - alpha) * self.mean_diff

        vel_cmd = env.command_manager.get_command(command_name)[:, :2]
        is_moving = (torch.norm(vel_cmd, dim=1) > command_threshold).float()
        return torch.abs(self.mean_diff) * is_moving
