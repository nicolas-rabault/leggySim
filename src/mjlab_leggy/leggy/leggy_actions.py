"""Custom action term for Leggy with motor-to-knee conversion.

Policy outputs motor commands that get converted to knee angles.
Passive joints are handled by MuJoCo's constraint solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat


def motor_to_knee(motor: torch.Tensor | float, hipX: torch.Tensor | float) -> torch.Tensor | float:
    """Convert motor command to knee angle (knee = motor - hipX)."""
    return motor - hipX


def knee_to_motor(knee: torch.Tensor | float, hipX: torch.Tensor | float) -> torch.Tensor | float:
    """Convert knee angle to motor command (motor = knee + hipX)."""
    return knee + hipX


class LeggyJointAction(JointPositionAction):
    """Action term that converts motor commands to knee angles.

    Policy outputs: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
    Converted to:   [LhipY, LhipX, Lknee,  RhipY, RhipX, Rknee]
    """

    cfg: LeggyJointActionCfg
    _motor_limits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._motor_limits = None

        # Fix offset for motor channels (indices 2 and 5):
        # Parent sets offset to default KNEE positions, but we treat these as
        # MOTOR commands. Convert to motor-space defaults.
        self._offset[:, 2] = knee_to_motor(self._offset[:, 2], self._offset[:, 1])
        self._offset[:, 5] = knee_to_motor(self._offset[:, 5], self._offset[:, 4])

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset

        if self._motor_limits is None:
            asset = self._env.scene[self.cfg.asset_name]
            motor_joint_ids = asset.find_joints(["LpassiveMotor", "RpassiveMotor"])[0]
            motor_soft_limits = asset.data.soft_joint_pos_limits[:, motor_joint_ids]
            self._motor_limits = torch.stack([
                motor_soft_limits[0, 0, 0],  # Lmotor min
                motor_soft_limits[0, 0, 1],  # Lmotor max
                motor_soft_limits[0, 1, 0],  # Rmotor min
                motor_soft_limits[0, 1, 1],  # Rmotor max
            ]).to(self._processed_actions.device)

        # Clamp motor commands before converting to knee space
        self._processed_actions[:, 2] = torch.clamp(
            self._processed_actions[:, 2],
            min=self._motor_limits[0], max=self._motor_limits[1],
        )
        self._processed_actions[:, 5] = torch.clamp(
            self._processed_actions[:, 5],
            min=self._motor_limits[2], max=self._motor_limits[3],
        )

        # Convert motor → knee using target hipX values
        self._processed_actions[:, 2] = motor_to_knee(self._processed_actions[:, 2], self._processed_actions[:, 1])
        self._processed_actions[:, 5] = motor_to_knee(self._processed_actions[:, 5], self._processed_actions[:, 4])


@dataclass(kw_only=True)
class LeggyJointActionCfg(JointPositionActionCfg):
    """Configuration for Leggy joint action with motor-to-knee conversion."""
    class_type: type[JointPositionAction] = LeggyJointAction
    asset_name: str = "robot"
    actuator_names: tuple[str, ...] = ("LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee")


_JOINT_NAMES = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]


def joint_pos_motor(env, asset_cfg=None) -> torch.Tensor:
    """Joint positions in motor space [num_envs, 6]."""
    asset = env.scene["robot"]
    joint_ids = asset.find_joints(_JOINT_NAMES)[0]
    pos = asset.data.joint_pos[:, joint_ids].clone()
    pos[:, 2] = knee_to_motor(pos[:, 2], pos[:, 1])
    pos[:, 5] = knee_to_motor(pos[:, 5], pos[:, 4])
    return pos


def joint_vel_motor(env, asset_cfg=None) -> torch.Tensor:
    """Joint velocities in motor space [num_envs, 6]."""
    asset = env.scene["robot"]
    joint_ids = asset.find_joints(_JOINT_NAMES)[0]
    vel = asset.data.joint_vel[:, joint_ids].clone()
    # Motor velocity = d/dt(knee + hipX) = knee_vel + hipX_vel
    vel[:, 2] = vel[:, 2] + vel[:, 1]
    vel[:, 5] = vel[:, 5] + vel[:, 4]
    return vel


def joint_torques_motor(env, asset_cfg=None) -> torch.Tensor:
    """Actuator torques in motor-space layout [num_envs, 6]."""
    asset = env.scene["robot"]
    actuator_ids = asset.find_actuators(_JOINT_NAMES)[0]
    return asset.data.actuator_force[:, actuator_ids]


def body_euler(env, asset_cfg=None) -> torch.Tensor:
    """Body orientation as Euler angles (roll, pitch, yaw) [num_envs, 3]."""
    asset = env.scene["robot"]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
    return torch.stack([roll, pitch, yaw], dim=1)
