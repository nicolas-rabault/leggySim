"""Standard observation configurations for Leggy robot.

Defines reusable observation terms that are common across all Leggy tasks.
"""

from copy import deepcopy

import torch

from mjlab.managers.observation_manager import ObservationTermCfg

from .leggy_actions import joint_pos_motor, joint_vel_motor, joint_torques_motor, body_euler
from .leggy_actions import get_mirror_flag, MIRROR_SWAP


def base_lin_vel_mirror(env, asset_cfg=None) -> torch.Tensor:
    """Get base linear velocity with L/R mirror applied.

    For mirrored environments, negates the lateral (Y) velocity component
    so the policy sees a mirror-consistent view.
    """
    asset = env.scene["robot"]
    vel = asset.data.root_link_lin_vel_b.clone()  # [num_envs, 3] in body frame

    # Mirror: negate vy (lateral velocity)
    mirror = get_mirror_flag(env)
    if mirror.any():
        vel[mirror, 1] = -vel[mirror, 1]

    return vel


def base_ang_vel_mirror(env, asset_cfg=None) -> torch.Tensor:
    """Get base angular velocity with L/R mirror applied.

    For mirrored environments, negates roll rate (wx) and yaw rate (wz).
    Pitch rate (wy) is unchanged under L/R reflection.
    """
    asset = env.scene["robot"]
    ang_vel = asset.data.root_link_ang_vel_b.clone()  # [num_envs, 3] in body frame

    # Mirror: negate roll rate and yaw rate
    mirror = get_mirror_flag(env)
    if mirror.any():
        ang_vel[mirror, 0] = -ang_vel[mirror, 0]  # roll rate
        ang_vel[mirror, 2] = -ang_vel[mirror, 2]  # yaw rate

    return ang_vel


def velocity_commands_mirror(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Get velocity commands with L/R mirror applied.

    For mirrored environments, negates lateral velocity (lin_vel_y) and
    yaw rate (ang_vel_z) commands so the policy sees mirrored targets.
    """
    command = env.command_manager.get_command(command_name).clone()  # [num_envs, 3]: [lin_x, lin_y, ang_z]

    # Mirror: negate lin_vel_y (index 1) and ang_vel_z (index 2)
    mirror = get_mirror_flag(env)
    if mirror.any():
        command[mirror, 1] = -command[mirror, 1]  # lin_vel_y
        command[mirror, 2] = -command[mirror, 2]  # ang_vel_z

    return command


def last_action_mirror(env, asset_cfg=None) -> torch.Tensor:
    """Get last action with L/R mirror re-applied.

    The action class un-mirrors actions before applying to the robot.
    This observation re-mirrors the stored action so the policy sees
    its own output consistently.
    """
    action = env.action_manager.action.clone()  # [num_envs, 6] in physical joint space

    # Re-mirror: swap L/R back to policy's mirrored space
    mirror = get_mirror_flag(env)
    if mirror.any():
        action[mirror] = action[mirror][:, MIRROR_SWAP]

    return action


def configure_leggy_observations(cfg, enable_corruption: bool = True):
    """Configure standard Leggy observations for both policy and critic.

    Replaces default observations with Leggy-specific ones:
    - Motor space joint positions/velocities (not knee space)
    - Body Euler angles (instead of projected gravity)
    - Motor torques
    - Observation history and corruption for sim-to-real

    Args:
        cfg: Environment configuration
        enable_corruption: Whether to enable sensor noise and delays

    Modifies:
        cfg.observations["policy"] - Policy observations
        cfg.observations["critic"] - Critic observations
    """
    # -------------------------------------------------------------------------
    # Replace with motor space observations
    # -------------------------------------------------------------------------

    # Joint positions in motor space (motor = knee - hipX)
    cfg.observations["policy"].terms["joint_pos"] = ObservationTermCfg(
        func=joint_pos_motor
    )
    cfg.observations["critic"].terms["joint_pos"] = ObservationTermCfg(
        func=joint_pos_motor
    )

    # Joint velocities in motor space
    cfg.observations["policy"].terms["joint_vel"] = ObservationTermCfg(
        func=joint_vel_motor
    )
    cfg.observations["critic"].terms["joint_vel"] = ObservationTermCfg(
        func=joint_vel_motor
    )

    # Remove projected_gravity (redundant with body orientation)
    if "projected_gravity" in cfg.observations["policy"].terms:
        del cfg.observations["policy"].terms["projected_gravity"]
    if "projected_gravity" in cfg.observations["critic"].terms:
        del cfg.observations["critic"].terms["projected_gravity"]

    # Add body orientation as Euler angles (available from real IMU)
    cfg.observations["policy"].terms["body_euler"] = ObservationTermCfg(
        func=body_euler
    )
    cfg.observations["critic"].terms["body_euler"] = ObservationTermCfg(
        func=body_euler
    )

    # Add motor torques (measured at motor outputs)
    cfg.observations["policy"].terms["joint_torques"] = ObservationTermCfg(
        func=joint_torques_motor
    )
    cfg.observations["critic"].terms["joint_torques"] = ObservationTermCfg(
        func=joint_torques_motor
    )

    # -------------------------------------------------------------------------
    # Mirror-aware framework observations
    # -------------------------------------------------------------------------
    # Replace framework observations with mirror-aware versions that swap
    # L/R for randomly mirrored environments (enforces gait symmetry)
    cfg.observations["policy"].terms["base_lin_vel"] = ObservationTermCfg(
        func=base_lin_vel_mirror
    )
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=base_lin_vel_mirror
    )

    cfg.observations["policy"].terms["base_ang_vel"] = ObservationTermCfg(
        func=base_ang_vel_mirror
    )
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
        func=base_ang_vel_mirror
    )

    cfg.observations["policy"].terms["command"] = ObservationTermCfg(
        func=velocity_commands_mirror,
        params={"command_name": "twist"},
    )
    cfg.observations["critic"].terms["command"] = ObservationTermCfg(
        func=velocity_commands_mirror,
        params={"command_name": "twist"},
    )

    cfg.observations["policy"].terms["actions"] = ObservationTermCfg(
        func=last_action_mirror
    )
    cfg.observations["critic"].terms["actions"] = ObservationTermCfg(
        func=last_action_mirror
    )

    # -------------------------------------------------------------------------
    # Observation history for temporal context
    # -------------------------------------------------------------------------
    cfg.observations["policy"].history_length = 5
    cfg.observations["policy"].flatten_history_dim = True

    # -------------------------------------------------------------------------
    # Sensor corruption for sim-to-real transfer
    # -------------------------------------------------------------------------
    cfg.observations["policy"].enable_corruption = enable_corruption
    cfg.observations["policy"].corruption_std = 0.01

    # Configure IMU sensor delays (realistic hardware latency)
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )
    cfg.observations["policy"].terms["body_euler"] = deepcopy(
        cfg.observations["policy"].terms["body_euler"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 2
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 4
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["policy"].terms["body_euler"].delay_min_lag = 2
    cfg.observations["policy"].terms["body_euler"].delay_max_lag = 4
    cfg.observations["policy"].terms["body_euler"].delay_update_period = 64


__all__ = [
    "base_lin_vel_mirror",
    "base_ang_vel_mirror",
    "velocity_commands_mirror",
    "last_action_mirror",
    "configure_leggy_observations",
]
