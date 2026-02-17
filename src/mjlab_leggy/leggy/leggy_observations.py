"""Standard observation configurations for Leggy robot.

Defines reusable observation terms that are common across all Leggy tasks.
"""

from copy import deepcopy

import torch

from mjlab.managers.observation_manager import ObservationTermCfg

from .leggy_actions import joint_pos_motor, joint_vel_motor, joint_torques_motor, body_euler


def base_lin_vel(env, asset_cfg=None) -> torch.Tensor:
    """Get base linear velocity in body frame."""
    asset = env.scene["robot"]
    return asset.data.root_link_lin_vel_b


def base_ang_vel(env, asset_cfg=None) -> torch.Tensor:
    """Get base angular velocity in body frame."""
    asset = env.scene["robot"]
    return asset.data.root_link_ang_vel_b


def velocity_commands(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Get velocity commands."""
    return env.command_manager.get_command(command_name)


def last_action(env, asset_cfg=None) -> torch.Tensor:
    """Get last action."""
    return env.action_manager.action


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

    # Joint positions in motor space (motor = knee + hipX)
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
    # Framework observations
    # -------------------------------------------------------------------------
    cfg.observations["policy"].terms["base_lin_vel"] = ObservationTermCfg(
        func=base_lin_vel
    )
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=base_lin_vel
    )

    cfg.observations["policy"].terms["base_ang_vel"] = ObservationTermCfg(
        func=base_ang_vel
    )
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
        func=base_ang_vel
    )

    cfg.observations["policy"].terms["command"] = ObservationTermCfg(
        func=velocity_commands,
        params={"command_name": "twist"},
    )
    cfg.observations["critic"].terms["command"] = ObservationTermCfg(
        func=velocity_commands,
        params={"command_name": "twist"},
    )

    cfg.observations["policy"].terms["actions"] = ObservationTermCfg(
        func=last_action
    )
    cfg.observations["critic"].terms["actions"] = ObservationTermCfg(
        func=last_action
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
    "base_lin_vel",
    "base_ang_vel",
    "velocity_commands",
    "last_action",
    "configure_leggy_observations",
]
