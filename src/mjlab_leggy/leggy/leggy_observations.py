"""Standard observation configurations for Leggy robot."""

from copy import deepcopy

import torch

from mjlab.managers.observation_manager import ObservationTermCfg

from .leggy_actions import joint_pos_motor, joint_vel_motor, joint_torques_motor, body_euler


def base_lin_vel(env, asset_cfg=None) -> torch.Tensor:
    """Base linear velocity in body frame."""
    return env.scene["robot"].data.root_link_lin_vel_b


def base_ang_vel(env, asset_cfg=None) -> torch.Tensor:
    """Base angular velocity in body frame."""
    return env.scene["robot"].data.root_link_ang_vel_b


def velocity_commands(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Velocity commands."""
    return env.command_manager.get_command(command_name)


def last_action(env, asset_cfg=None) -> torch.Tensor:
    """Last action."""
    return env.action_manager.action


def configure_leggy_observations(cfg, enable_corruption: bool = True):
    """Configure Leggy observations for policy and critic.

    Sets up motor-space joints, body Euler angles, motor torques,
    observation history, and sensor corruption for sim-to-real.
    """
    # Observation terms shared between policy and critic
    shared_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos_motor),
        "joint_vel": ObservationTermCfg(func=joint_vel_motor),
        "body_euler": ObservationTermCfg(func=body_euler),
        "joint_torques": ObservationTermCfg(func=joint_torques_motor),
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
        "command": ObservationTermCfg(func=velocity_commands, params={"command_name": "twist"}),
        "actions": ObservationTermCfg(func=last_action),
    }

    for group in ("policy", "critic"):
        # Remove projected_gravity (redundant with body Euler)
        cfg.observations[group].terms.pop("projected_gravity", None)
        # Set all shared terms
        for name, term_cfg in shared_terms.items():
            cfg.observations[group].terms[name] = deepcopy(term_cfg)

    # Policy-specific: history and corruption
    cfg.observations["policy"].history_length = 5
    cfg.observations["policy"].flatten_history_dim = True
    cfg.observations["policy"].enable_corruption = enable_corruption
    cfg.observations["policy"].corruption_std = 0.01

    # IMU sensor delays (realistic hardware latency)
    for term_name in ("base_ang_vel", "body_euler"):
        term = cfg.observations["policy"].terms[term_name]
        term.delay_min_lag = 2
        term.delay_max_lag = 4
        term.delay_update_period = 64
