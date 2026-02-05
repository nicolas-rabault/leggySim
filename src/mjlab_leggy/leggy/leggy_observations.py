"""Standard observation configurations for Leggy robot.

Defines reusable observation terms that are common across all Leggy tasks.
"""

from copy import deepcopy

from mjlab.managers.observation_manager import ObservationTermCfg

from .leggy_actions import joint_pos_motor, joint_vel_motor, joint_torques_motor, body_euler


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
    "configure_leggy_observations",
]
