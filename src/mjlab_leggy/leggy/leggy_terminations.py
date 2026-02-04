"""Custom termination functions for Leggy robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def illegal_contact_debug(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Terminate on illegal contact."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    result = torch.any(sensor.data.found, dim=-1)

    return result


def illegal_contact_curriculum(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    enable_after_iterations: int = 1000,
    mean_episode_length_threshold: float = 100.0
) -> torch.Tensor:
    """Terminate on illegal contact with curriculum learning.

    Only activates after:
    1. Training has progressed past enable_after_iterations, AND
    2. Mean episode length exceeds mean_episode_length_threshold

    This prevents the agent from exploiting the termination by falling slowly
    before it learns to stand properly.

    Args:
        env: The environment.
        sensor_name: Name of the contact sensor to check.
        enable_after_iterations: Minimum training iterations before enabling (default: 1000).
        mean_episode_length_threshold: Minimum mean episode length before enabling (default: 100).
    """
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    # Check if we should enable the termination based on curriculum
    current_iteration = getattr(env, "common_step_counter", 0)

    # Get mean episode length from extras if available
    mean_ep_length = getattr(env.episode_length_buf, "float", lambda: torch.tensor(0.0))().mean().item()

    # Only activate termination if both conditions are met
    curriculum_active = (
        current_iteration >= enable_after_iterations and
        mean_ep_length >= mean_episode_length_threshold
    )

    if not curriculum_active:
        # Return all False (no terminations) during curriculum phase
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Normal collision detection after curriculum phase
    result = torch.any(sensor.data.found, dim=-1)

    return result


__all__ = ["illegal_contact_debug", "illegal_contact_curriculum"]
