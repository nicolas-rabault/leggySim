"""Curriculum callback for jump training.

Dynamically adjusts reward weights based on training phase.
This implements the 6-phase curriculum for learning to jump.

Usage:
    Import and attach to the RL runner as a callback.
"""

from typing import Any

import torch


class JumpCurriculumCallback:
    """Curriculum manager for jump training.

    Adjusts reward weights based on current training iteration.

    Phases:
        1. Crouch (0-2000): Learn to lower center of mass
        2. Explode (2000-5000): Rapid extension movements
        3. Liftoff (5000-8000): Break contact with ground
        4. Landing (8000-12000): Safe landing control
        5. Height Optimization (12000-20000): Maximize jump height
        6. Commanded (20000+): Jump to specified heights
    """

    def __init__(self, env):
        """Initialize curriculum callback.

        Args:
            env: The RL environment instance.
        """
        self.env = env
        self.current_phase = "crouch"
        self.iteration = 0

        # Store original reward weights for restoration
        self.original_weights = {}
        for name, term in env.reward_manager._terms.items():
            self.original_weights[name] = term.weight

    def __call__(self, runner: Any) -> None:
        """Update reward weights based on current iteration.

        Called by the RL runner at each iteration.

        Args:
            runner: The RL runner instance (provides iteration count).
        """
        self.iteration = runner.current_iteration
        phase = self._get_phase(self.iteration)

        # Only update if phase changed
        if phase != self.current_phase:
            print(f"\n{'='*60}")
            print(f"CURRICULUM: Transitioning to phase '{phase}' at iteration {self.iteration}")
            print(f"{'='*60}\n")
            self.current_phase = phase

        # Apply phase-specific weights
        self._apply_phase_weights(phase)

        # Apply phase-specific randomization
        self._apply_phase_randomization(phase)

    def _get_phase(self, iteration: int) -> str:
        """Determine current training phase.

        Args:
            iteration: Current training iteration.

        Returns:
            Phase name.
        """
        if iteration < 2000:
            return "crouch"
        elif iteration < 5000:
            return "explode"
        elif iteration < 8000:
            return "liftoff"
        elif iteration < 12000:
            return "landing"
        elif iteration < 20000:
            return "height_opt"
        else:
            return "commanded"

    def _apply_phase_weights(self, phase: str) -> None:
        """Apply reward weights for current phase.

        Args:
            phase: Current training phase.
        """
        reward_mgr = self.env.reward_manager

        # Define phase-specific weight schedules
        if phase == "crouch":
            # Phase 1: Learn to crouch
            weights = {
                "crouch_depth": 3.0,
                "upright": 1.0,
                "pose": 5.0,
                "action_rate_l2": -0.5,
                # All jump rewards disabled
                "vertical_velocity": 0.0,
                "joint_extension": 0.0,
                "leg_coordination": 0.0,
                "air_time_both": 0.0,
                "jump_height": 0.0,
                "landing_stability": 0.0,
                "soft_landing": 0.0,
            }

        elif phase == "explode":
            # Phase 2: Learn explosive extension
            weights = {
                "crouch_depth": 1.0,  # Reduce but keep
                "vertical_velocity": 5.0,  # Main focus
                "joint_extension": 2.0,
                "leg_coordination": 3.0,  # Important for symmetry
                "upright": 0.5,  # Relax during explosion
                "pose": 2.0,  # Relax to allow movement
                "action_rate_l2": 0.0,  # Disable smoothness
                # Liftoff rewards not yet
                "air_time_both": 0.0,
                "jump_height": 0.0,
                "landing_stability": 0.0,
                "soft_landing": 0.0,
            }

        elif phase == "liftoff":
            # Phase 3: Achieve liftoff
            weights = {
                "crouch_depth": 0.5,
                "vertical_velocity": 3.0,
                "joint_extension": 1.0,
                "leg_coordination": 2.0,
                "air_time_both": 10.0,  # Main focus
                "jump_height": 8.0,  # Encourage height
                "upright": 1.0,  # Maintain orientation
                "pose": 2.0,
                "action_rate_l2": 0.0,
                # Landing not yet critical
                "landing_stability": 0.0,
                "soft_landing": 0.0,
            }

        elif phase == "landing":
            # Phase 4: Safe landing
            weights = {
                "crouch_depth": 0.0,
                "vertical_velocity": 2.0,
                "joint_extension": 0.5,
                "leg_coordination": 2.0,
                "air_time_both": 5.0,
                "jump_height": 5.0,
                "landing_stability": 5.0,  # Main focus
                "soft_landing": 4.0,  # Avoid hard impacts
                "upright": 2.0,  # Important for landing
                "pose": 3.0,  # Return to good pose
                "action_rate_l2": -1.0,  # Some smoothness
            }

        elif phase == "height_opt":
            # Phase 5: Maximize height
            weights = {
                "crouch_depth": 0.0,
                "vertical_velocity": 2.0,
                "joint_extension": 1.0,
                "leg_coordination": 2.0,
                "air_time_both": 3.0,
                "jump_height": 15.0,  # Main focus: maximize height
                "landing_stability": 5.0,  # Must maintain landing success
                "soft_landing": 3.0,
                "upright": 1.5,
                "pose": 2.5,
                "action_rate_l2": -1.0,
            }

        else:  # commanded
            # Phase 6: Track commanded heights
            weights = {
                "crouch_depth": 0.0,
                "vertical_velocity": 1.5,
                "joint_extension": 0.5,
                "leg_coordination": 2.0,
                "air_time_both": 2.0,
                "jump_height": 10.0,  # Track commanded height
                "landing_stability": 5.0,
                "soft_landing": 3.0,
                "upright": 1.5,
                "pose": 2.5,
                "action_rate_l2": -1.5,
            }

        # Apply weights
        for name, weight in weights.items():
            if name in reward_mgr._terms:
                reward_mgr._terms[name].weight = weight

    def _apply_phase_randomization(self, phase: str) -> None:
        """Apply phase-specific environment randomization.

        Args:
            phase: Current training phase.
        """
        # Gradually increase randomization as training progresses
        if phase in ["crouch", "explode"]:
            # Minimal randomization during early learning
            friction_range = (1.2, 1.5)
            push_range = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
            noise_std = 0.005

        elif phase in ["liftoff", "landing"]:
            # Medium randomization
            friction_range = (1.0, 1.8)
            push_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
            noise_std = 0.01

        else:  # height_opt, commanded
            # Full randomization for robustness
            friction_range = (0.8, 2.0)
            push_range = {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}
            noise_std = 0.02

        # Apply to environment (if event managers support dynamic updates)
        # Note: This assumes the event manager allows runtime parameter updates
        # If not supported by MJLab, these will just be ignored
        try:
            if "foot_friction" in self.env.event_manager._terms:
                self.env.event_manager._terms["foot_friction"].cfg.params["ranges"] = friction_range

            if "push_robot" in self.env.event_manager._terms:
                self.env.event_manager._terms["push_robot"].cfg.params["velocity_range"] = push_range

            # Update observation corruption
            self.env.observation_manager._groups["policy"]._corruption_std = noise_std

        except (AttributeError, KeyError):
            # If dynamic updates not supported, skip
            pass


def create_jump_curriculum_callback(env):
    """Factory function to create curriculum callback.

    Args:
        env: The RL environment instance.

    Returns:
        Configured curriculum callback.
    """
    return JumpCurriculumCallback(env)
