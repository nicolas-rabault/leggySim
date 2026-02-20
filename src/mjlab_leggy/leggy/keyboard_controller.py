"""Keyboard controller for velocity commands using arrow keys.

Designed as a key_callback for mujoco.viewer.launch_passive or NativeMujocoViewer.
Up/Down arrows control forward speed (lin_vel_x).
Left/Right arrows control yaw rotation (ang_vel_z).
"""

import numpy as np

# GLFW key codes for arrow keys
_KEY_UP = 265
_KEY_DOWN = 264
_KEY_LEFT = 263
_KEY_RIGHT = 262


class KeyboardController:
    """Controls velocity targets via arrow key input.

    Works with mjlab environments (used by leggy-play).

    Args:
        lin_vel_x_step: Increment per key press for forward speed (m/s).
        ang_vel_z_step: Increment per key press for yaw rate (rad/s).
        lin_vel_x_range: (min, max) clamp for forward speed.
        ang_vel_z_range: (min, max) clamp for yaw rate.
    """

    def __init__(
        self,
        lin_vel_x_step: float = 0.1,
        ang_vel_z_step: float = 0.2,
        lin_vel_x_range: tuple[float, float] = (-0.5, 1.0),
        ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.lin_vel_x = 0.0
        self.ang_vel_z = 0.0
        self.lin_vel_x_step = lin_vel_x_step
        self.ang_vel_z_step = ang_vel_z_step
        self.lin_vel_x_range = lin_vel_x_range
        self.ang_vel_z_range = ang_vel_z_range

    def key_callback(self, keycode: int) -> None:
        """Handle a key press. Pass this to mujoco.viewer.launch_passive(key_callback=...)."""
        if keycode == _KEY_UP:
            self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_x_step, self.lin_vel_x_range[1])
        elif keycode == _KEY_DOWN:
            self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_x_step, self.lin_vel_x_range[0])
        elif keycode == _KEY_RIGHT:
            self.ang_vel_z = max(self.ang_vel_z - self.ang_vel_z_step, self.ang_vel_z_range[0])
        elif keycode == _KEY_LEFT:
            self.ang_vel_z = min(self.ang_vel_z + self.ang_vel_z_step, self.ang_vel_z_range[1])
        else:
            return

        self._print_target()

    @property
    def command(self) -> np.ndarray:
        """Current command as [lin_vel_x, lin_vel_y, ang_vel_z]."""
        return np.array([self.lin_vel_x, 0.0, self.ang_vel_z], dtype=np.float32)

    def apply_to_env(self, env, command_name: str = "twist") -> None:
        """Override the environment's velocity command with keyboard values.

        Call this every step before observations are computed to ensure
        the policy always sees the keyboard-set command.
        """
        cmd_term = env.unwrapped.command_manager.get_term(command_name)
        cmd_term.vel_command_b[:, 0] = self.lin_vel_x
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = self.ang_vel_z
        # Prevent the command manager from resampling over our values.
        cmd_term.time_left[:] = 1e9

    def _print_target(self) -> None:
        print(f"Target -> lin_vel_x: {self.lin_vel_x:+.2f} m/s  |  ang_vel_z: {self.ang_vel_z:+.2f} rad/s")
