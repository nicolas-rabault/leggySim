"""Keyboard controller for velocity commands using arrow keys.

Up/Down arrows control forward speed (lin_vel_x).
Left/Right arrows control yaw rotation (ang_vel_z).
"""

import threading

import numpy as np

from mjlab_leggy.leggy.leggy_curriculums import _FINAL

_KEY_UP = 265
_KEY_DOWN = 264
_KEY_LEFT = 263
_KEY_RIGHT = 262
_KEY_COLON = 46


class KeyboardController:
    """Controls velocity targets via arrow key input."""

    def __init__(
        self,
        lin_vel_x_step: float = 0.1,
        ang_vel_z_step: float = 0.2,
        lin_vel_x_range: tuple[float, float] = _FINAL["lin_vel_x"],
        ang_vel_z_range: tuple[float, float] = _FINAL["ang_vel_z"],
    ):
        self.lin_vel_x = 0.0
        self.ang_vel_z = 0.0
        self.jump = False
        self.lin_vel_x_step = lin_vel_x_step
        self.ang_vel_z_step = ang_vel_z_step
        self.lin_vel_x_range = lin_vel_x_range
        self.ang_vel_z_range = ang_vel_z_range
        self._jump_timer: threading.Timer | None = None

    def key_callback(self, keycode: int) -> None:
        """Handle a key press."""
        if keycode == _KEY_UP:
            self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_x_step, self.lin_vel_x_range[1])
        elif keycode == _KEY_DOWN:
            self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_x_step, self.lin_vel_x_range[0])
        elif keycode == _KEY_RIGHT:
            self.ang_vel_z = max(self.ang_vel_z - self.ang_vel_z_step, self.ang_vel_z_range[0])
        elif keycode == _KEY_LEFT:
            self.ang_vel_z = min(self.ang_vel_z + self.ang_vel_z_step, self.ang_vel_z_range[1])
        elif keycode == _KEY_COLON:
            if self._jump_timer is not None:
                self._jump_timer.cancel()
            self.jump = True
            print("Jump: ON")
            self._jump_timer = threading.Timer(0.7, self._jump_off)
            self._jump_timer.start()
            return
        else:
            return
        self._print_target()

    @property
    def command(self) -> np.ndarray:
        """Current command as [lin_vel_x, lin_vel_y, ang_vel_z]."""
        return np.array([self.lin_vel_x, 0.0, self.ang_vel_z], dtype=np.float32)

    def apply_to_env(self, env, command_name: str = "twist") -> None:
        """Override the environment's velocity and jump commands with keyboard values."""
        cmd_term = env.unwrapped.command_manager.get_term(command_name)
        cmd_term.vel_command_b[:, 0] = self.lin_vel_x
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = self.ang_vel_z
        cmd_term.time_left[:] = 1e9

        if "jump" in env.unwrapped.command_manager._terms:
            jump_term = env.unwrapped.command_manager.get_term("jump")
            jump_term._jump_cmd[:] = 1.0 if self.jump else 0.0
            jump_term.time_left[:] = 1e9

    def _jump_off(self) -> None:
        self.jump = False
        print("Jump: OFF")

    def _print_target(self) -> None:
        print(f"Target -> lin_vel_x: {self.lin_vel_x:+.2f} m/s  |  ang_vel_z: {self.ang_vel_z:+.2f} rad/s")
