"""Play script with keyboard-controlled velocity commands.

Arrow keys: Up/Down = forward speed, Left/Right = rotation.

Usage:
    uv run leggy-play Mjlab-Leggy --wandb-run-path <path>
    uv run leggy-play Mjlab-Leggy --checkpoint-file <path>
    uv run leggy-play Mjlab-Leggy --agent random
"""

import mjlab.scripts.play as _play
from mjlab.viewer import NativeMujocoViewer

from mjlab_leggy.leggy.keyboard_controller import KeyboardController
from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

# -- Debug observation logging (first N steps) --
DEBUG_OBS_STEPS = 3

# Observation terms to log, matching the web viewer's OBS_TERMS order.
_OBS_TERM_FUNCS = None

def _get_obs_term_funcs():
    global _OBS_TERM_FUNCS
    if _OBS_TERM_FUNCS is not None:
        return _OBS_TERM_FUNCS
    from mjlab_leggy.leggy.leggy_observations import base_lin_vel, base_ang_vel, velocity_commands, last_action
    from mjlab_leggy.leggy.leggy_actions import joint_pos_motor, joint_vel_motor, joint_torques_motor, body_euler
    from mjlab_leggy.leggy.leggy_jump import jump_command_obs
    _OBS_TERM_FUNCS = [
        ("base_lin_vel",   lambda env: base_lin_vel(env)),
        ("base_ang_vel",   lambda env: base_ang_vel(env)),
        ("joint_pos",      lambda env: joint_pos_motor(env)),
        ("joint_vel",      lambda env: joint_vel_motor(env)),
        ("actions",        lambda env: last_action(env)),
        ("command",        lambda env: velocity_commands(env, command_name="twist")),
        ("body_euler",     lambda env: body_euler(env)),
        ("joint_torques",  lambda env: joint_torques_motor(env)),
        ("jump_command",   lambda env: jump_command_obs(env, command_name="jump")),
    ]
    return _OBS_TERM_FUNCS

def _fmt(tensor, env_idx=0):
    """Format tensor values for env_idx to 6 decimal places."""
    vals = tensor[env_idx].detach().cpu().tolist()
    if isinstance(vals, float):
        return f"{vals:.6f}"
    return ", ".join(f"{v:.6f}" for v in vals)


class KeyboardViewer(NativeMujocoViewer):
    """NativeMujocoViewer with keyboard velocity control."""

    def __init__(self, env, policy):
        self.kb = KeyboardController()
        self._debug_step = 0
        super().__init__(env, policy, key_callback=self.kb.key_callback)

    def step_simulation(self):
        if not self._is_paused:
            self.kb.apply_to_env(self.env)

        # Log raw observation terms BEFORE the step (same timing as web viewer).
        if self._debug_step < DEBUG_OBS_STEPS and not self._is_paused:
            raw_env = self.env.unwrapped
            print(f"\n=== Policy step {self._debug_step} ===")
            for name, func in _get_obs_term_funcs():
                print(f"  {name}: {_fmt(func(raw_env), self.env_idx)}")

        super().step_simulation()

        # Print step count after the step completes.
        if self._debug_step < DEBUG_OBS_STEPS and not self._is_paused:
            print(f"  -> step_count: {self._step_count}")
            self._debug_step += 1


_play.ManagerBasedRlEnv = LeggyRlEnv
_play.NativeMujocoViewer = KeyboardViewer


def main():
    print("Arrow keys: Up/Down = forward speed, Left/Right = rotation, ; = jump")
    _play.main()


if __name__ == "__main__":
    main()
