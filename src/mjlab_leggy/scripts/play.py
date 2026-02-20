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


class KeyboardViewer(NativeMujocoViewer):
    """NativeMujocoViewer with keyboard velocity control."""

    def __init__(self, env, policy):
        self.kb = KeyboardController()
        super().__init__(env, policy, key_callback=self.kb.key_callback)

    def step_simulation(self):
        if not self._is_paused:
            self.kb.apply_to_env(self.env)
        super().step_simulation()


_play.ManagerBasedRlEnv = LeggyRlEnv
_play.NativeMujocoViewer = KeyboardViewer


def main():
    print("Arrow keys: Up/Down = forward speed, Left/Right = rotation")
    _play.main()


if __name__ == "__main__":
    main()
