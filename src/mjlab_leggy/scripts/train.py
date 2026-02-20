"""Training script that uses LeggyRlEnv for torque diagnostics.

Reuses mjlab's training logic but swaps in LeggyRlEnv (which tracks
per-episode mean/peak torques logged to wandb under Torque/).

Usage:
    uv run leggy-train Mjlab-Stand-up-Flat-Leggy [--flags]
"""

import mjlab.scripts.train as _train

from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

# Replace the env class used by the training script.
_train.ManagerBasedRlEnv = LeggyRlEnv


def main():
    _train.main()


if __name__ == "__main__":
    main()
