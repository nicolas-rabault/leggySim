"""Training script with torque diagnostics.

Usage:
    uv run leggy-train Mjlab-Leggy [--flags]
"""

import mjlab.scripts.train as _train

from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

_train.ManagerBasedRlEnv = LeggyRlEnv


def main():
    _train.main()


if __name__ == "__main__":
    main()
