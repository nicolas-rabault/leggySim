"""Training script with torque diagnostics and symmetry augmentation.

Usage:
    uv run leggy-train Mjlab-Leggy [--flags]
"""

import mjlab.scripts.train as _train
from mjlab.rl import MjlabOnPolicyRunner

from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

_train.ManagerBasedRlEnv = LeggyRlEnv

SYMMETRY_CFG = {
    "use_data_augmentation": True,
    "use_mirror_loss": False,
    "data_augmentation_func": "mjlab_leggy.leggy.leggy_symmetry:leggy_mirror",
    "mirror_loss_coeff": 0.0,
}

_orig_construct = MjlabOnPolicyRunner._construct_algorithm


def _construct_with_symmetry(self, obs):
    self.alg_cfg["symmetry_cfg"] = SYMMETRY_CFG
    return _orig_construct(self, obs)


MjlabOnPolicyRunner._construct_algorithm = _construct_with_symmetry


def main():
    _train.main()


if __name__ == "__main__":
    main()
