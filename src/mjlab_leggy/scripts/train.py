"""Training script with torque diagnostics and symmetry augmentation.

Usage:
    uv run leggy-train Mjlab-Leggy [--flags]
"""

import json

import mjlab.scripts.train as _train
from mjlab.rl import MjlabOnPolicyRunner

from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

_train.ManagerBasedRlEnv = LeggyRlEnv

# Patch dump_yaml: env config can contain un-serializable MjSpec objects
_orig_dump_yaml = _train.dump_yaml


def _safe_dump_yaml(filename, data, sort_keys=False):
    try:
        _orig_dump_yaml(filename, data, sort_keys=sort_keys)
    except TypeError:
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(str(filename).replace(".yaml", ".json"), "w") as f:
            json.dump(data, f, indent=2, default=str)


_train.dump_yaml = _safe_dump_yaml

SYMMETRY_CFG = {
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "data_augmentation_func": "mjlab_leggy.leggy.leggy_symmetry:leggy_mirror",
    "mirror_loss_coeff": 5.0,
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
