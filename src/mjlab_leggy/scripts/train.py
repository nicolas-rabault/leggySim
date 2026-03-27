"""Training script with torque diagnostics and symmetry augmentation.

Usage:
    uv run leggy-train Mjlab-Leggy [--flags]
"""

import mjlab.scripts.train as _train

from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

_train.ManagerBasedRlEnv = LeggyRlEnv

_orig_run_train = _train.run_train


def _run_train_with_symmetry(task_id, cfg, log_dir):
    """Wrap run_train to inject symmetry_cfg into the algorithm config."""
    _orig_asdict = _train.asdict

    def _patched_asdict(obj):
        result = _orig_asdict(obj)
        if hasattr(obj, "algorithm"):
            result["algorithm"]["symmetry_cfg"] = {
                "use_data_augmentation": True,
                "use_mirror_loss": False,
                "data_augmentation_func": "mjlab_leggy.leggy.leggy_symmetry:leggy_mirror",
                "mirror_loss_coeff": 0.0,
            }
        return result

    _train.asdict = _patched_asdict
    try:
        _orig_run_train(task_id, cfg, log_dir)
    finally:
        _train.asdict = _orig_asdict


_train.run_train = _run_train_with_symmetry


def main():
    _train.main()


if __name__ == "__main__":
    main()
