"""Custom ManagerBasedRlEnv with torque diagnostics logging.

Subclasses ManagerBasedRlEnv to track per-episode mean and peak torques
for hipY, hipX, and knee joints. Metrics appear in wandb under Torque/.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


# Actuator order: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
_ACTUATOR_NAMES = ("LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee")
# Joint groups: (name, left_index, right_index)
_JOINT_GROUPS = [
    ("hipY", 0, 3),
    ("hipX", 1, 4),
    ("knee", 2, 5),
]


class LeggyRlEnv(ManagerBasedRlEnv):
    """ManagerBasedRlEnv with per-episode torque metrics."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)

        # Resolve actuator indices once.
        asset = self.scene["robot"]
        self._torque_actuator_ids = asset.find_actuators(_ACTUATOR_NAMES)[0]

        # Per-env accumulators: [num_envs, 6]
        self._torque_abs_sum = torch.zeros(self.num_envs, 6, device=self.device)
        self._torque_abs_peak = torch.zeros(self.num_envs, 6, device=self.device)
        self._torque_step_count = torch.zeros(self.num_envs, device=self.device)

    def step(self, action: torch.Tensor):
        result = super().step(action)

        # Accumulate torque data (runs after physics and after resets,
        # so reset envs start accumulating for their new episode).
        asset = self.scene["robot"]
        abs_torques = asset.data.actuator_force[:, self._torque_actuator_ids].abs()
        self._torque_abs_sum += abs_torques
        self._torque_abs_peak = torch.max(self._torque_abs_peak, abs_torques)
        self._torque_step_count += 1

        return result

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None and len(env_ids) > 0:
            # Compute metrics for completed episodes before resetting.
            count = self._torque_step_count[env_ids].unsqueeze(-1).clamp(min=1)
            mean_torques = self._torque_abs_sum[env_ids] / count
            peak_torques = self._torque_abs_peak[env_ids]

            # Store temporarily — parent's _reset_idx overwrites extras["log"].
            self._pending_torque_log = {}
            for name, l_idx, r_idx in _JOINT_GROUPS:
                mean_val = (mean_torques[:, l_idx] + mean_torques[:, r_idx]) / 2.0
                peak_val = torch.max(peak_torques[:, l_idx], peak_torques[:, r_idx])
                self._pending_torque_log[f"Torque/mean_{name}"] = torch.mean(mean_val)
                self._pending_torque_log[f"Torque/peak_{name}"] = torch.mean(peak_val)

            # Reset accumulators for these environments.
            self._torque_abs_sum[env_ids] = 0.0
            self._torque_abs_peak[env_ids] = 0.0
            self._torque_step_count[env_ids] = 0.0

        # Call parent (creates extras["log"] from all managers).
        super()._reset_idx(env_ids)

        # Inject our metrics into extras["log"].
        if hasattr(self, "_pending_torque_log"):
            self.extras["log"].update(self._pending_torque_log)
            del self._pending_torque_log
