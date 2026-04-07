"""Custom ManagerBasedRlEnv with torque diagnostics logging.

Subclasses ManagerBasedRlEnv to track per-episode mean, peak, and saturation
torques for hipY, hipX, and knee joints. Metrics appear in wandb under Torque/.
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
# Actuator force limit from robot.xml forcerange="-2.36 2.36"
_FORCE_LIMIT = 2.36


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
        self._torque_saturated_steps = torch.zeros(self.num_envs, 6, device=self.device)
        self._torque_step_count = torch.zeros(self.num_envs, device=self.device)

    def step(self, action: torch.Tensor):
        result = super().step(action)

        # Accumulate torque data (runs after physics and after resets,
        # so reset envs start accumulating for their new episode).
        asset = self.scene["robot"]
        abs_torques = asset.data.actuator_force[:, self._torque_actuator_ids].abs()
        self._torque_abs_sum += abs_torques
        self._torque_abs_peak = torch.max(self._torque_abs_peak, abs_torques)
        self._torque_saturated_steps += (abs_torques >= _FORCE_LIMIT - 0.01).float()
        self._torque_step_count += 1

        return result

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None and len(env_ids) > 0:
            # Compute metrics for completed episodes before resetting.
            count = self._torque_step_count[env_ids].unsqueeze(-1).clamp(min=1)
            mean_torques = self._torque_abs_sum[env_ids] / count
            peak_torques = self._torque_abs_peak[env_ids]
            sat_ratio = self._torque_saturated_steps[env_ids] / count

            # Store temporarily — parent's _reset_idx overwrites extras["log"].
            self._pending_torque_log = {}
            for name, l_idx, r_idx in _JOINT_GROUPS:
                l_mean = mean_torques[:, l_idx]
                r_mean = mean_torques[:, r_idx]
                l_peak = peak_torques[:, l_idx]
                r_peak = peak_torques[:, r_idx]
                l_sat = sat_ratio[:, l_idx]
                r_sat = sat_ratio[:, r_idx]
                self._pending_torque_log[f"Torque/L{name}_mean"] = torch.mean(l_mean)
                self._pending_torque_log[f"Torque/R{name}_mean"] = torch.mean(r_mean)
                self._pending_torque_log[f"Torque/L{name}_peak"] = torch.mean(l_peak)
                self._pending_torque_log[f"Torque/R{name}_peak"] = torch.mean(r_peak)
                self._pending_torque_log[f"Torque/L{name}_sat"] = torch.mean(l_sat)
                self._pending_torque_log[f"Torque/R{name}_sat"] = torch.mean(r_sat)

            # Reset accumulators for these environments.
            self._torque_abs_sum[env_ids] = 0.0
            self._torque_abs_peak[env_ids] = 0.0
            self._torque_saturated_steps[env_ids] = 0.0
            self._torque_step_count[env_ids] = 0.0

        # Call parent (creates extras["log"] from all managers).
        super()._reset_idx(env_ids)

        # Restore ctrl to default joint positions so PD actuators don't spike.
        # mjwarp.reset_data zeroes ctrl but sets qpos to home — causing a huge
        # torque impulse on the first forward() call. Fix by writing the default
        # joint positions into joint_pos_target before the next write_data_to_sim.
        if env_ids is not None and len(env_ids) > 0:
            robot = self.scene["robot"]
            robot.set_joint_position_target(
                robot.data.default_joint_pos[env_ids], env_ids=env_ids
            )

        # Inject our metrics into extras["log"].
        if hasattr(self, "_pending_torque_log"):
            self.extras["log"].update(self._pending_torque_log)
            del self._pending_torque_log
