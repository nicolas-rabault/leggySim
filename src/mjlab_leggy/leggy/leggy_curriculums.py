"""Curriculum stage definitions for Leggy training."""

import torch

from .leggy_constants import NUM_STEPS_PER_ENV

_N_STAGES = 32
_LAST_ITERATION = 80000
_LAST_STEP = _LAST_ITERATION * NUM_STEPS_PER_ENV
_INITIAL = {"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)}
_FINAL = {"lin_vel_x": (-2, 3.0), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-2.0, 2.0)}

# Staggered ramps: lin_vel finishes early (50%), ang_vel starts late (25%) and
# finishes at 90%. This prevents "winner takes all" where the policy optimizes
# lin_vel and abandons ang_vel when both get harder simultaneously.
_RAMP = {
    "lin_vel_x": (0.0, 0.5),
    "lin_vel_y": (0.0, 0.5),
    "ang_vel_z": (0.25, 0.9),
}


def _ramp_t(global_t: float, start: float, end: float) -> float:
    if global_t <= start:
        return 0.0
    if global_t >= end:
        return 1.0
    return (global_t - start) / (end - start)


VELOCITY_STAGES_STANDARD = []
for _i in range(_N_STAGES):
    _t = _i / (_N_STAGES - 1)
    _stage = {"step": round(_LAST_STEP * _t)}
    for _key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        _rt = _ramp_t(_t, *_RAMP[_key])
        _lo = _INITIAL[_key][0] + _rt * (_FINAL[_key][0] - _INITIAL[_key][0])
        _hi = _INITIAL[_key][1] + _rt * (_FINAL[_key][1] - _INITIAL[_key][1])
        _stage[_key] = (round(_lo, 4), round(_hi, 4))
    VELOCITY_STAGES_STANDARD.append(_stage)


class commands_vel_gated:
    """Performance-gated velocity curriculum.

    Stages advance only when step minimum is met AND both tracking
    reward EMAs exceed gate_threshold. EMA tracks raw reward function
    output (0-1 range) by dividing episode sums by weight * dt * steps.
    """

    def __init__(self, cfg, env):
        self.current_stage = 0
        self.ema_lin = 0.0
        self.ema_ang = 0.0
        self._dt = env.step_dt

    def reset(self, env_ids):
        pass  # No per-env state to reset

    def __call__(
        self,
        env,
        env_ids,
        command_name: str,
        velocity_stages: list,
        reward_terms: tuple[str, str] = ("track_linear_velocity", "track_angular_velocity"),
        gate_threshold: float = 0.5,
        ema_alpha: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        # Update EMAs from completed episodes
        if not isinstance(env_ids, slice):
            ep_lens = env.episode_length_buf[env_ids].float().clamp(min=1.0)
            rm = env.reward_manager
            for i, term_name in enumerate(reward_terms):
                ep_sum = rm._episode_sums[term_name][env_ids]
                weight = abs(rm._term_cfgs[rm._term_names.index(term_name)].weight)
                raw = ep_sum / (weight * self._dt * ep_lens)
                mean_val = raw.mean().item()
                if i == 0:
                    self.ema_lin = ema_alpha * mean_val + (1 - ema_alpha) * self.ema_lin
                else:
                    self.ema_ang = ema_alpha * mean_val + (1 - ema_alpha) * self.ema_ang

        # Check advancement
        gated = False
        next_stage = self.current_stage + 1
        if next_stage < len(velocity_stages):
            step_ready = env.common_step_counter > velocity_stages[next_stage]["step"]
            perf_ready = self.ema_lin >= gate_threshold and self.ema_ang >= gate_threshold
            if step_ready and perf_ready:
                self.current_stage = next_stage
            elif step_ready and not perf_ready:
                gated = True

        # Apply current stage velocities
        stage = velocity_stages[self.current_stage]
        cmd_term = env.command_manager.get_term(command_name)
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            if key in stage:
                setattr(cmd_term.cfg.ranges, key, stage[key])

        device = env.device
        return {
            "stage": torch.tensor(self.current_stage, dtype=torch.float32, device=device),
            "ema_lin_vel": torch.tensor(self.ema_lin, dtype=torch.float32, device=device),
            "ema_ang_vel": torch.tensor(self.ema_ang, dtype=torch.float32, device=device),
            "gated": torch.tensor(float(gated), device=device),
            "lin_vel_x": torch.tensor(stage.get("lin_vel_x", (0, 0))[1], dtype=torch.float32, device=device),
            "ang_vel_z": torch.tensor(stage.get("ang_vel_z", (0, 0))[1], dtype=torch.float32, device=device),
        }
