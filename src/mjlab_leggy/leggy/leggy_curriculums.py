"""Curriculum stage definitions for Leggy training."""

from .leggy_constants import NUM_STEPS_PER_ENV

_N_STAGES = 10
_LAST_ITERATION = 50000
_LAST_STEP = _LAST_ITERATION * NUM_STEPS_PER_ENV
_INITIAL = {"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)}
_FINAL = {"lin_vel_x": (-2, 3.0), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-2.0, 2.0)}

# Fast early (easy stages plateau in ~2500 steps), slow late (hard stages need consolidation).
# Ramp 0-70% of training, leaving 30% for consolidation at max speeds.
_RAMP = {
    "lin_vel_x": (0.0, 0.7),
    "lin_vel_y": (0.0, 0.7),
    "ang_vel_z": (0.0, 0.7),
}

# Non-uniform stage spacing: early stages are short, late stages are long.
# Steps as fractions of total training: 0, 3%, 8%, 15%, 25%, 37%, 50%, 63%, 77%, 100%
_STAGE_FRACTIONS = [0.0, 0.03, 0.08, 0.15, 0.25, 0.37, 0.50, 0.63, 0.77, 1.0]


def _ramp_t(global_t: float, start: float, end: float) -> float:
    if global_t <= start:
        return 0.0
    if global_t >= end:
        return 1.0
    return (global_t - start) / (end - start)


VELOCITY_STAGES_STANDARD = []
for _i in range(_N_STAGES):
    _t = _STAGE_FRACTIONS[_i]
    _stage = {"step": round(_LAST_STEP * _t)}
    for _key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        _rt = _ramp_t(_t, *_RAMP[_key])
        _lo = _INITIAL[_key][0] + _rt * (_FINAL[_key][0] - _INITIAL[_key][0])
        _hi = _INITIAL[_key][1] + _rt * (_FINAL[_key][1] - _INITIAL[_key][1])
        _stage[_key] = (round(_lo, 4), round(_hi, 4))
    VELOCITY_STAGES_STANDARD.append(_stage)
