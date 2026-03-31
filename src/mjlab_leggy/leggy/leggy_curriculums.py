"""Curriculum stage definitions for Leggy training."""

from .leggy_constants import NUM_STEPS_PER_ENV

_N_STAGES = 32
_LAST_ITERATION = 80000
_LAST_STEP = _LAST_ITERATION * NUM_STEPS_PER_ENV
_INITIAL = {"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)}
_FINAL = {"lin_vel_x": (-1.5, 2.0), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-2.0, 2.0)}

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
