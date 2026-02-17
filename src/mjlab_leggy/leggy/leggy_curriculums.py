"""Curriculum stage definitions for Leggy training.

Provides velocity stage definitions for progressive training curricula.
"""

from .leggy_constants import NUM_STEPS_PER_ENV

# -------------------------------------------------------------------------
# Velocity Stages (for locomotion curriculum)
# -------------------------------------------------------------------------

# Standard velocity curriculum: progressively increase running speed
# Linearly interpolates from near-zero to the final target values.
_N_STAGES = 8
_LAST_ITERATION = 40000
_LAST_STEP = _LAST_ITERATION * NUM_STEPS_PER_ENV
_INITIAL = {"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)}
_FINAL = {"lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-1.0, 1.0)}

VELOCITY_STAGES_STANDARD = []
for _i in range(_N_STAGES):
    _t = _i / (_N_STAGES - 1)
    _stage = {"step": round(_LAST_STEP * _t)}
    for _key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        _lo = _INITIAL[_key][0] + _t * (_FINAL[_key][0] - _INITIAL[_key][0])
        _hi = _INITIAL[_key][1] + _t * (_FINAL[_key][1] - _INITIAL[_key][1])
        _stage[_key] = (round(_lo, 4), round(_hi, 4))
    VELOCITY_STAGES_STANDARD.append(_stage)


__all__ = [
    "VELOCITY_STAGES_STANDARD",
]
