"""Left-right symmetry augmentation for Leggy biped.

Mirrors observations and actions by swapping left/right joints and
negating lateral components. Used as data_augmentation_func in rsl_rl's
symmetry_cfg to double training data with physical symmetry.

Policy observation layout per timestep (36 dims):
  [0:3]   base_lin_vel   — vx, vy, vz
  [3:6]   base_ang_vel   — wx, wy, wz
  [6:12]  joint_pos      — LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor
  [12:18] joint_vel      — same order
  [18:24] actions        — same order
  [24:27] command        — vx_cmd, vy_cmd, wz_cmd
  [27:30] body_euler     — roll, pitch, yaw
  [30:36] joint_torques  — LhipY, LhipX, Lknee, RhipY, RhipX, Rknee
"""

import torch

OBS_DIM = 36
HISTORY_LENGTH = 5

# Per-timestep permutation: source index for each output element
_OBS_PERM = [
    0, 1, 2,              # base_lin_vel: vx, vy, vz
    3, 4, 5,              # base_ang_vel: wx, wy, wz
    9, 10, 11, 6, 7, 8,   # joint_pos: R→L slot, L→R slot
    15, 16, 17, 12, 13, 14,  # joint_vel: same swap
    21, 22, 23, 18, 19, 20,  # actions: same swap
    24, 25, 26,            # command: vx, vy, wz
    27, 28, 29,            # body_euler: roll, pitch, yaw
    33, 34, 35, 30, 31, 32,  # joint_torques: same swap
]

# Per-timestep sign flips
_OBS_SIGN = [
    1, -1, 1,             # base_lin_vel: negate vy
    -1, 1, -1,            # base_ang_vel: negate wx, wz
    1, -1, 1, 1, -1, 1,   # joint_pos: negate hipX
    1, -1, 1, 1, -1, 1,   # joint_vel: negate hipX
    1, -1, 1, 1, -1, 1,   # actions: negate hipX
    1, -1, -1,            # command: negate vy, wz
    -1, 1, -1,            # body_euler: negate roll, yaw
    1, -1, 1, 1, -1, 1,   # joint_torques: negate hipX
]

# Action permutation (6 dims): swap L↔R, negate hipX
_ACT_PERM = [3, 4, 5, 0, 1, 2]
_ACT_SIGN = [1, -1, 1, 1, -1, 1]

# Precomputed for full flattened observation (history_length * obs_dim)
_FULL_OBS_PERM = []
_FULL_OBS_SIGN = []
for h in range(HISTORY_LENGTH):
    offset = h * OBS_DIM
    _FULL_OBS_PERM.extend([p + offset for p in _OBS_PERM])
    _FULL_OBS_SIGN.extend(_OBS_SIGN)

# Module-level cached tensors (created on first call)
_cached_device = None
_cached_obs_perm = None
_cached_obs_sign = None
_cached_act_perm = None
_cached_act_sign = None


def _ensure_cached(device):
    global _cached_device, _cached_obs_perm, _cached_obs_sign, _cached_act_perm, _cached_act_sign
    if _cached_device == device:
        return
    _cached_obs_perm = torch.tensor(_FULL_OBS_PERM, dtype=torch.long, device=device)
    _cached_obs_sign = torch.tensor(_FULL_OBS_SIGN, dtype=torch.float32, device=device)
    _cached_act_perm = torch.tensor(_ACT_PERM, dtype=torch.long, device=device)
    _cached_act_sign = torch.tensor(_ACT_SIGN, dtype=torch.float32, device=device)
    _cached_device = device


def leggy_mirror(obs=None, actions=None, env=None):
    """Mirror observations and/or actions for left-right symmetry.

    Returns tensors with doubled batch size: [original; mirrored].
    """
    obs_out = None
    act_out = None

    if obs is not None:
        _ensure_cached(obs.device)
        mirrored = obs[:, _cached_obs_perm] * _cached_obs_sign
        obs_out = torch.cat([obs, mirrored], dim=0)

    if actions is not None:
        device = actions.device
        _ensure_cached(device)
        mirrored = actions[:, _cached_act_perm] * _cached_act_sign
        act_out = torch.cat([actions, mirrored], dim=0)

    return obs_out, act_out
