"""Left-right symmetry augmentation for Leggy biped.

Mirrors observations and actions by swapping left/right joints and
negating lateral/yaw components. Joint values (pos, vel, actions, torques)
are swapped L↔R without sign changes (joint conventions are symmetric).

Policy observation layout per timestep (36 dims, ×5 history = 180):
  [0:3]   base_lin_vel   — vx, vy, vz
  [3:6]   base_ang_vel   — wx, wy, wz
  [6:12]  joint_pos      — LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor
  [12:18] joint_vel      — same order
  [18:24] actions        — same order
  [24:27] command        — vx_cmd, vy_cmd, wz_cmd
  [27:30] body_euler     — roll, pitch, yaw
  [30:36] joint_torques  — LhipY, LhipX, Lknee, RhipY, RhipX, Rknee

Critic observation layout (48 dims):
  [0:27]  same as policy per-timestep (through command)
  [27:29] foot_height         — L, R
  [29:31] foot_air_time       — L, R
  [31:33] foot_contact        — L, R
  [33:39] foot_contact_forces — Lx, Ly, Lz, Rx, Ry, Rz
  [39:42] body_euler          — roll, pitch, yaw
  [42:48] joint_torques       — LhipY, LhipX, Lknee, RhipY, RhipX, Rknee
"""

import torch
from tensordict import TensorDict

OBS_DIM = 36
HISTORY_LENGTH = 5

# Per-timestep policy obs permutation (36 dims)
_POLICY_STEP_PERM = [
    0, 1, 2,                           # base_lin_vel
    3, 4, 5,                           # base_ang_vel
    9, 10, 11, 6, 7, 8,               # joint_pos: swap L↔R
    15, 16, 17, 12, 13, 14,           # joint_vel: swap L↔R
    21, 22, 23, 18, 19, 20,           # actions: swap L↔R
    24, 25, 26,                        # command
    27, 28, 29,                        # body_euler
    33, 34, 35, 30, 31, 32,           # joint_torques: swap L↔R
]
_POLICY_STEP_SIGN = [
    1, -1, 1,                          # base_lin_vel: negate vy
    -1, 1, -1,                         # base_ang_vel: negate wx, wz
    1, 1, 1, 1, 1, 1,                 # joint_pos: swap only
    1, 1, 1, 1, 1, 1,                 # joint_vel
    1, 1, 1, 1, 1, 1,                 # actions
    1, -1, -1,                         # command: negate vy, wz
    -1, 1, -1,                         # body_euler: negate roll, yaw
    1, 1, 1, 1, 1, 1,                 # joint_torques
]

# Full policy obs: tile across history frames
_POLICY_PERM = []
_POLICY_SIGN = []
for h in range(HISTORY_LENGTH):
    off = h * OBS_DIM
    _POLICY_PERM.extend([p + off for p in _POLICY_STEP_PERM])
    _POLICY_SIGN.extend(_POLICY_STEP_SIGN)

# Critic obs permutation (48 dims) — foot terms between command and body_euler
_CRITIC_PERM = [
    0, 1, 2,                           # [0:3] base_lin_vel
    3, 4, 5,                           # [3:6] base_ang_vel
    9, 10, 11, 6, 7, 8,               # [6:12] joint_pos: swap L↔R
    15, 16, 17, 12, 13, 14,           # [12:18] joint_vel: swap L↔R
    21, 22, 23, 18, 19, 20,           # [18:24] actions: swap L↔R
    24, 25, 26,                        # [24:27] command
    28, 27,                            # [27:29] foot_height: swap L↔R
    30, 29,                            # [29:31] foot_air_time: swap L↔R
    32, 31,                            # [31:33] foot_contact: swap L↔R
    36, 37, 38, 33, 34, 35,           # [33:39] foot_contact_forces: swap feet
    39, 40, 41,                        # [39:42] body_euler
    45, 46, 47, 42, 43, 44,           # [42:48] joint_torques: swap L↔R
]
_CRITIC_SIGN = [
    1, -1, 1,                          # base_lin_vel: negate vy
    -1, 1, -1,                         # base_ang_vel: negate wx, wz
    1, 1, 1, 1, 1, 1,                 # joint_pos: swap only
    1, 1, 1, 1, 1, 1,                 # joint_vel
    1, 1, 1, 1, 1, 1,                 # actions
    1, -1, -1,                         # command: negate vy, wz
    1, 1,                              # foot_height
    1, 1,                              # foot_air_time
    1, 1,                              # foot_contact
    1, -1, 1, 1, -1, 1,               # foot_contact_forces: negate y
    -1, 1, -1,                         # body_euler: negate roll, yaw
    1, 1, 1, 1, 1, 1,                 # joint_torques
]

# Action permutation (6 dims): swap L↔R
_ACT_PERM = [3, 4, 5, 0, 1, 2]
_ACT_SIGN = [1, 1, 1, 1, 1, 1]

# Cached tensors per device
_cache = {}


def _get_cached(device):
    if device not in _cache:
        _cache[device] = {
            "policy_perm": torch.tensor(_POLICY_PERM, dtype=torch.long, device=device),
            "policy_sign": torch.tensor(_POLICY_SIGN, dtype=torch.float32, device=device),
            "critic_perm": torch.tensor(_CRITIC_PERM, dtype=torch.long, device=device),
            "critic_sign": torch.tensor(_CRITIC_SIGN, dtype=torch.float32, device=device),
            "act_perm": torch.tensor(_ACT_PERM, dtype=torch.long, device=device),
            "act_sign": torch.tensor(_ACT_SIGN, dtype=torch.float32, device=device),
        }
    return _cache[device]


def _mirror_tensor(tensor, perm, sign):
    mirrored = tensor[:, perm] * sign
    return torch.cat([tensor, mirrored], dim=0)


def leggy_mirror(obs=None, actions=None, env=None):
    """Mirror observations and/or actions for left-right symmetry.

    obs is a TensorDict with "policy" (180 dims) and "critic" (48 dims) keys.
    Returns TensorDict/tensor with doubled batch size: [original; mirrored].
    """
    obs_out = None
    act_out = None

    if obs is not None:
        device = obs.device
        c = _get_cached(device)
        mirrored_dict = {}
        for key in obs.keys():
            tensor = obs[key]
            if tensor.shape[-1] == HISTORY_LENGTH * OBS_DIM:
                mirrored_dict[key] = _mirror_tensor(tensor, c["policy_perm"], c["policy_sign"])
            elif tensor.shape[-1] == 48:
                mirrored_dict[key] = _mirror_tensor(tensor, c["critic_perm"], c["critic_sign"])
            else:
                mirrored_dict[key] = torch.cat([tensor, tensor], dim=0)
        obs_out = TensorDict(mirrored_dict, batch_size=[obs.batch_size[0] * 2])

    if actions is not None:
        c = _get_cached(actions.device)
        act_out = _mirror_tensor(actions, c["act_perm"], c["act_sign"])

    return obs_out, act_out
