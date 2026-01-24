"""Custom action term for Leggy robot with motor-to-knee conversion and passive joint handling.

This module provides an action term that handles Leggy's unique kinematic chain:
- Converts motor commands to knee angles using hipX offset
- Automatically updates passive joints to follow knee positions
- Works correctly with parallel environments (no global state)

Usage in your task configuration:
    from mjlab_leggy.leggy.leggy_actions import LeggyJointActionCfg

    cfg.actions = {
        "joint_pos": LeggyJointActionCfg(
            asset_name="robot",
            scale=0.5,
            use_default_offset=True,
        )
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg
import time

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# =============================================================================
# Kinematic Conversion Functions
# =============================================================================

def motor_to_knee(motor: torch.Tensor | float, hipX: torch.Tensor | float) -> torch.Tensor | float:
    """Convert motor command to knee angle.

    Leggy's knee is mechanically offset by the hipX angle.
    The motor command must be adjusted to get the actual knee angle.

    Args:
        motor: Motor command angle [rad] (tensor or scalar)
        hipX: Current hipX joint angle [rad] (tensor or scalar)

    Returns:
        Knee angle [rad] (same type as input)
    """
    result = hipX + motor
    return result


def knee_to_motor(knee: torch.Tensor | float, hipX: torch.Tensor | float) -> torch.Tensor | float:
    """Convert knee angle to motor command.

    Used for observations: convert actual knee angle to motor representation.

    Args:
        knee: Knee joint angle [rad] (tensor or scalar)
        hipX: Current hipX joint angle [rad] (tensor or scalar)

    Returns:
        Motor command angle [rad] (same type as input)
    """
    result = knee - hipX
    return result


# =============================================================================
# Action Term Implementation
# =============================================================================

class LeggyJointAction(ActionTerm):
    """Action term that handles Leggy's motor-to-knee conversion and passive joints.

    This action term processes policy actions through the following pipeline:

    1. **process_actions()** - Called once per environment step:
       - Receives raw policy actions: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
       - Applies scale and offset transformations

    2. **apply_actions()** - Called every physics step (decimation times):
       - Reads current hipX angles from robot state
       - Converts Lmotor → Lknee and Rmotor → Rknee using hipX
       - Sets actuated joint targets (6 joints)
       - Sets passive joint targets (4 joints) to match knee angles

    This approach:
    - Avoids global MuJoCo callbacks (no race conditions)
    - Works correctly with parallel environments
    - Maintains per-environment state properly
    - Uses real-time hipX values for accurate conversion
    """

    cfg: LeggyJointActionCfg

    def __init__(self, cfg: LeggyJointActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = env.device

        # =====================================================================
        # Joint Index Resolution
        # =====================================================================
        # Find joint indices for actuated joints in canonical order
        actuated_names = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
        self._actuated_joint_ids = self._asset.find_joints(actuated_names)[0]

        # Find joint indices for passive joints
        passive_names = ["LpassiveMotor", "Lpassive2", "RpassiveMotor", "Rpassive2"]
        self._passive_joint_ids = self._asset.find_joints(passive_names)[0]

        # =====================================================================
        # Action Buffers
        # =====================================================================
        # Policy action dimension is 6 (doesn't directly control passive joints)
        self._action_dim = 6
        self._raw_actions = torch.zeros(self._num_envs, self._action_dim, device=self._device)
        self._processed_actions = torch.zeros(self._num_envs, self._action_dim, device=self._device)

        # =====================================================================
        # Action Scaling and Offset
        # =====================================================================
        self._scale = cfg.scale

        if cfg.use_default_offset:
            # Use default joint positions for the 6 actuated joints as offset
            # This allows the policy to output small deltas around the default pose
            default_pos = self._asset.data.default_joint_pos[:, self._actuated_joint_ids]
            self._offset = default_pos.clone()

            # IMPORTANT: Convert knee defaults to motor defaults for indices 2 and 5
            # The policy works in motor space, but default_joint_pos contains knee angles.
            # This matches the joint_pos computation in leggy_constants.py where:
            #   - ".*knee.*": stand_pose["knee"]  (knee space)
            #   - "LpassiveMotor": knee_to_motor(stand_pose["knee"], stand_pose["hipX"])  (motor space)
            default_hipX_left = default_pos[:, 1]   # LhipX
            default_hipX_right = default_pos[:, 4]  # RhipX
            self._offset[:, 2] = knee_to_motor(default_pos[:, 2], default_hipX_left)   # Lknee → Lmotor
            self._offset[:, 5] = knee_to_motor(default_pos[:, 5], default_hipX_right)  # Rknee → Rmotor
        else:
            self._offset = torch.zeros(self._num_envs, self._action_dim, device=self._device)

        # =====================================================================
        # Debug counters
        # =====================================================================
        self._step_count = 0
        self._debug_interval = 50  # Print every N steps

        # Print initialization info
        print("\n" + "="*80)
        print("LeggyJointAction Initialization")
        print("="*80)
        print(f"Default positions (actuated joints): {default_pos[0].cpu().numpy() if cfg.use_default_offset else 'N/A'}")
        print(f"  [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]")
        if cfg.use_default_offset:
            print(f"  Degrees: {[f'{x:.2f}°' for x in (default_pos[0].cpu().numpy() * 180 / 3.14159)]}")
        print(f"\nOffset (motor space for indices 2,5): {self._offset[0].cpu().numpy()}")
        print(f"  [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]")
        print(f"  Degrees: {[f'{x:.2f}°' for x in (self._offset[0].cpu().numpy() * 180 / 3.14159)]}")
        print(f"\nScale: {self._scale}")
        print("="*80 + "\n")

    @property
    def action_dim(self) -> int:
        """Policy outputs 6 actions: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]."""
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        """Raw actions from policy before any processing."""
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Preprocess actions: apply scale and offset.

        This is called ONCE per environment step (before decimation loop).
        The actual motor-to-knee conversion happens in apply_actions() using
        the current hipX state, since hipX may change during decimation.

        Args:
            actions: Raw actions from policy with shape [num_envs, 6]
                     Columns: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
        """
        self._raw_actions[:] = actions

        # Apply scale and offset to get target joint positions
        # Note: Still in "motor space" for knee joints at indices 2 and 5
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset

        # Debug printing
        self._step_count += 1
        if self._step_count % self._debug_interval == 0:
            env_id = 0
            print(f"\n--- Step {self._step_count} (Env {env_id}) process_actions ---")
            print(f"Raw actions: {self._raw_actions[env_id].cpu().numpy()}")
            print(f"  [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]")
            print(f"Processed actions (motor space): {self._processed_actions[env_id].cpu().numpy()}")
            print(f"  Degrees: {[f'{x:.2f}°' for x in (self._processed_actions[env_id].cpu().numpy() * 180 / 3.14159)]}")

    def apply_actions(self) -> None:
        """Apply actions to actuators with motor-to-knee conversion and passive joint updates.

        This is called EVERY physics step (decimation times per environment step).

        Processing pipeline:
        1. Extract target hipX from action vector (policy's commanded hipX)
        2. Convert motor commands → knee commands using target hipX values
        3. Set actuated joint targets: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
        4. Read CURRENT knee positions from simulator state
        5. Set passive joint targets to match CURRENT knee positions (mechanical coupling)
        """
        # =====================================================================
        # Step 1: Extract target hipX and motor commands from action vector
        # =====================================================================
        # Use the target hipX from the action for motor-to-knee conversion.
        # This ensures the commanded knee angle is consistent with the policy's
        # expectation for the motor command.
        hipX_left_target = self._processed_actions[:, 1]   # LhipX target from policy
        hipX_right_target = self._processed_actions[:, 4]  # RhipX target from policy
        motor_left = self._processed_actions[:, 2]  # Lmotor target from policy
        motor_right = self._processed_actions[:, 5]  # Rmotor target from policy

        # =====================================================================
        # Step 2: Convert motor commands to knee angle targets
        # =====================================================================
        # processed_actions layout: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
        # target_positions layout:  [LhipY, LhipX, Lknee,  RhipY, RhipX, Rknee]
        target_positions = self._processed_actions.clone()
        target_positions[:, 2] = motor_to_knee(motor_left, hipX_left_target)   # Lmotor → Lknee target
        target_positions[:, 5] = motor_to_knee(motor_right, hipX_right_target)  # Rmotor → Rknee target

        # =====================================================================
        # Step 3: Set actuated joint targets
        # =====================================================================
        self._asset.set_joint_position_target(
            target_positions,
            joint_ids=self._actuated_joint_ids,
        )

        # =====================================================================
        # Step 4: Update passive joints based on CURRENT knee positions
        # =====================================================================
        # The passive joints are mechanically coupled to the knee joints.
        # They should follow the ACTUAL knee position in the simulator,
        # not the commanded target. This correctly simulates the mechanical linkage.
        #
        # Since passive joints don't have actuators, we directly write their
        # positions to the simulation state (qpos) rather than setting targets.
        current_joint_pos = self._asset.data.joint_pos[:, self._actuated_joint_ids]
        knee_left_current = current_joint_pos[:, 2]   # Current Lknee position
        hipX_left_current = current_joint_pos[:, 1]   # Current LhipX position
        knee_right_current = current_joint_pos[:, 5]  # Current Rknee position
        hipX_right_current = current_joint_pos[:, 4]   # Current RhipX position

        passive_positions = torch.stack([
            knee_to_motor(knee_left_current, hipX_left_current),   # LpassiveMotor follows current Lknee
            knee_left_current,   # Lpassive2 follows current Lknee
            knee_to_motor(knee_right_current, hipX_right_current),  # RpassiveMotor follows current Rknee
            knee_right_current,  # Rpassive2 follows current Rknee
        ], dim=1)

        # Directly write passive joint positions to simulation (not targets)
        # This is necessary because passive joints don't have actuators
        self._asset.write_joint_position_to_sim(
            passive_positions,
            joint_ids=self._passive_joint_ids,
        )

        # Debug printing (only on the first decimation step of each debug interval)
        if self._step_count % self._debug_interval == 0:
            env_id = 0
            print(f"\n--- Step {self._step_count} (Env {env_id}) apply_actions ---")
            print(f"TARGET hipX_left: {hipX_left_target[env_id].item():.6f} rad = {hipX_left_target[env_id].item() * 180/3.14159:.2f}°")
            print(f"TARGET motor_left: {motor_left[env_id].item():.6f} rad = {motor_left[env_id].item() * 180/3.14159:.2f}°")
            print(f"TARGET knee_left (computed): {target_positions[env_id, 2].item():.6f} rad = {target_positions[env_id, 2].item() * 180/3.14159:.2f}°")
            print(f"  Computation: knee = motor + hipX = {motor_left[env_id].item():.6f} + {hipX_left_target[env_id].item():.6f} = {target_positions[env_id, 2].item():.6f}")
            print(f"\nCURRENT hipX_left: {hipX_left_current[env_id].item():.6f} rad = {hipX_left_current[env_id].item() * 180/3.14159:.2f}°")
            print(f"CURRENT knee_left: {knee_left_current[env_id].item():.6f} rad = {knee_left_current[env_id].item() * 180/3.14159:.2f}°")
            print(f"CURRENT motor_left (computed from current): {passive_positions[env_id, 0].item():.6f} rad = {passive_positions[env_id, 0].item() * 180/3.14159:.2f}°")
            print(f"  Computation: motor = knee - hipX = {knee_left_current[env_id].item():.6f} - {hipX_left_current[env_id].item():.6f} = {passive_positions[env_id, 0].item():.6f}")
            print(f"\nPassive joints (LEFT): LpassiveMotor={passive_positions[env_id, 0].item() * 180/3.14159:.2f}°, Lpassive2={passive_positions[env_id, 1].item() * 180/3.14159:.2f}°")
            print(f"Passive joints (RIGHT): RpassiveMotor={passive_positions[env_id, 2].item() * 180/3.14159:.2f}°, Rpassive2={passive_positions[env_id, 3].item() * 180/3.14159:.2f}°")


# =============================================================================
# Action Term Configuration
# =============================================================================

@dataclass(kw_only=True)
class LeggyJointActionCfg(ActionTermCfg):
    """Configuration for Leggy joint action with motor conversion and passive joints.

    The policy outputs 6 actions for: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
    This action term automatically:
    - Converts motor commands → knee angles using current hipX
    - Updates passive joints (LpassiveMotor, Lpassive2, RpassiveMotor, Rpassive2) to match knees
    - Handles all joint targets per environment

    Attributes:
        class_type: The action term class (automatically set to LeggyJointAction)
        asset_name: Name of the robot asset (default: "robot")
        scale: Action scaling factor applied to policy outputs (default: 0.5)
        use_default_offset: If True, add default joint positions as offset (default: True)
    """
    class_type: type[ActionTerm] = LeggyJointAction
    asset_name: str = "robot"
    scale: float = 0.5
    use_default_offset: bool = True


# =============================================================================
# Observation Helper Function
# =============================================================================

def joint_pos_rel_motor(env, asset_cfg=None) -> torch.Tensor:
    """Get joint positions with knee→motor conversion for observations.

    This converts the actual knee angles in the observation to motor representation,
    which is the inverse of what the action term does. This ensures the policy
    sees consistent motor-space representations.

    Args:
        env: The environment instance
        asset_cfg: Asset configuration (default: SceneEntityCfg("robot") with actuated joints only)

    Returns:
        Joint positions with motor representation [num_envs, 6]
        Layout: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
    """
    from mjlab.envs.mdp import observations as mdp_obs
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    if asset_cfg is None:
        # IMPORTANT: Explicitly specify actuated joints only, in correct order!
        # Without this, joint_pos_rel returns ALL joints (including passive),
        # which would make indices 4,5 point to LpassiveMotor,RhipY instead of RhipX,Rknee
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=("LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee")
        )

    # Get regular joint positions observation (relative to default)
    obs = mdp_obs.joint_pos_rel(env, asset_cfg=asset_cfg).clone()

    # Debug printing
    if hasattr(env, '_leggy_obs_counter'):
        env._leggy_obs_counter += 1
    else:
        env._leggy_obs_counter = 1

    if env._leggy_obs_counter % 50 == 0:
        env_id = 0
        print(f"\n--- Observation {env._leggy_obs_counter} (Env {env_id}) ---")
        print(f"RELATIVE obs (knee space): {obs[env_id].cpu().numpy()}")
        print(f"  [LhipY, LhipX, Lknee_rel, RhipY, RhipX, Rknee_rel]")
        print(f"  Degrees: {[f'{x:.2f}°' for x in (obs[env_id].cpu().numpy() * 180 / 3.14159)]}")
        print(f"  LhipX_rel={obs[env_id, 1].item() * 180/3.14159:.2f}°, Lknee_rel={obs[env_id, 2].item() * 180/3.14159:.2f}°")

    # Convert knee angles to motor representation
    # obs layout: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
    # Index 1 = LhipX, Index 2 = Lknee
    # Index 4 = RhipX, Index 5 = Rknee
    obs[:, 2] = knee_to_motor(obs[:, 2], obs[:, 1])  # Lknee → Lmotor
    obs[:, 5] = knee_to_motor(obs[:, 5], obs[:, 4])  # Rknee → Rmotor

    if env._leggy_obs_counter % 50 == 0:
        env_id = 0
        print(f"RELATIVE obs (motor space): {obs[env_id].cpu().numpy()}")
        print(f"  [LhipY, LhipX, Lmotor_rel, RhipY, RhipX, Rmotor_rel]")
        print(f"  Degrees: {[f'{x:.2f}°' for x in (obs[env_id].cpu().numpy() * 180 / 3.14159)]}")
        print(f"  Lmotor_rel={obs[env_id, 2].item() * 180/3.14159:.2f}° (should be Lknee_rel - LhipX_rel)")

    return obs


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "motor_to_knee",
    "knee_to_motor",
    "joint_pos_rel_motor",
    "LeggyJointAction",
    "LeggyJointActionCfg",
]
