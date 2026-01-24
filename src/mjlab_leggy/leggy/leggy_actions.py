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
# those 2 functions are validated by there usage in leggy_constants.py
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
       - Applies per-joint scale and offset transformations
       - Converts Lmotor → Lknee and Rmotor → Rknee using target hipX values
       - Stores result as [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]

    2. **apply_actions()** - Called every physics step (decimation times):
       - Sets actuated joint targets directly from processed actions
       - Reads current knee and hipX positions from simulator state
       - Updates passive joint positions to match current knee angles (mechanical coupling)

    This approach:
    - Avoids global MuJoCo callbacks (no race conditions)
    - Works correctly with parallel environments
    - Maintains per-environment state properly
    - Computes motor-to-knee conversion only once per step (efficient)
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
        # NOTE: Order must match the XML order (Lpassive2 comes before LpassiveMotor in the kinematic tree)
        passive_names = ["Lpassive2", "LpassiveMotor", "Rpassive2", "RpassiveMotor"]
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
        # Read joint limits from the asset
        # Shape: [num_envs, num_joints, 2] where last dim is [min, max]
        joint_limits_all = self._asset.data.joint_pos_limits
        joint_limits_actuated = joint_limits_all[:, self._actuated_joint_ids, :]  # [num_envs, 6, 2]

        # Compute per-joint scale factors based on usable range
        # Scale = (max - min) * soft_limit_factor / 2
        # This maps policy actions from [-1, 1] to the usable joint range
        self._soft_limit_factor = cfg.soft_joint_pos_limit_factor
        joint_ranges = (joint_limits_actuated[:, :, 1] - joint_limits_actuated[:, :, 0]) * self._soft_limit_factor
        self._scale = joint_ranges / 2.0  # [num_envs, 6]

        # Override with global scale if specified (for backward compatibility)
        if cfg.scale != 0.5:  # If user explicitly set a non-default scale
            self._scale = torch.full_like(self._scale, cfg.scale)

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
        # Debug counters (set to 0 to disable debug prints)
        # =====================================================================
        self._step_count = 0
        self._debug_interval = 0  # Set to >0 to enable debug prints (e.g., 50 for every 50 steps, 0 to disable)

        # Print scale factors once for debugging
        if self._debug_interval > 0:
            env_id = 0
            print(f"\n========== LeggyJointAction Initialization ==========")
            print(f"Per-joint scale factors (Env {env_id}):")
            joint_names = ["LhipY", "LhipX", "Lmotor", "RhipY", "RhipX", "Rmotor"]
            for i, name in enumerate(joint_names):
                print(f"  {name}: {self._scale[env_id, i].item():.6f} rad = {self._scale[env_id, i].item() * 180/3.14159:.2f}°")
            print(f"Soft limit factor: {self._soft_limit_factor}")
            print(f"Action mapping: policy_output [-1, 1] → joint_target [default - scale, default + scale]")

    @property
    def action_dim(self) -> int:
        """Policy outputs 6 actions: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]."""
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        """Raw actions from policy before any processing."""
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Preprocess actions: apply scale, offset, and motor-to-knee conversion.

        This is called ONCE per environment step (before decimation loop).
        Converts motor commands to knee commands using target hipX values.

        Args:
            actions: Raw actions from policy with shape [num_envs, 6]
                     Columns: [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]
        """
        self._raw_actions[:] = actions

        # Apply scale and offset to get target joint positions
        # Note: Still in "motor space" for knee joints at indices 2 and 5
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset

        # Convert motor commands to knee commands (indices 2 and 5)
        # Use target hipX values (indices 1 and 4) for the conversion
        # After this, _processed_actions contains: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
        hipX_left_target = self._processed_actions[:, 1]
        hipX_right_target = self._processed_actions[:, 4]
        self._processed_actions[:, 2] = motor_to_knee(self._processed_actions[:, 2], hipX_left_target)
        self._processed_actions[:, 5] = motor_to_knee(self._processed_actions[:, 5], hipX_right_target)

        # Debug printing
        self._step_count += 1
        if self._debug_interval > 0 and (self._step_count <= 5 or self._step_count % self._debug_interval == 0):
            env_id = 0
            print(f"\n========== Step {self._step_count} (Env {env_id}) process_actions ==========")
            print(f"Raw actions: {self._raw_actions[env_id].cpu().numpy()}")
            print(f"  [LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor]")
            print(f"Processed actions (knee space): {self._processed_actions[env_id].cpu().numpy()}")
            print(f"  [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]")
            print(f"  Degrees: {[f'{x:.2f}°' for x in (self._processed_actions[env_id].cpu().numpy() * 180 / 3.14159)]}")

    def apply_actions(self) -> None:
        """Apply actions to actuators and update passive joints.

        This is called EVERY physics step (decimation times per environment step).

        Processing pipeline:
        1. Set actuated joint targets: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
           (already computed in process_actions, stored in _processed_actions)
        2. Read CURRENT knee positions from simulator state
        3. Set passive joint targets to match CURRENT knee positions (mechanical coupling)
        """
        # =====================================================================
        # Step 1: Set actuated joint targets
        # =====================================================================
        # _processed_actions already contains: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
        # Motor-to-knee conversion was done once in process_actions()
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._actuated_joint_ids,
        )

        # =====================================================================
        # Step 2: Update passive joints based on CURRENT knee positions
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
            knee_left_current,   # Lpassive2 follows current Lknee (index 0)
            knee_to_motor(knee_left_current, hipX_left_current),   # LpassiveMotor follows current Lknee (index 1)
            knee_right_current,  # Rpassive2 follows current Rknee (index 2)
            knee_to_motor(knee_right_current, hipX_right_current),  # RpassiveMotor follows current Rknee (index 3)
        ], dim=1)

        # Directly write passive joint positions to simulation (not targets)
        # This is necessary because passive joints don't have actuators
        self._asset.write_joint_position_to_sim(
            passive_positions,
            joint_ids=self._passive_joint_ids,
        )

        # Debug printing (only on the first decimation step of each debug interval)
        if self._debug_interval > 0 and (self._step_count <= 5 or self._step_count % self._debug_interval == 0):
            env_id = 0
            print(f"\n========== Step {self._step_count} (Env {env_id}) apply_actions ==========")

            # Show actuated joint info
            print(f"\n--- ACTUATED JOINTS ---")
            print(f"Actuated joint IDs: {self._actuated_joint_ids}")
            print(f"Actuated joint names: {['LhipY', 'LhipX', 'Lknee', 'RhipY', 'RhipX', 'Rknee']}")
            print(f"TARGET positions:  {self._processed_actions[env_id].cpu().numpy()}")
            print(f"CURRENT positions: {current_joint_pos[env_id].cpu().numpy()}")
            print(f"\nDegrees:")
            print(f"  TARGET:  {[f'{x:.2f}°' for x in (self._processed_actions[env_id].cpu().numpy() * 180 / 3.14159)]}")
            print(f"  CURRENT: {[f'{x:.2f}°' for x in (current_joint_pos[env_id].cpu().numpy() * 180 / 3.14159)]}")

            # Show passive joint computation details
            print(f"\n--- PASSIVE JOINT COMPUTATION ---")
            print(f"LEFT LEG:")
            print(f"  Current LhipX: {hipX_left_current[env_id].item():.6f} rad = {hipX_left_current[env_id].item() * 180/3.14159:.2f}°")
            print(f"  Current Lknee: {knee_left_current[env_id].item():.6f} rad = {knee_left_current[env_id].item() * 180/3.14159:.2f}°")
            print(f"  → Lpassive2 = knee = {passive_positions[env_id, 0].item():.6f} rad ({passive_positions[env_id, 0].item() * 180/3.14159:.2f}°)")
            print(f"  → LpassiveMotor = knee - hipX = {knee_left_current[env_id].item():.6f} - {hipX_left_current[env_id].item():.6f} = {passive_positions[env_id, 1].item():.6f} rad ({passive_positions[env_id, 1].item() * 180/3.14159:.2f}°)")

            print(f"\nRIGHT LEG:")
            print(f"  Current RhipX: {hipX_right_current[env_id].item():.6f} rad = {hipX_right_current[env_id].item() * 180/3.14159:.2f}°")
            print(f"  Current Rknee: {knee_right_current[env_id].item():.6f} rad = {knee_right_current[env_id].item() * 180/3.14159:.2f}°")
            print(f"  → Rpassive2 = knee = {passive_positions[env_id, 2].item():.6f} rad ({passive_positions[env_id, 2].item() * 180/3.14159:.2f}°)")
            print(f"  → RpassiveMotor = knee - hipX = {knee_right_current[env_id].item():.6f} - {hipX_right_current[env_id].item():.6f} = {passive_positions[env_id, 3].item():.6f} rad ({passive_positions[env_id, 3].item() * 180/3.14159:.2f}°)")

            # Show passive joint mapping
            print(f"\n--- PASSIVE JOINT MAPPING TO MUJOCO ---")
            print(f"Passive joint IDs: {self._passive_joint_ids}")
            print(f"Passive joint names (XML order): {['Lpassive2', 'LpassiveMotor', 'Rpassive2', 'RpassiveMotor']}")
            print(f"Values being written: {passive_positions[env_id].cpu().numpy()}")
            print(f"Values in degrees: {[f'{x:.2f}°' for x in (passive_positions[env_id].cpu().numpy() * 180 / 3.14159)]}")
            print(f"\nMapping:")
            for i, (joint_id, name) in enumerate(zip(self._passive_joint_ids, ['Lpassive2', 'LpassiveMotor', 'Rpassive2', 'RpassiveMotor'])):
                print(f"  passive_positions[{i}] = {passive_positions[env_id, i].item():.6f} rad ({passive_positions[env_id, i].item() * 180/3.14159:.2f}°) → MuJoCo joint_id {joint_id} ({name})")


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
        scale: Global action scaling factor (default: 0.5). If set to 0.5 (default),
               per-joint scaling based on joint limits is used. Otherwise, applies
               uniform scaling to all joints for backward compatibility.
        soft_joint_pos_limit_factor: Fraction of joint range to use (default: 0.9)
        use_default_offset: If True, add default joint positions as offset (default: True)
    """
    class_type: type[ActionTerm] = LeggyJointAction
    asset_name: str = "robot"
    scale: float = 0.5
    soft_joint_pos_limit_factor: float = 0.9
    use_default_offset: bool = True


# =============================================================================
# Observation Helper Functions
# =============================================================================

def base_lin_acc(env, asset_cfg=None) -> torch.Tensor:
    """Get linear acceleration from IMU (accelerometer reading).

    This matches what a real IMU accelerometer measures, including gravity.
    We compute acceleration by finite difference of velocity and add gravity.

    Args:
        env: The environment instance
        asset_cfg: Asset configuration (default: "robot")

    Returns:
        Linear acceleration in body frame [num_envs, 3] (m/s²)
    """
    if asset_cfg is None:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")

    asset = env.scene[asset_cfg.name]

    # Get current velocity in body frame
    lin_vel_b = asset.data.root_link_lin_vel_b

    # Store previous velocity for finite difference (per-environment state)
    if not hasattr(env, '_prev_lin_vel_b'):
        env._prev_lin_vel_b = torch.zeros_like(lin_vel_b)

    # Compute acceleration by finite difference: a = (v - v_prev) / dt
    dt = env.step_dt
    lin_acc_b = (lin_vel_b - env._prev_lin_vel_b) / dt
    env._prev_lin_vel_b = lin_vel_b.clone()

    # Add gravity in body frame (what accelerometer measures)
    # projected_gravity_b is gravity rotated to body frame (points down in body frame)
    gravity_b = asset.data.projected_gravity_b

    # Accelerometer reading = computed_acc - gravity_b (since projected_gravity_b points up relative to body)
    # Actually, projected_gravity_b gives the direction, so we multiply by magnitude
    accelerometer_reading = lin_acc_b - gravity_b * 9.81

    return accelerometer_reading


def base_ang_pos(env, asset_cfg=None) -> torch.Tensor:
    """Get orientation as Euler angles (roll, pitch, yaw) from IMU sensor fusion.

    This matches what a real IMU outputs after sensor fusion (gyro + accel + mag).
    We compute Euler angles from the projected gravity and angular velocity.
    Euler angles are in XYZ (roll-pitch-yaw) convention.

    Args:
        env: The environment instance
        asset_cfg: Asset configuration (default: "robot")

    Returns:
        Orientation as Euler angles [num_envs, 3] (rad)
        Order: [roll, pitch, yaw]
    """
    if asset_cfg is None:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")

    asset = env.scene[asset_cfg.name]

    # Get projected gravity (normalized gravity vector in body frame)
    gravity_b = asset.data.projected_gravity_b

    # Compute roll and pitch from gravity vector
    # When robot is upright, gravity points to [0, 0, -1] in body frame
    # roll = atan2(gy, gz)
    # pitch = atan2(-gx, sqrt(gy^2 + gz^2))

    roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])
    pitch = torch.atan2(gravity_b[:, 0], torch.sqrt(gravity_b[:, 1]**2 + gravity_b[:, 2]**2))

    # Yaw cannot be determined from gravity alone - would need magnetometer
    # For now, return zero (or could integrate angular velocity)
    yaw = torch.zeros_like(roll)

    euler_angles = torch.stack([roll, pitch, yaw], dim=1)

    return euler_angles


def joint_torques_motor(env, asset_cfg=None) -> torch.Tensor:
    """Get actuator torques in motor-space representation.

    Since Leggy has torque sensors at the motor outputs, we want to observe
    torques as the robot would measure them:
    - Hip torques: directly from hip actuators
    - Motor torques: from knee actuators (which drive the motors through the mechanism)

    Args:
        env: The environment instance
        asset_cfg: Asset configuration (not used, kept for API compatibility)

    Returns:
        Actuator torques [num_envs, 6]
        Layout: [LhipY_torque, LhipX_torque, Lmotor_torque, RhipY_torque, RhipX_torque, Rmotor_torque]
    """
    # Get the robot asset
    asset = env.scene["robot"]

    # Find actuator indices in the order: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
    actuator_names = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
    actuator_ids = asset.find_actuators(actuator_names)[0]

    # Read actuator forces (torques) from simulation
    # data.actuator_force contains the force/torque each actuator is producing
    actuator_torques = asset.data.actuator_force[:, actuator_ids]

    # The layout matches our observation joints perfectly:
    # - Index 0: LhipY actuator → LhipY torque
    # - Index 1: LhipX actuator → LhipX torque
    # - Index 2: Lknee actuator → Lmotor torque (knee drives motor through mechanism)
    # - Index 3: RhipY actuator → RhipY torque
    # - Index 4: RhipX actuator → RhipX torque
    # - Index 5: Rknee actuator → Rmotor torque (knee drives motor through mechanism)

    return actuator_torques


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "motor_to_knee",
    "knee_to_motor",
    "base_lin_acc",
    "base_ang_pos",
    "joint_torques_motor",
    "LeggyJointAction",
    "LeggyJointActionCfg",
]
