"""Leggy stand up environment"""

import numpy as np

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


# Leggy foot collision geom names
FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
FOOT_SITE_NAMES = ("left_foot", "right_foot")

# =============================================================================
# REWARD WEIGHTS
# =============================================================================
# These coefficients scale each reward term's contribution to the total reward.
# Positive weights encourage the behavior, negative weights penalize it.
# The final reward is: sum(weight_i * reward_fn_i(env) * dt)

# -- Velocity tracking rewards --
# Reward for tracking commanded linear velocity in the XY plane
REWARD_TRACK_LIN_VEL_XY_EXP_WEIGHT = 1.5
# Reward for tracking commanded angular velocity around Z axis
REWARD_TRACK_ANG_VEL_Z_EXP_WEIGHT = 0.75

# -- Stability penalties --
# Penalty for vertical (Z) linear velocity - discourages bouncing/jumping
REWARD_LIN_VEL_Z_L2_WEIGHT = -2.0
# Penalty for angular velocity in XY plane - discourages rolling/pitching
REWARD_ANG_VEL_XY_L2_WEIGHT = -0.05

# -- Pose and orientation rewards --
# Reward for maintaining upright orientation (gravity aligned with body Z axis)
REWARD_UPRIGHT_WEIGHT = 0.2
# Penalty for non-flat orientation (body tilting away from horizontal)
REWARD_FLAT_ORIENTATION_WEIGHT = -1.0
# Reward for maintaining the default joint pose - prevents drift from home position
REWARD_POSE_WEIGHT = 0.5
# Reward for maintaining target base height above ground
REWARD_BASE_HEIGHT_WEIGHT = 0.2

# -- Energy efficiency penalties --
# Penalty for high joint torques - encourages energy-efficient motions
REWARD_JOINT_TORQUES_L2_WEIGHT = -0.0001
# Penalty for high joint accelerations - encourages smooth motions
REWARD_JOINT_ACC_L2_WEIGHT = -2.5e-7
# Penalty for high action rate (change between timesteps) - reduces jitter
REWARD_ACTION_RATE_L2_WEIGHT = -0.01

# -- Gait and foot rewards --
# Reward for feet achieving clearance during swing phase - promotes proper stepping
REWARD_FOOT_CLEARANCE_WEIGHT = 0.5
# Reward for achieving minimum swing height - ensures feet lift properly
REWARD_FOOT_SWING_HEIGHT_WEIGHT = 0.1
# Penalty for foot slip (foot velocity while in contact) - encourages stable contacts
REWARD_FOOT_SLIP_WEIGHT = -0.1
# Penalty for feet air time deviation from target - encourages rhythmic gait
REWARD_FEET_AIR_TIME_WEIGHT = 0.5

# -- Regularization penalties --
# Penalty for body angular velocity magnitude - reduces spinning
REWARD_BODY_ANG_VEL_WEIGHT = -0.05
# Penalty for joint velocity magnitude - reduces jerky movements
REWARD_JOINT_VEL_L2_WEIGHT = -0.0001

# =============================================================================
# POSE REWARD STANDARD DEVIATIONS
# =============================================================================
# These control how tightly the robot should maintain its default joint pose.
# Smaller std = tighter constraint (harsher penalty for deviation)
# Larger std = more relaxed (allows greater deviation before penalty kicks in)
# Values are in radians for joint angle deviations.

# Standing mode: tight constraints for balance (robot mostly stationary)
POSE_STD_STANDING_HIP = 0.15   # Hip joints need tight control for balance
POSE_STD_STANDING_KNEE = 0.2  # Knee joints slightly more relaxed

# Walking mode: medium constraints for locomotion
POSE_STD_WALKING_HIP = 0.3
POSE_STD_WALKING_KNEE = 0.4

# Running mode: relaxed constraints for dynamic motion
POSE_STD_RUNNING_HIP = 0.5
POSE_STD_RUNNING_KNEE = 0.6

# =============================================================================
# COMMAND RANGES
# =============================================================================
# Velocity command ranges for the twist (velocity) command generator.
# During training, commands are sampled uniformly from these ranges.

# Angular velocity around Z axis (yaw rate) in rad/s - controls turning
CMD_ANG_VEL_Z_RANGE = (-0.2, 0.2)
# Lateral (Y) linear velocity in m/s - controls side-stepping
CMD_LIN_VEL_Y_RANGE = (-0.1, 0.1)
# Forward (X) linear velocity in m/s - controls forward/backward motion
CMD_LIN_VEL_X_RANGE = (-0.1, 0.1)
# Fraction of environments that receive zero velocity command (standing still)
CMD_REL_STANDING_ENVS = 0.8

# =============================================================================
# PPO ALGORITHM COEFFICIENTS
# =============================================================================
# Proximal Policy Optimization hyperparameters for training

# Weight of value function loss in total loss (higher = more emphasis on critic)
PPO_VALUE_LOSS_COEF = 1.0
# Clipping parameter for policy ratio - limits policy update magnitude
PPO_CLIP_PARAM = 0.2
# Entropy coefficient - encourages exploration (higher = more random actions)
PPO_ENTROPY_COEF = 0.01
# Number of epochs to train on each batch of experience
PPO_NUM_LEARNING_EPOCHS = 5
# Number of mini-batches to split experience into per epoch
PPO_NUM_MINI_BATCHES = 4
# Learning rate for optimizer
PPO_LEARNING_RATE = 1.0e-3
# Discount factor for future rewards (higher = more farsighted)
PPO_GAMMA = 0.99
# GAE lambda for advantage estimation (higher = less bias, more variance)
PPO_LAM = 0.95
# Target KL divergence for adaptive learning rate schedule
PPO_DESIRED_KL = 0.01
# Maximum gradient norm for clipping (prevents exploding gradients)
PPO_MAX_GRAD_NORM = 1.0

# =============================================================================
# POLICY NETWORK ARCHITECTURE
# =============================================================================
# Initial standard deviation for action noise during exploration
POLICY_INIT_NOISE_STD = 1.0
# Hidden layer dimensions for actor (policy) network
POLICY_ACTOR_HIDDEN_DIMS = (512, 256, 128)
# Hidden layer dimensions for critic (value) network
POLICY_CRITIC_HIDDEN_DIMS = (512, 256, 128)

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
# Number of simulation steps per environment before each policy update
TRAIN_NUM_STEPS_PER_ENV = 24
# Interval (in iterations) for saving model checkpoints
TRAIN_SAVE_INTERVAL = 500
# Maximum training iterations
TRAIN_MAX_ITERATIONS = 50_000

# =============================================================================
# TERMINATION THRESHOLDS
# =============================================================================
# Minimum body height before episode terminates (prevents floor exploitation)
TERMINATION_MIN_BODY_HEIGHT = 0.02


def leggy_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy stand up environment configuration."""
    cfg = make_velocity_env_cfg()

    # Set leggy robot
    cfg.scene.entities = {"robot": LEGGY_ROBOT_CFG}

    # Add contact sensors for feet
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Note: Leggy only has foot collision geoms (left_foot_collision, right_foot_collision)
    # so we skip the nonfoot ground contact sensor
    cfg.scene.sensors = (feet_ground_cfg,)

    # Configure viewer (main body is called "boddy" in robot.xml)
    cfg.viewer.body_name = "boddy"

    # Configure action offsets to match initial joint positions from HOME_FRAME
    # This ensures action=0 targets the initial standing pose, not joint position 0
    # (The XML joints don't have 'ref' attributes, so use_default_offset would use 0)
    # cfg.actions["joint_pos"].use_default_offset = False
    # cfg.actions["joint_pos"].offset = {
    #     ".*hipY": 6 * np.pi / 180.0,
    #     ".*hipX": 30 * np.pi / 180.0,
    #     ".*knee": 30 * np.pi / 180.0,
    # }

    # Configure velocity command ranges (mostly standing for balance training)
    twist_cmd = cfg.commands["twist"]
    twist_cmd.viz.z_offset = 1.0
    twist_cmd.ranges.ang_vel_z = CMD_ANG_VEL_Z_RANGE
    twist_cmd.ranges.lin_vel_y = CMD_LIN_VEL_Y_RANGE
    twist_cmd.ranges.lin_vel_x = CMD_LIN_VEL_X_RANGE
    twist_cmd.rel_standing_envs = CMD_REL_STANDING_ENVS
    twist_cmd.rel_heading_envs = 0.0

    # Update observations that reference foot sites
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Update events that reference foot geoms
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES

    # Update rewards that reference foot sites
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Update rewards that reference body names (main body is "boddy")
    # The default velocity config uses "trunclink" which doesn't exist in Leggy
    for reward_name in ["upright", "body_ang_vel", "flat_orientation", "base_height", "lin_vel_z_l2"]:
        if reward_name in cfg.rewards and hasattr(cfg.rewards[reward_name].params.get("asset_cfg", None), "body_names"):
            cfg.rewards[reward_name].params["asset_cfg"].body_names = ("boddy",)

    # Configure the pose reward with joint-specific std values for Leggy
    # Only use actuated joints (exclude passive joints which aren't observable on real robot)
    if "pose" in cfg.rewards:
        # Limit to actuated joints only (exclude Lpassive*, Rpassive*)
        cfg.rewards["pose"].params["asset_cfg"].joint_names = (
            "LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"
        )
        cfg.rewards["pose"].params["std_standing"] = {
            ".*hip.*": POSE_STD_STANDING_HIP,
            ".*knee.*": POSE_STD_STANDING_KNEE,
        }
        cfg.rewards["pose"].params["std_walking"] = {
            ".*hip.*": POSE_STD_WALKING_HIP,
            ".*knee.*": POSE_STD_WALKING_KNEE,
        }
        cfg.rewards["pose"].params["std_running"] = {
            ".*hip.*": POSE_STD_RUNNING_HIP,
            ".*knee.*": POSE_STD_RUNNING_KNEE,
        }

    # -------------------------------------------------------------------------
    # Configure reward weights
    # -------------------------------------------------------------------------
    # Velocity tracking rewards
    if "track_lin_vel_xy_exp" in cfg.rewards:
        cfg.rewards["track_lin_vel_xy_exp"].weight = REWARD_TRACK_LIN_VEL_XY_EXP_WEIGHT
    if "track_ang_vel_z_exp" in cfg.rewards:
        cfg.rewards["track_ang_vel_z_exp"].weight = REWARD_TRACK_ANG_VEL_Z_EXP_WEIGHT

    # Stability penalties
    if "lin_vel_z_l2" in cfg.rewards:
        cfg.rewards["lin_vel_z_l2"].weight = REWARD_LIN_VEL_Z_L2_WEIGHT
    if "ang_vel_xy_l2" in cfg.rewards:
        cfg.rewards["ang_vel_xy_l2"].weight = REWARD_ANG_VEL_XY_L2_WEIGHT

    # Pose and orientation rewards
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = REWARD_UPRIGHT_WEIGHT
    if "flat_orientation" in cfg.rewards:
        cfg.rewards["flat_orientation"].weight = REWARD_FLAT_ORIENTATION_WEIGHT
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].weight = REWARD_POSE_WEIGHT
    if "base_height" in cfg.rewards:
        cfg.rewards["base_height"].weight = REWARD_BASE_HEIGHT_WEIGHT

    # Energy efficiency penalties
    if "joint_torques_l2" in cfg.rewards:
        cfg.rewards["joint_torques_l2"].weight = REWARD_JOINT_TORQUES_L2_WEIGHT
    if "joint_acc_l2" in cfg.rewards:
        cfg.rewards["joint_acc_l2"].weight = REWARD_JOINT_ACC_L2_WEIGHT
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = REWARD_ACTION_RATE_L2_WEIGHT

    # Gait and foot rewards
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = REWARD_FOOT_CLEARANCE_WEIGHT
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = REWARD_FOOT_SWING_HEIGHT_WEIGHT
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = REWARD_FOOT_SLIP_WEIGHT
    if "feet_air_time" in cfg.rewards:
        cfg.rewards["feet_air_time"].weight = REWARD_FEET_AIR_TIME_WEIGHT

    # Regularization penalties
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = REWARD_BODY_ANG_VEL_WEIGHT
    if "joint_vel_l2" in cfg.rewards:
        cfg.rewards["joint_vel_l2"].weight = REWARD_JOINT_VEL_L2_WEIGHT

    # Walking on plane only
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum (if present)
    if cfg.curriculum is not None and "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]

    # Add termination condition: reset if body goes below the floor
    # This prevents the policy from learning to exploit floor penetration
    cfg.terminations["body_below_floor"] = TerminationTermCfg(
        func=mdp_terminations.root_height_below_minimum,
        params={"minimum_height": TERMINATION_MIN_BODY_HEIGHT},
    )

    # Apply play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)

        # Disable observation corruption and push events for play mode
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def leggy_stand_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Leggy stand up task."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=POLICY_INIT_NOISE_STD,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=POLICY_ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=POLICY_CRITIC_HIDDEN_DIMS,
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=PPO_VALUE_LOSS_COEF,
            use_clipped_value_loss=True,
            clip_param=PPO_CLIP_PARAM,
            entropy_coef=PPO_ENTROPY_COEF,
            num_learning_epochs=PPO_NUM_LEARNING_EPOCHS,
            num_mini_batches=PPO_NUM_MINI_BATCHES,
            learning_rate=PPO_LEARNING_RATE,
            schedule="adaptive",
            gamma=PPO_GAMMA,
            lam=PPO_LAM,
            desired_kl=PPO_DESIRED_KL,
            max_grad_norm=PPO_MAX_GRAD_NORM,
        ),
        experiment_name="leggy_stand_up",
        save_interval=TRAIN_SAVE_INTERVAL,
        num_steps_per_env=TRAIN_NUM_STEPS_PER_ENV,
        max_iterations=TRAIN_MAX_ITERATIONS,
    )
