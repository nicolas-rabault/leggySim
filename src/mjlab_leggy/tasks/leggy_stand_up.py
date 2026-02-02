"""Leggy stand up environment."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_leggy.leggy.leggy_constants import LEGGY_ROBOT_CFG
from mjlab_leggy.leggy.leggy_actions import (
    LeggyJointActionCfg,
    joint_pos_motor,
    joint_vel_motor,
    joint_torques_motor,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.entity import Entity


def foot_height_terrain_relative(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Foot height relative to terrain origin (terrain-invariant)."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z_world = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    terrain_z = env.scene.env_origins[:, 2].unsqueeze(1)
    return foot_z_world - terrain_z


def feet_clearance_terrain_relative(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Foot clearance reward - terrain-relative."""
    from mjlab.tasks.velocity.mdp.rewards import feet_clearance
    asset: Entity = env.scene[asset_cfg.name]

    # Compute terrain-relative foot heights
    foot_z_world = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    terrain_z = env.scene.env_origins[:, 2].unsqueeze(1)
    foot_z_relative = foot_z_world - terrain_z

    # Use the same logic as standard feet_clearance but with relative heights
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    delta = torch.abs(foot_z_relative - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class feet_swing_height_terrain_relative:
    """Swing height reward - terrain-relative."""

    def __init__(self, cfg, env):
        from mjlab.managers.manager_term_config import RewardTermCfg
        self.sensor_name = cfg.params["sensor_name"]
        self.site_names = cfg.params["asset_cfg"].site_names
        self.peak_heights = torch.zeros(
            (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
        )
        self.step_dt = env.step_dt

    def __call__(self, env, sensor_name: str, target_height: float, command_name: str,
                 command_threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        from mjlab.sensor import ContactSensor
        from mjlab.entity import Entity

        asset: Entity = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        # Terrain-relative heights
        foot_z_world = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        terrain_z = env.scene.env_origins[:, 2].unsqueeze(1)
        foot_heights = foot_z_world - terrain_z

        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
            num_landings, min=1
        )
        env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def leggy_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Leggy stand up environment configuration."""
    cfg = make_velocity_env_cfg()

    # Set control frequency to 100Hz (was 50Hz with decimation=4)                                                                                                                                               
    cfg.decimation = 2

    # Set mujoco sim parameters to improve stability and collision detection
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 45

    # Set leggy robot
    cfg.scene.entities = {"robot": LEGGY_ROBOT_CFG}

    # -------------------------------------------------------------------------
    # Custom Actions: Motor-to-knee conversion
    # -------------------------------------------------------------------------
    # LeggyJointAction converts motor commands to knee angles.
    # Passive joints are handled automatically by MuJoCo's constraint solver.
    cfg.actions = {
        "joint_pos": LeggyJointActionCfg()
    }

    # -------------------------------------------------------------------------
    # Custom Observations: Compute motor space positions from knee angles
    # -------------------------------------------------------------------------
    # Instead of observing passive motor joints, we compute motor space knee angles
    # directly from current knee positions using knee_to_motor conversion.
    # This allows MuJoCo to handle the passive joint loop automatically.

    cfg.observations["policy"].terms["joint_pos"] = ObservationTermCfg(
        func=joint_pos_motor
    )
    cfg.observations["critic"].terms["joint_pos"] = ObservationTermCfg(
        func=joint_pos_motor
    )

    # Also update joint velocities to use motor space computation
    cfg.observations["policy"].terms["joint_vel"] = ObservationTermCfg(
        func=joint_vel_motor
    )
    cfg.observations["critic"].terms["joint_vel"] = ObservationTermCfg(
        func=joint_vel_motor
    )

    # Add motor torques (measured at motor outputs via torque sensors)
    # For passiveMotor torques, we use the knee actuator forces since the knee
    # drives the motor through the parallel differential mechanism
    cfg.observations["policy"].terms["joint_torques"] = ObservationTermCfg(
        func=joint_torques_motor
    )
    cfg.observations["critic"].terms["joint_torques"] = ObservationTermCfg(
        func=joint_torques_motor
    )

    # -------------------------------------------------------------------------
    # Contact sensors
    # -------------------------------------------------------------------------
    # Foot geometry and site names from robot.xml
    foot_geom_names = ("left_foot_collision", "right_foot_collision")
    foot_site_names = ("left_foot", "right_foot")

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Note: Leggy only has foot collision geoms so we skip the nonfoot ground contact sensor
    cfg.scene.sensors = (feet_ground_cfg,)

    # Configure viewer (main body is called "boddy" in robot.xml)
    cfg.viewer.body_name = "boddy"

    # -------------------------------------------------------------------------
    # Velocity command ranges
    # -------------------------------------------------------------------------
    # During training, commands are sampled uniformly from these ranges.
    # 80% of environments receive zero velocity (standing still) for balance training.
    twist_cmd = cfg.commands["twist"]
    twist_cmd.viz.z_offset = 1.0
    twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)  # Yaw rate in rad/s - controls turning
    twist_cmd.ranges.lin_vel_y = (-0.8, 0.3)  # Forward velocity in m/s
    twist_cmd.ranges.lin_vel_x = (-0.2, 0.2)  # Lateral velocity in m/s - side-stepping
    twist_cmd.rel_standing_envs = 0.5  # Fraction with zero velocity command
    twist_cmd.rel_heading_envs = 0.5

    # -------------------------------------------------------------------------
    # Update asset references for Leggy's geometry
    # -------------------------------------------------------------------------
    # Foot height observation - terrain-relative for height-invariant behavior
    cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
        func=foot_height_terrain_relative,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names)},
    )

    # Foot geoms for friction randomization
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)  # min, max friction

    # Foot sites for foot_slip reward (clearance and swing_height are set below with custom functions)
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].params["asset_cfg"].site_names = foot_site_names

    # Body name for orientation rewards (main body is "boddy", not "trunclink")
    for reward_name in ["upright", "body_ang_vel"]:
        if reward_name in cfg.rewards and hasattr(cfg.rewards[reward_name].params.get("asset_cfg", None), "body_names"):
            cfg.rewards[reward_name].params["asset_cfg"].body_names = ("boddy",)

    # -------------------------------------------------------------------------
    # Pose reward configuration
    # -------------------------------------------------------------------------
    # Controls how tightly the robot should maintain its default joint pose.
    # Smaller std = tighter constraint (harsher penalty for deviation)
    # Larger std = more relaxed (allows greater deviation before penalty kicks in)
    # Values are in radians for joint angle deviations.
    if "pose" in cfg.rewards:
        # Limit to actuated joints only (exclude Lpassive*, Rpassive*)
        cfg.rewards["pose"].params["asset_cfg"].joint_names = (
            "LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"
        )
        # Standing mode: tight constraints for balance (robot mostly stationary)
        cfg.rewards["pose"].params["std_standing"] = {
            ".*hipY.*": 0.2,  # HipY needs tight control for balance
            ".*hipX.*": 0.4,  # HipX needs tight control for balance
            ".*knee.*": 0.4,  # Knee slightly more relaxed
        }
        # Walking mode: medium constraints for locomotion
        cfg.rewards["pose"].params["std_walking"] = {
            ".*hipY.*": 0.2,
            ".*hipX.*": 0.6,
            ".*knee.*": 0.8,
        }
        # Running mode: relaxed constraints for dynamic motion
        cfg.rewards["pose"].params["std_running"] = {
            ".*hipY.*": 0.4,
            ".*hipX.*": 1.0,
            ".*knee.*": 1.2,
        }

    # -------------------------------------------------------------------------
    # Reward weights
    # -------------------------------------------------------------------------
    # Each reward term is: weight * reward_fn(env) * dt
    # Positive weights encourage behavior, negative weights penalize it.

    # -- Velocity tracking --
    # Reward for matching commanded linear velocity (forward/backward, left/right)
    cfg.rewards["track_linear_velocity"].weight = 8.0
    # Reward for matching commanded angular velocity (turning rate)
    cfg.rewards["track_angular_velocity"].weight = 8.0

    # -- Pose and orientation --
    # Reward for keeping body upright (gravity aligned with body Z axis)
    cfg.rewards["upright"].weight = 1.0
    # Reward for staying close to default joint angles - prevents drift from home pose
    cfg.rewards["pose"].weight = 3.5

    # -- Energy efficiency --
    # Penalty for rapid action changes between timesteps - reduces jittery motion
    cfg.rewards["action_rate_l2"].weight = -2.0

    # -- Gait and foot behavior --
    # Use terrain-relative rewards for terrain-invariant behavior
    from mjlab.managers.manager_term_config import RewardTermCfg
    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=feet_clearance_terrain_relative,
        weight=1.0,
        params={
            "target_height": 0.03,
            "command_name": "twist",
            "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
        },
    )
    cfg.rewards["foot_swing_height"] = RewardTermCfg(
        func=feet_swing_height_terrain_relative,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
        },
    )
    # Air time tracking - encourages slower gait with feet spending time in air
    cfg.rewards["air_time"].weight = 6.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.05
    # Penalty for foot slipping on ground during contact
    cfg.rewards["foot_slip"].weight = -6.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0

    # -- Regularization --
    # Penalty for body angular velocity - reduces unwanted spinning/wobbling
    cfg.rewards["body_ang_vel"].weight = -0.05
    # Penalty for angular momentum - reduces unwanted spinning/wobbling
    cfg.rewards["angular_momentum"].weight = -0.02

    # -------------------------------------------------------------------------
    # Terrain configuration
    # -------------------------------------------------------------------------
    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_generator is not None
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.3)
    for sub_cfg in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        if hasattr(sub_cfg, "step_height_range"):
            sub_cfg.step_height_range = (0.0, 0.03)
        if hasattr(sub_cfg, "slope_range"):
            sub_cfg.slope_range = (0.0, 0.3)
        if hasattr(sub_cfg, "noise_range"):
            sub_cfg.noise_range = (0.01, 0.03)
        if hasattr(sub_cfg, "amplitude_range"):
            sub_cfg.amplitude_range = (0.0, 0.05)

    # Higher friction for better grip
    cfg.scene.terrain.friction = "1.5 0.005 0.0001"
    # Smaller timeconst (0.005) = stiffer contact, less bouncing
    cfg.scene.terrain.solref = "0.005 1"
    # Higher dmin/dmax (0.995, 0.9995) = less penetration, more precise collision
    cfg.scene.terrain.solimp = "0.995 0.9995 0.001 0.5 2"
    cfg.scene.terrain.contact = "enable"

    # Disable terrain curriculum
    # del cfg.curriculum["terrain_levels"]
    # del cfg.curriculum["command_vel"]

    # -------------------------------------------------------------------------
    # Observation noise and delay configuration (sim-to-real)
    # -------------------------------------------------------------------------
    # Add realistic sensor noise and delays
    cfg.observations["policy"].enable_corruption = True
    cfg.observations["policy"].corruption_std = 0.01

    # Configure sensor delays
    cfg.observations["policy"].terms["projected_gravity"] = deepcopy(
        cfg.observations["policy"].terms["projected_gravity"]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 2
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 4
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["policy"].terms["projected_gravity"].delay_min_lag = 2
    cfg.observations["policy"].terms["projected_gravity"].delay_max_lag = 4
    cfg.observations["policy"].terms["projected_gravity"].delay_update_period = 64

    cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)
    cfg.commands["twist"].ranges.lin_vel_x = (-0.3, 0.3)

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
    }

    # -------------------------------------------------------------------------
    # Termination conditions
    # -------------------------------------------------------------------------
    # Reset if body goes below the floor - prevents exploiting floor penetration
    cfg.terminations["body_below_floor"] = TerminationTermCfg(
        func=mdp_terminations.root_height_below_minimum,
        params={"minimum_height": 0.00},  # meters
    )

    # -------------------------------------------------------------------------
    # Play mode overrides
    # -------------------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)  # Effectively infinite episode length
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events.pop("foot_friction", None)

        # Disable terrain curriculum for play mode
        assert cfg.scene.terrain is not None
        assert cfg.scene.terrain.terrain_generator is not None
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0
        # Disable velocity commands
        # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)

    return cfg


def leggy_stand_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Leggy stand up task.

    Returns PPO training configuration with network architecture and hyperparameters.
    """
    return RslRlOnPolicyRunnerCfg(
        # ---------------------------------------------------------------------
        # Policy network architecture
        # ---------------------------------------------------------------------
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,  # Initial std for action noise during exploration
            actor_obs_normalization=True,  # Enable normalization for numerical stability
            critic_obs_normalization=True,  # Enable normalization for numerical stability
            actor_hidden_dims=(512, 256, 128),  # Actor (policy) network layers
            critic_hidden_dims=(512, 256, 128),  # Critic (value) network layers
            activation="elu",
        ),
        # ---------------------------------------------------------------------
        # PPO algorithm hyperparameters
        # ---------------------------------------------------------------------
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,  # Weight of value loss (higher = more emphasis on critic)
            use_clipped_value_loss=True,
            clip_param=0.2,  # Policy ratio clipping - limits update magnitude
            entropy_coef=0.01,  # Encourages exploration (higher = more random)
            num_learning_epochs=5,  # Epochs per batch of experience
            num_mini_batches=4,  # Mini-batches per epoch
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,  # Discount factor (higher = more farsighted)
            lam=0.95,  # GAE lambda (higher = less bias, more variance)
            desired_kl=0.01,  # Target KL for adaptive learning rate
            max_grad_norm=1.0,  # Gradient clipping (prevents exploding gradients)
        ),
        # ---------------------------------------------------------------------
        # Training parameters
        # ---------------------------------------------------------------------
        experiment_name="leggy_stand_up",
        save_interval=500,  # Checkpoint saving interval (iterations)
        num_steps_per_env=12,  # Simulation steps before each policy update
        max_iterations=50_000,
    )
