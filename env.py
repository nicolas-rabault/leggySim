"""
LeggySim Environment - Biped robot learning to walk

Minimal Gymnasium environment for Leggy biped robot.

DEPRECATED: This file is kept for reference and future use.
For training, use the mjlab-based environment instead:
    uv run leggy-train Mjlab-Stand-up-Flat-Leggy --env.scene.num-envs 2048
"""

import gymnasium as gym
import mujoco
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class LeggyEnv(gym.Env):
    """Leggy biped robot environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # Load scene model (includes robot + ground)
        model_path = Path(__file__).parent / "leggy" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Rendering
        self.render_mode = render_mode
        self.renderer = None

        # Simulation
        self.dt = 0.02  # 20ms per step
        self.frame_skip = 10  # 10 physics steps per env step

        # Robot configuration
        # TODO: Adjust these for your robot
        self.n_joints = 6  # Number of actuated joints
        self.initial_height = 0.15  # Standing height in meters
        self.min_height = 0.08  # Termination height

        # Initial joint positions (standing pose)
        # TODO: Set from your config.json qpos0 values
        self.initial_qpos = np.array([
            0.10472, 0.10472, 0.523599,  # Left leg
            0.10472, 0.10472, 0.523599,  # Right leg
        ])

        # Observation and action spaces
        obs_dim = (
            3 +  # IMU acceleration
            3 +  # IMU gyroscope
            4 +  # IMU orientation
            self.n_joints +  # Joint positions
            self.n_joints +  # Joint velocities
            self.n_joints +  # Motor torques
            self.n_joints  # Previous action
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # State
        self.previous_action = np.zeros(self.n_joints, dtype=np.float32)
        self.prev_linvel = None
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose
        self.data.qpos[0:3] = [0.0, 0.0, self.initial_height]  # Position
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Orientation (upright)
        self.data.qpos[7:7+self.n_joints] = self.initial_qpos  # Joints

        mujoco.mj_forward(self.model, self.data)

        self.previous_action = np.zeros(self.n_joints, dtype=np.float32)
        self.prev_linvel = self.data.qvel[0:3].copy()
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Execute one step."""
        action = np.clip(action, -1.0, 1.0)

        # Scale action to joint limits
        scaled_action = self._scale_action(action)
        self.data.ctrl[:self.n_joints] = scaled_action

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Get results
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.data.qpos[2] < self.min_height  # Fell down
        truncated = self.step_count >= 1000  # Max steps

        self.previous_action = action
        self.step_count += 1

        info = {
            "height": float(self.data.qpos[2]),
            "forward_velocity": float(self.data.qvel[0]),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation from sensors."""
        # IMU acceleration (numerical derivative)
        current_vel = self.data.qvel[0:3]
        imu_accel = (current_vel - self.prev_linvel) / self.dt
        self.prev_linvel = current_vel.copy()

        # IMU gyroscope
        imu_gyro = self.data.qvel[3:6]

        # IMU orientation
        imu_quat = self.data.qpos[3:7]

        # Joint encoders
        joint_pos = self.data.qpos[7:7+self.n_joints]
        joint_vel = self.data.qvel[6:6+self.n_joints]

        # Motor torques
        motor_torques = self.data.actuator_force[:self.n_joints]

        # Concatenate
        obs = np.concatenate([
            imu_accel,
            imu_gyro,
            imu_quat,
            joint_pos,
            joint_vel,
            motor_torques,
            self.previous_action,
        ])

        return obs.astype(np.float32)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1,1] to joint limits."""
        limits = self.model.jnt_range[0:self.n_joints]
        low, high = limits[:, 0], limits[:, 1]
        return low + (action + 1.0) * 0.5 * (high - low)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Calculate reward."""
        # TODO: Tune these weights for your task
        forward_vel = self.data.qvel[0]
        height = self.data.qpos[2]
        energy = np.sum(np.square(self.data.ctrl[:self.n_joints]))

        reward = (
            1.0 * forward_vel +  # Move forward
            0.5 * np.exp(-10.0 * abs(height - self.initial_height)) -  # Stay upright
            0.001 * energy  # Minimize energy
        )

        return float(reward)

    def render(self):
        """Render environment."""
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "human":
            # Human rendering not supported in vectorized envs
            # Use: python -m mujoco.viewer --mjcf=leggy/robot.xml
            pass

    def close(self):
        """Clean up."""
        if self.renderer is not None:
            if hasattr(self.renderer, 'close'):
                self.renderer.close()
            self.renderer = None


def make_env(n_envs: int = 1, use_async_envs: bool = False):
    """
    Create vectorized environments for LeRobot.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Use async (parallel) or sync (sequential)

    Returns:
        dict: {suite_name: {task_id: VectorEnv}}
    """
    def _make():
        return LeggyEnv()

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    vec_env = env_cls([_make for _ in range(n_envs)])

    return {"leggy": {0: vec_env}}


def test_envhub():
    """Test make_env function using envHub pattern."""
    from lerobot.envs.utils import _load_module_from_path, _call_make_env, _normalize_hub_result

    print("Testing make_env with envHub utilities...\n")

    # Load this module
    module = _load_module_from_path("./env.py")

    # Test the make_env function
    print("Creating 2 environments...")
    result = _call_make_env(module, n_envs=2, use_async_envs=False)
    normalized = _normalize_hub_result(result)

    # Verify it works
    suite_name = next(iter(normalized))
    env = normalized[suite_name][0]

    print(f"✓ Environment created: {suite_name}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")

    # Test a few steps
    print(f"\nRunning 10 test steps...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step == 0:
            print(f"  Step {step}: obs shape={obs.shape}, reward shape={reward.shape if hasattr(reward, 'shape') else 'scalar'}")

    print(f"\n✓ All steps successful")
    print(f"  Final observation shape: {obs.shape}")
    print(f"  Rewards: {reward}")

    env.close()
    print("\n✅ envHub test passed!")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="LeggySim Environment")
    parser.add_argument("--envhub", action="store_true", help="Test make_env function with envHub utilities")
    parser.add_argument("n_envs", nargs="?", type=int, default=4, help="Number of environments for visualization")

    args = parser.parse_args()

    if args.envhub:
        test_envhub()
    else:
        # Original visualization code
        import time
        import uuid
        import rerun as rr

        n_envs = args.n_envs

        print(f"Visualizing {n_envs} robots in parallel")
        print("Press Ctrl+C to stop\n")

        # Start Rerun viewer with unique recording ID
        rr.init(f"LeggySim_{uuid.uuid4().hex[:8]}", spawn=True)

        # Create N environments with their renderers
        envs = []
        renderers = []
        cameras = []

        for i in range(n_envs):
            env = LeggyEnv()
            env.reset()
            envs.append(env)

            renderer = mujoco.Renderer(env.model, height=480, width=640)
            renderers.append(renderer)

            camera = mujoco.MjvCamera()
            camera.distance = 2.5
            camera.azimuth = 45
            camera.elevation = -25
            camera.lookat[:] = [0, 0, 0.15]
            cameras.append(camera)

        # Run simulation loop
        robot_actions = [np.zeros(6) for _ in range(n_envs)]
        step = 0

        try:
            while True:
                rr.set_time("step", sequence=step)

                for i, (env, renderer, camera) in enumerate(zip(envs, renderers, cameras)):
                    # Update action every 10 steps
                    if step % 10 == 0:
                        robot_actions[i] = env.action_space.sample()

                    # Step simulation
                    obs, reward, terminated, truncated, info = env.step(robot_actions[i])

                    # Render and log scene
                    renderer.update_scene(env.data, camera)
                    img = renderer.render()
                    rr.log(f"robot_{i}", rr.Image(img))

                    # Log metrics
                    rr.log(f"stats/robot_{i}/height", rr.Scalars(env.data.qpos[2]))
                    rr.log(f"stats/robot_{i}/velocity", rr.Scalars(env.data.qvel[0]))
                    rr.log(f"stats/robot_{i}/reward", rr.Scalars(reward))

                # Print status every 50 steps
                if step % 50 == 0:
                    print(f"Step {step}")
                    for i, env in enumerate(envs):
                        print(f"  Robot {i}: height={env.data.qpos[2]:.3f}m, vel={env.data.qvel[0]:+.2f}m/s")

                step += 1
                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\nStopped")

        for env in envs:
            env.close()
