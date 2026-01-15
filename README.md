# LeggySim

A robot simulation environment package for training biped robots using reinforcement learning. This project enables importing robots from Onshape, creating MuJoCo-based simulation environments with parallel execution, and preparing them for RL policy training with LeRobot.

## Overview

LeggySim bridges the gap between CAD design and RL training by providing:
- Automated robot import from Onshape CAD models
- MuJoCo-based physics simulation with [mjlab](https://github.com/mujocolab/mjlab) integration
- Parallel environment execution for efficient training
- [LeRobot envHub](https://huggingface.co/docs/lerobot/en/envhub) compatibility for seamless RL training

This package focuses on the **robot + simulation layer**. Training and policy deployment are handled separately using LeRobot.

## Features

### Current Features
- **Gymnasium Environment**: Full implementation of biped robot environment with:
  - IMU sensor readings (acceleration, gyroscope, orientation)
  - Joint position, velocity, and torque feedback
  - Configurable observation and action spaces
  - Customizable reward functions
- **Parallel Execution**: Run N synchronized or asynchronous environments
- **Visualization**: Browser-based multi-environment visualization using Viser
- **LeRobot Integration**: envHub-compatible environment factory

### Planned Features
- [ ] Command-line robot import from Onshape
- [ ] Scene description configuration
- [ ] Input/output vector definitions via config files
- [ ] Reward function templates and customization
- [ ] Environment parameter randomization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd leggySim

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `onshape-to-robot`: Robot URDF/XML generation from Onshape CAD
- `mjlab`: MuJoCo lab integration for robot simulation
- `lerobot`: LeRobot framework for RL training
- `viser`: 3D visualization in browser
- `huggingface_hub`: Environment distribution

Note: mjlab includes MuJoCo, Gymnasium, NumPy, and PyTorch as transitive dependencies.

## Usage

### 1. Import Robot from Onshape

```bash
# TODO: Command to import robot from Onshape
# onshape-to-robot <onshape-url> --output leggy/
```

This will generate the robot XML files in the `leggy/` directory that MuJoCo can load.

### 2. Run Simulation

#### Basic Test with N Parallel Environments

```bash
# Run with 4 parallel environments (default)
python env.py

# Run with custom number of environments
python env.py 8
```

This will:
- Create N parallel robot instances
- Run random actions on all robots
- Print height and reward metrics
- Continue until interrupted (Ctrl+C)

#### Visual Debugging

```bash
# Visualize single robot with MuJoCo viewer
python -m mujoco.viewer --mjcf=leggy/robot.xml

# Visualize multiple robots in browser (Viser)
python visualize_envs.py 4
```

Open http://localhost:8080 to see all robots running in parallel in your browser.

### 3. Integrate with LeRobot

```python
from env import make_env

# Create environment factory for LeRobot
envs_dict = make_env(n_envs=16, use_async_envs=True)

# Use with LeRobot training
# (See LeRobot documentation for full training setup)
```

The `make_env()` function returns a dictionary in the format expected by LeRobot envHub:
```python
{
    "suite_name": {
        task_id: VectorEnv
    }
}
```

## Project Structure

```
leggySim/
├── env.py                  # Main Gymnasium environment implementation
├── visualize_envs.py       # Viser-based multi-env visualization
├── requirements.txt        # Python dependencies
├── leggy/                  # Robot model directory
│   ├── robot.xml          # MuJoCo robot model (generated from Onshape)
│   └── config.json        # Robot configuration (optional)
└── README.md              # This file
```

## Environment Details

### Observation Space
The environment provides sensor readings typically available on real hardware:
- **IMU Acceleration** (3D): Linear acceleration from accelerometer
- **IMU Gyroscope** (3D): Angular velocity from gyroscope
- **IMU Orientation** (4D): Quaternion orientation
- **Joint Positions** (N): Encoder readings for each joint
- **Joint Velocities** (N): Joint angular velocities
- **Motor Torques** (N): Current torque output of each actuator
- **Previous Action** (N): Action from previous timestep

Total dimension: 3 + 3 + 4 + 4N (where N = number of joints)

### Action Space
- **Continuous**: Box space in [-1, 1] for each joint
- **Scaled**: Actions are automatically scaled to joint limits
- **Dimension**: N (number of actuated joints)

### Reward Function
Current reward encourages forward walking:
- Forward velocity (positive reward for moving forward)
- Height stability (exponential penalty for deviating from standing height)
- Energy efficiency (L2 penalty on motor torques)

Weights are tunable in `env.py:157`.

## Configuration

### Robot Parameters
Edit these in `env.py` to match your robot:
- `n_joints`: Number of actuated joints
- `initial_height`: Standing height in meters
- `min_height`: Fall detection threshold
- `initial_qpos`: Default joint positions for standing pose

### Simulation Parameters
- `dt`: Timestep (20ms by default)
- `frame_skip`: Physics steps per environment step
- `render_fps`: Rendering frequency

### Reward Tuning
Modify `_compute_reward()` in `env.py:157` to customize:
- Task objectives (walking, jumping, balancing, etc.)
- Penalty terms (energy, smoothness, joint limits, etc.)
- Weight coefficients

## Next Steps

### For This Repository
1. Implement Onshape import command
2. Add scene description system (obstacles, terrain, props)
3. Create config files for observation/action space definitions
4. Build reward function library and configuration system
5. Add domain randomization for sim-to-real transfer

### For Training (Separate Repository)
This environment package is designed to be imported into a training repository that will:
1. Load environments using `make_env()`
2. Configure RL algorithm (PPO, SAC, etc.)
3. Set up training loop with LeRobot
4. Handle model checkpointing and logging
5. Deploy policies on real hardware

## Contributing

This is a work in progress. Key areas needing development:
- Onshape import automation
- Scene/task configuration system
- Reward function templates
- Documentation and examples

## Resources

- [LeRobot envHub Documentation](https://huggingface.co/docs/lerobot/en/envhub)
- [mjlab GitHub](https://github.com/mujocolab/mjlab)
- [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

[Add your license here]
