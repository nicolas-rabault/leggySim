# LeggySim

A robot simulation environment package for training biped robots using reinforcement learning. This project enables importing robots from Onshape, creating MuJoCo-based simulation environments with parallel execution, and training RL policies with [mjlab](https://github.com/mujocolab/mjlab).

## Overview

LeggySim bridges the gap between CAD design and RL training by providing:
- Automated robot import from Onshape CAD models
- MuJoCo-based physics simulation with mjlab integration
- Parallel environment execution for efficient training
- Ready-to-use training tasks for the Leggy biped robot

## Features

### Current Features
- **Stand-up Task**: Train Leggy to stand up from various initial positions
- **Gymnasium Environment**: Full implementation of biped robot environment with:
  - IMU sensor readings (acceleration, gyroscope, orientation)
  - Joint position, velocity, and torque feedback
  - Configurable observation and action spaces
  - Customizable reward functions
- **Parallel Execution**: Run thousands of synchronized environments for efficient training
- **mjlab Integration**: Seamless integration with mjlab's training infrastructure

### Planned Features
- [ ] Command-line robot import from Onshape
- [ ] Scene description configuration
- [ ] Additional locomotion tasks (walking, jumping)
- [ ] Environment parameter randomization for sim-to-real transfer

## Installation

### Using pip (recommended for development)

```bash
# Clone the repository
git clone <repository-url>
cd leggySim

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install with visualization support
pip install -e ".[visualization]"

# Install with all development dependencies
pip install -e ".[dev]"
```

### Using uv

```bash
# Clone the repository
git clone <repository-url>
cd leggySim

# Install and sync dependencies
uv sync
```

### Dependencies

Core dependency:
- `mjlab`: MuJoCo lab integration for robot simulation (includes MuJoCo, Gymnasium, NumPy, and PyTorch)

Optional dependencies:
- `rerun-sdk`: 3D visualization
- `onshape-to-robot`: Robot URDF/XML generation from Onshape CAD (development only)

## Usage

### Training

Run training with mjlab using the `leggy-train` command:

```bash
# Using uv (recommended)
uv run leggy-train Mjlab-Stand-up-Flat-Leggy --env.scene.num-envs 2048

# Or if installed via pip
leggy-train Mjlab-Stand-up-Flat-Leggy --env.scene.num-envs 2048
```

#### Training Options

```bash
# Adjust number of parallel environments
uv run leggy-train Mjlab-Stand-up-Flat-Leggy --env.scene.num-envs 4096

# See all available options
uv run leggy-train --help
```

### Import Robot from Onshape

```bash
# TODO: Command to import robot from Onshape
# onshape-to-robot <onshape-url> --output leggy/
```

This will generate the robot XML files in the `leggy/` directory that MuJoCo can load.

### Visual Debugging

```bash
# Visualize single robot with MuJoCo viewer
python -m mujoco.viewer --mjcf=src/mjlab_leggy/leggy/robot.xml
```

## Project Structure

```
leggySim/
├── pyproject.toml              # Package configuration and dependencies
├── requirements.txt            # Legacy requirements (for reference)
├── env.py                      # Deprecated standalone environment (kept for reference)
├── infer_policy.py             # Policy inference script
└── src/
    └── mjlab_leggy/            # Main package
        ├── __init__.py
        ├── leggy/              # Robot model directory
        │   ├── robot.xml       # MuJoCo robot model
        │   ├── scene.xml       # Scene with robot + ground
        │   ├── sensors.xml     # Sensor definitions
        │   ├── config.json     # Robot configuration
        │   ├── leggy_constants.py  # Robot constants
        │   └── assets/         # Robot mesh files (STL)
        ├── scripts/
        │   └── train.py        # Training entry point
        └── tasks/
            └── leggy_stand_up.py  # Stand-up task definition
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

## Available Tasks

| Task Name | Description |
|-----------|-------------|
| `Mjlab-Stand-up-Flat-Leggy` | Train Leggy to stand up on flat ground |

## Next Steps

### For This Repository
1. Implement Onshape import command
2. Add scene description system (obstacles, terrain, props)
3. Create config files for observation/action space definitions
4. Build reward function library and configuration system
5. Add domain randomization for sim-to-real transfer

### For Deployment
1. Export trained policies for hardware deployment
2. Implement sim-to-real transfer techniques
3. Test on physical Leggy robot

## Contributing

This is a work in progress. Key areas needing development:
- Onshape import automation
- Scene/task configuration system
- Reward function templates
- Documentation and examples

## Resources

- [mjlab GitHub](https://github.com/mujocolab/mjlab)
- [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

[Add your license here]
