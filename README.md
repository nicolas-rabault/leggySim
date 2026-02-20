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
- **Stand-up Task**: Train Leggy to stand up and balance
- **mjlab Integration**: Full integration with mjlab's training infrastructure
- **Parallel Execution**: Run thousands of synchronized environments for efficient training
- **Contact Sensors**: Foot contact detection for locomotion rewards

### Planned Features
- [ ] Command-line robot import from Onshape
- [ ] Scene description configuration
- [ ] Additional locomotion tasks (walking, jumping)
- [ ] Environment parameter randomization for sim-to-real transfer

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd leggySim

# Install and sync dependencies
uv sync
```

### Dependencies

Core dependency:
- `mjlab`: MuJoCo lab integration for robot simulation (pinned to stable revision `d1d32d8`)

Optional dependencies:
- `rerun-sdk`: 3D visualization
- `onshape-to-robot`: Robot URDF/XML generation from Onshape CAD (development only)

## Usage

### Training

Run training using mjlab's train command:

```bash
# Train with default settings
uv run train Mjlab-Stand-up-Flat-Leggy

# Adjust number of parallel environments
uv run train Mjlab-Stand-up-Flat-Leggy --env.scene.num-envs 2048

# See all available options
uv run train Mjlab-Stand-up-Flat-Leggy --help
```

### Play/Evaluation

```bash
# Play with a wandb checkpoint
uv run leggy-play Mjlab-Stand-up-Flat-Leggy --wandb-run-path <your-wandb-path>

# Play with a local checkpoint
uv run leggy-play Mjlab-Stand-up-Flat-Leggy --checkpoint-file logs/rsl_rl/leggy_stand_up/<run>/model_*.pt

# Use random actions (for testing)
uv run leggy-play Mjlab-Stand-up-Flat-Leggy --agent random
```

### Import Robot from Onshape

```bash
# TODO: Command to import robot from Onshape
# onshape-to-robot <onshape-url> --output leggy/
```

This will generate the robot XML files in the `leggy/` directory that MuJoCo can load.

### Visual Debugging

```bash
# Visualize robot with MuJoCo viewer
python -m mujoco.viewer --mjcf=src/mjlab_leggy/leggy/robot.xml
```

## Project Structure

```
leggySim/
├── pyproject.toml              # Package configuration and dependencies
├── env.py                      # Deprecated standalone environment (kept for reference)
├── infer_policy.py             # ONNX policy inference script
└── src/
    └── mjlab_leggy/            # Main package
        ├── __init__.py
        ├── leggy/              # Robot model directory
        │   ├── robot.xml       # MuJoCo robot model
        │   ├── scene.xml       # Scene with robot + ground
        │   ├── sensors.xml     # Sensor definitions
        │   ├── config.json     # Robot configuration
        │   ├── leggy_constants.py  # Robot constants and config
        │   └── assets/         # Robot mesh files (STL)
        └── tasks/
            ├── __init__.py     # Task registration
            └── leggy_stand_up.py  # Stand-up task definition
```

## Environment Details

### Observation Space

The policy observation includes (38 dimensions):
- **Base Linear Velocity** (3D): Robot body velocity
- **Base Angular Velocity** (3D): Robot body angular velocity
- **Projected Gravity** (3D): Gravity vector in body frame
- **Joint Positions** (10D): All joint positions (including passive joints)
- **Joint Velocities** (10D): All joint velocities
- **Actions** (6D): Previous action
- **Command** (3D): Velocity command (lin_vel_x, lin_vel_y, ang_vel_z)

The critic observation includes additional information (50 dimensions):
- All policy observations
- **Foot Height** (2D): Height of each foot
- **Foot Air Time** (2D): Time since last ground contact
- **Foot Contact** (2D): Binary contact state
- **Foot Contact Forces** (6D): Contact force vectors

### Action Space
- **Continuous**: 6 joint position targets
- **Joints**: LhipY, LhipX, Lknee, RhipY, RhipX, Rknee

## Available Tasks

| Task Name | Description |
|-----------|-------------|
| `Mjlab-Stand-up-Flat-Leggy` | Train Leggy to stand and balance on flat ground |

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

## License

[Add your license here]
