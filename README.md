# LeggySim

A robot simulation environment package for training biped robots using reinforcement learning. This project enables importing robots from Onshape, creating MuJoCo-based simulation environments with parallel execution, and training RL policies with [mjlab](https://github.com/mujocolab/mjlab).

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <repository-url>
cd leggySim
uv sync
```

## Usage

### Training

```bash
# Train with torque metrics logged to wandb
uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048

# See all available options
uv run leggy-train Mjlab-Leggy --help
```

### Play/Evaluation

```bash
# Play with a wandb checkpoint
uv run leggy-play Mjlab-Leggy --wandb-run-path <your-wandb-path>

# Play with a local checkpoint
uv run leggy-play Mjlab-Leggy --checkpoint-file logs/rsl_rl/leggy/<run>/model_*.pt

# Use random actions (for testing)
uv run leggy-play Mjlab-Leggy --agent random
```

Arrow keys: Up/Down = forward speed, Left/Right = rotation.

### Visual Debugging

```bash
python -m mujoco.viewer --mjcf=src/mjlab_leggy/leggy/robot.xml
```

## Project Structure

```
leggySim/
├── pyproject.toml
└── src/
    └── mjlab_leggy/
        ├── __init__.py
        ├── scripts/
        │   ├── train.py            # Training entry point (torque logging)
        │   └── play.py             # Play entry point (keyboard control)
        ├── leggy/                   # Robot model and configuration
        │   ├── robot.xml            # MuJoCo robot model
        │   ├── scene.xml            # Scene with robot + ground
        │   ├── sensors.xml          # Sensor definitions
        │   ├── config.json          # Robot configuration
        │   ├── assets/              # Robot mesh files (STL)
        │   ├── leggy_constants.py   # Robot constants
        │   ├── leggy_actions.py     # Motor-to-knee action term
        │   ├── leggy_observations.py # Observation configuration
        │   ├── leggy_rewards.py     # Custom reward functions
        │   ├── leggy_config.py      # Reusable config functions
        │   ├── leggy_curriculums.py # Velocity curriculum stages
        │   ├── leggy_env.py         # Custom env with torque logging
        │   └── keyboard_controller.py # Arrow key velocity control
        └── tasks/
            ├── __init__.py          # Task registration
            └── leggy.py             # Leggy task definition
```

## Environment Details

### Observation Space

Policy observations (with 5-step history):
- **Body Euler Angles** (3D): Roll, pitch, yaw from IMU
- **Base Linear Velocity** (3D): Robot body velocity
- **Base Angular Velocity** (3D): Robot body angular velocity
- **Joint Positions** (6D): Motor-space positions
- **Joint Velocities** (6D): Motor-space velocities
- **Joint Torques** (6D): Motor torques
- **Actions** (6D): Previous action
- **Command** (3D): Velocity command (lin_vel_x, lin_vel_y, ang_vel_z)

### Action Space
- **Continuous**: 6 joint position targets
- **Joints**: LhipY, LhipX, Lmotor, RhipY, RhipX, Rmotor (converted to knee space)

## Available Tasks

| Task Name | Description |
|-----------|-------------|
| `Mjlab-Leggy` | Locomotion training with progressive velocity curriculum |

## Resources

- [mjlab GitHub](https://github.com/mujocolab/mjlab)
- [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
