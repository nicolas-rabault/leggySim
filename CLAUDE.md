# LeggySim

Biped robot locomotion training using mjlab + MuJoCo + rsl_rl (PPO).

## Commands

```bash
uv sync                                              # Install dependencies
uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048  # Train
uv run leggy-play Mjlab-Leggy --checkpoint-file <path>    # Play
```

Training happens on a remote GPU server, not on the dev machine.

## The Robot

Leggy is a small biped designed for dynamic locomotion (fast running, jumping).

- **Naturally unstable**: feet are dot contacts (no flat foot), the robot must actively balance
- **6 actuated joints**: LhipY, LhipX, Lknee, RhipY, RhipX, Rknee
- **4 passive joints**: LpassiveMotor, RpassiveMotor, Lpassive2, Rpassive2 (4-bar linkage)
- **Motor-to-knee conversion**: knee = motor - hipX (the 4-bar mechanism offsets knee by hipX angle)
- **Strong motors** relative to robot weight — enables dynamic gaits
- **Sim tuning**: contact physics (solref, solimp, friction) are carefully tuned for the 4-bar mechanism
- **Physical robot exists** — sim-to-real transfer is a future goal

## Architecture

```
src/mjlab_leggy/
├── scripts/          # Entry points (train.py, play.py)
├── leggy/            # Robot config, actions, observations, rewards, curriculums
│   ├── leggy_actions.py      # Motor-to-knee action term + observation helpers
│   ├── leggy_constants.py    # Robot config, stand pose, HOME_FRAME
│   ├── leggy_observations.py # Policy/critic observation setup
│   ├── leggy_rewards.py      # Custom reward functions
│   ├── leggy_config.py       # Reusable config functions (sensors, terrain, etc.)
│   ├── leggy_curriculums.py  # Velocity curriculum stages
│   └── leggy_env.py          # Custom env with torque logging to wandb
└── tasks/
    └── leggy.py              # Task definition (env_cfg + rl_cfg)
```

Key pattern: scripts monkey-patch `_train.ManagerBasedRlEnv = LeggyRlEnv` to inject custom env without modifying mjlab.

## Conventions

- Keep code concise — minimal comments, no verbose docstrings
- Use `uv` for all dependency management (never pip)
- Only commit when explicitly asked
- Never push without asking
- Task name: `Mjlab-Leggy`
- Actuator order everywhere: `[LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]`
- Motor space in observations, knee space in simulation
