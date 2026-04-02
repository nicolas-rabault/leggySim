# RL Training Configuration

## Robot
- Name: Leggy
- Type: biped
- Specificities: Dot contacts (no flat foot), naturally unstable, must actively balance. Strong motors relative to weight — enables dynamic gaits. Physical robot exists, sim-to-real is future goal.
- Actuators: [LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]
- Special mechanics: 4-bar linkage with passive joints (LpassiveMotor, RpassiveMotor, Lpassive2, Rpassive2). Motor-to-knee conversion: knee = motor - hipX. Inverse: motor = knee + hipX. Motor space in observations, knee space in simulation.

## Task
- Name: Mjlab-Leggy
- Simulator: MuJoCo (via MuJoCo Warp — GPU-based)
- Framework: mjlab
- Algorithm: PPO (rsl_rl)
- Objective: Dynamic biped locomotion — running, turning, side-stepping with velocity tracking

## Training
- Command: uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048
- Execution: remote
- Env count: 2048
- Dependencies command: uv sync
- Screen name: leggy

## Monitoring
- Tool: wandb
- Metric categories: [Episode_Reward/, Train/, Episode_Termination/, Curriculum/command_vel/, Torque/, Loss/, Metrics/twist/]
- Key metrics: [Train/mean_reward, Metrics/twist/error_vel_xy, Metrics/twist/error_vel_yaw, Episode_Termination/time_out]
- Kill threshold: 2
- Max iterations: 10

## Evaluation
- Scenarios:
  - stand: vx=0.0, vy=0.0, vz=0.0, steps=100
  - walk_forward: vx=0.5, vy=0.0, vz=0.0, steps=200
  - run_forward: vx=2.0, vy=0.0, vz=0.0, steps=200
  - turn_left: vx=0.5, vy=0.0, vz=1.0, steps=200
  - turn_right: vx=0.5, vy=0.0, vz=-1.0, steps=200
  - side_step: vx=0.0, vy=0.5, vz=0.0, steps=200
- Metrics: [rms_vel_error_xy, rms_vel_error_yaw, falls, mean_torque_per_joint]
- Video: true

## Decision Criteria
- KEEP: Train/mean_reward increasing or stable, Metrics/twist/error_vel_xy decreasing
- BAD: Train/mean_reward dropped > 10% from peak, or error_vel_xy increasing for 2+ consecutive monitors
- FINISH: Train/mean_reward plateaued (< 2% change over 3 monitors) and error_vel_xy < 0.3

## Notifications
- Enabled: true
- Method: script
- When: [training_started, monitor_update, eval_complete, training_killed, iteration_started, blocker]

## Source Files
- Task config: src/mjlab_leggy/tasks/leggy_run.py
- Rewards: src/mjlab_leggy/leggy/leggy_rewards.py
- Observations: src/mjlab_leggy/leggy/leggy_observations.py
- Curriculums: src/mjlab_leggy/leggy/leggy_curriculums.py
- Actions: src/mjlab_leggy/leggy/leggy_actions.py
- Environment: src/mjlab_leggy/leggy/leggy_env.py
