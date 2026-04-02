# Generic RL Training Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the rl-training skill project-agnostic by extracting all project-specific config and scripts into project-local files, with a setup agent to generate them for new projects.

**Architecture:** Generic SKILL.md reads project-local `.claude/rl-training/config.md` + infra memory + scripts. A SETUP phase (setup agent) interviews the user and generates these files for new projects. LeggySim is migrated first as the reference implementation.

**Tech Stack:** Bash, Python, Markdown (SKILL.md), WandB API, SSH

**Spec:** `docs/superpowers/specs/2026-04-01-generic-rl-training-skill-design.md`

---

### Task 1: Create LeggySim config.md

**Files:**
- Create: `.claude/rl-training/config.md`

**Important:** This creates the `.claude/rl-training/` directory structure. The existing `scripts/` directory remains unchanged — it is used by other tools (e.g., `export_web.py`). Do not modify or delete it.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p .claude/rl-training/scripts
```

- [ ] **Step 2: Write config.md**

```markdown
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

## Notifications
- Enabled: true
- Method: skill:discord-notify
- When: [training_started, monitor_update, eval_complete, training_killed, iteration_started, blocker]

## Source Files
- Task config: src/mjlab_leggy/tasks/leggy_run.py
- Rewards: src/mjlab_leggy/leggy/leggy_rewards.py
- Observations: src/mjlab_leggy/leggy/leggy_observations.py
- Curriculums: src/mjlab_leggy/leggy/leggy_curriculums.py
- Actions: src/mjlab_leggy/leggy/leggy_actions.py
- Environment: src/mjlab_leggy/leggy/leggy_env.py
```

- [ ] **Step 3: Verify config.md is complete**

Read `.claude/rl-training/config.md` and confirm all sections from the spec schema are present: Robot, Task, Training, Monitoring, Evaluation, Notifications, Source Files.

- [ ] **Step 4: Commit**

```bash
git add .claude/rl-training/config.md
git commit -m "feat: add LeggySim rl-training config.md"
```

---

### Task 2: Create LeggySim infra memory

**Files:**
- Create: `~/.claude/projects/-Users-nicolasrabault-Projects-LeParkour-leggySim/memory/rl_training_infra.md`
- Modify: `~/.claude/projects/-Users-nicolasrabault-Projects-LeParkour-leggySim/memory/MEMORY.md`

- [ ] **Step 1: Write rl_training_infra.md**

```markdown
---
name: rl_training_infra
description: SSH, WandB, and remote access details for RL training on lerobot GPU server
type: project
---

## Remote Access
- SSH host: lerobot
- SSH user: (default — uses SSH config alias)
- Remote project path: ~/leggySim
- SSH tunnel command: sft ssh lerobot
- GPU check command: nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
- Screen session name: leggy-train
- Remote PATH setup: export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

## WandB
- Project: rabault-nicolas-leggy/mjlab
- Dashboard: https://wandb.ai/rabault-nicolas-leggy/mjlab
```

- [ ] **Step 2: Update MEMORY.md index**

Add this line under the appropriate section:

```markdown
- [RL Training Infra](rl_training_infra.md) — SSH host, WandB project, remote paths for training
```

- [ ] **Step 3: Verify the memory file exists and reads correctly**

Read the file back to confirm content and frontmatter are correct.

---

### Task 3: Create project-local init_session.sh

**Files:**
- Create: `.claude/rl-training/scripts/init_session.sh`

- [ ] **Step 1: Write init_session.sh**

This script is nearly generic — only the session directory path matters. Adapted from `scripts/init_session.sh`:

```bash
#!/usr/bin/env bash
# Initialize or reset a training session directory.
# Usage: .claude/rl-training/scripts/init_session.sh "<goal>" "<branch>"

set -euo pipefail

GOAL="$1"
BRANCH="$2"

SESSION_DIR="logs/training_session"

if [ -f "$SESSION_DIR/session_state.json" ]; then
    BACKUP="$SESSION_DIR/session_state.$(date +%Y%m%d_%H%M%S).json"
    cp "$SESSION_DIR/session_state.json" "$BACKUP"
    echo "Previous state backed up to $BACKUP"
fi

mkdir -p "$SESSION_DIR/run_001"

cat > "$SESSION_DIR/session_state.json" << EOF
{
  "goal": "$GOAL",
  "branch": "$BRANCH",
  "current_run": 1,
  "wandb_run_path": "",
  "phase": "CODE",
  "monitor_count": 0,
  "consecutive_bad": 0,
  "iterations": []
}
EOF

echo "Session initialized at $SESSION_DIR"
echo "State: $SESSION_DIR/session_state.json"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x .claude/rl-training/scripts/init_session.sh
```

- [ ] **Step 3: Verify**

```bash
cat .claude/rl-training/scripts/init_session.sh
```

Confirm script is valid bash with correct session_state.json structure.

- [ ] **Step 4: Commit**

```bash
git add .claude/rl-training/scripts/init_session.sh
git commit -m "feat: add project-local init_session.sh for rl-training"
```

---

### Task 4: Create project-local train.sh

**Files:**
- Create: `.claude/rl-training/scripts/train.sh`

- [ ] **Step 1: Write train.sh**

Adapted from `scripts/train_remote.sh`. Reads infra memory for host details, reads config for training command. The SKILL.md agent reads these files and passes values as arguments.

```bash
#!/usr/bin/env bash
# Launch training — remote or local.
# Usage: .claude/rl-training/scripts/train.sh <branch> <ssh-host> <remote-dir> <screen-name> <train-command> [<deps-command>] [<remote-path-setup>]
#
# For local training:
#   .claude/rl-training/scripts/train.sh <branch> local "" "" "<train-command>" [<deps-command>]

set -euo pipefail

BRANCH="$1"
SSH_HOST="$2"
REMOTE_DIR="$3"
SCREEN_NAME="$4"
TRAIN_CMD="$5"
DEPS_CMD="${6:-}"
REMOTE_ENV="${7:-}"

if [ "$SSH_HOST" = "local" ]; then
    echo "=== Local training ==="
    git checkout "$BRANCH"
    [ -n "$DEPS_CMD" ] && eval "$DEPS_CMD"
    echo "=== Launching training ==="
    eval "$TRAIN_CMD" &
    echo "Training PID: $!"
    exit 0
fi

echo "=== Checking SSH connection ==="
if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot connect to $SSH_HOST" >&2
    exit 1
fi

echo "=== Checking GPU usage ==="
GPU_UTIL=$(ssh "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
echo "GPU utilization: ${GPU_UTIL}%"
if [ "$GPU_UTIL" -gt 50 ]; then
    echo "ERROR: GPU utilization is ${GPU_UTIL}% — someone else may be using it" >&2
    exit 1
fi

echo "=== Pulling branch $BRANCH ==="
ssh "$SSH_HOST" "cd $REMOTE_DIR && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"

if [ -n "$DEPS_CMD" ]; then
    echo "=== Syncing dependencies ==="
    if [ -n "$REMOTE_ENV" ]; then
        ssh "$SSH_HOST" "$REMOTE_ENV && cd $REMOTE_DIR && $DEPS_CMD"
    else
        ssh "$SSH_HOST" "cd $REMOTE_DIR && $DEPS_CMD"
    fi
fi

echo "=== Launching training in screen ==="
ssh "$SSH_HOST" "screen -ls | grep -q $SCREEN_NAME && screen -S $SCREEN_NAME -X stuff \$'\\003' || screen -dmS $SCREEN_NAME"
sleep 1

FULL_CMD="cd $REMOTE_DIR && $TRAIN_CMD"
[ -n "$REMOTE_ENV" ] && FULL_CMD="$REMOTE_ENV && $FULL_CMD"
ssh "$SSH_HOST" "screen -S $SCREEN_NAME -X stuff '$FULL_CMD\n'"

echo "=== Training launched ==="
echo "Monitor with: ssh $SSH_HOST 'screen -r $SCREEN_NAME'"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x .claude/rl-training/scripts/train.sh
```

- [ ] **Step 3: Verify syntax**

```bash
bash -n .claude/rl-training/scripts/train.sh
```

Expected: no output (no syntax errors).

- [ ] **Step 4: Commit**

```bash
git add .claude/rl-training/scripts/train.sh
git commit -m "feat: add project-local train.sh for rl-training"
```

---

### Task 5: Create project-local kill_training.sh

**Files:**
- Create: `.claude/rl-training/scripts/kill_training.sh`

- [ ] **Step 1: Write kill_training.sh**

```bash
#!/usr/bin/env bash
# Kill a training session — remote or local.
# Usage: .claude/rl-training/scripts/kill_training.sh <ssh-host> <screen-name>
#
# For local: .claude/rl-training/scripts/kill_training.sh local <pid>

set -euo pipefail

SSH_HOST="$1"
SCREEN_NAME="$2"

if [ "$SSH_HOST" = "local" ]; then
    echo "Killing local process $SCREEN_NAME..."
    kill "$SCREEN_NAME" 2>/dev/null || true
    echo "Training killed."
    exit 0
fi

echo "Sending Ctrl-C to $SCREEN_NAME screen on $SSH_HOST..."
ssh "$SSH_HOST" "screen -S $SCREEN_NAME -X stuff \$'\\003'"
sleep 2

if ssh "$SSH_HOST" "screen -ls | grep -q $SCREEN_NAME" 2>/dev/null; then
    echo "Screen still exists, sending quit..."
    ssh "$SSH_HOST" "screen -S $SCREEN_NAME -X quit" 2>/dev/null || true
fi

echo "Training killed."
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x .claude/rl-training/scripts/kill_training.sh
bash -n .claude/rl-training/scripts/kill_training.sh
```

- [ ] **Step 3: Commit**

```bash
git add .claude/rl-training/scripts/kill_training.sh
git commit -m "feat: add project-local kill_training.sh for rl-training"
```

---

### Task 6: Create project-local get_latest_run.py

**Files:**
- Create: `.claude/rl-training/scripts/get_latest_run.py`

- [ ] **Step 1: Write get_latest_run.py**

Adapted from `scripts/get_latest_wandb_run.py`. Takes WandB project as argument instead of hardcoding it.

```python
#!/usr/bin/env python3
"""Find the latest wandb run for a project.

Usage:
    uv run .claude/rl-training/scripts/get_latest_run.py <wandb-project> [--state running] [--wait 300]

Output: run path (entity/project/runs/run_id) on stdout, exit 1 if not found.
"""

import argparse
import sys
import time

import wandb


def find_run(api, project, state):
    runs = api.runs(project, filters={"state": state}, order="-created_at", per_page=1)
    for run in runs:
        return f"{project}/runs/{run.id}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="wandb project: entity/project")
    parser.add_argument("--state", default="running")
    parser.add_argument("--wait", type=int, default=0, help="Max seconds to poll")
    args = parser.parse_args()

    api = wandb.Api()

    if args.wait <= 0:
        path = find_run(api, args.project, args.state)
        if path:
            print(path)
        else:
            print("No run found", file=sys.stderr)
            sys.exit(1)
        return

    deadline = time.time() + args.wait
    while time.time() < deadline:
        path = find_run(api, args.project, args.state)
        if path:
            print(path)
            return
        time.sleep(15)

    print(f"No {args.state} run found after {args.wait}s", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x .claude/rl-training/scripts/get_latest_run.py
python3 -m py_compile .claude/rl-training/scripts/get_latest_run.py
```

- [ ] **Step 3: Commit**

```bash
git add .claude/rl-training/scripts/get_latest_run.py
git commit -m "feat: add project-local get_latest_run.py for rl-training"
```

---

### Task 7: Create project-local monitor.py

**Files:**
- Create: `.claude/rl-training/scripts/monitor.py`

- [ ] **Step 1: Write monitor.py**

Adapted from `scripts/monitor_wandb.py`. Takes metric categories as arguments so it's not hardcoded to Leggy metric names.

```python
#!/usr/bin/env python3
"""Fetch wandb metrics and output a concise markdown summary.

Usage:
    uv run .claude/rl-training/scripts/monitor.py <run-path> [--previous <file>] [--categories <cat1,cat2,...>]
"""

import argparse
import json
import sys
from pathlib import Path

import wandb


def trend(current, previous, key):
    if previous is None or key not in previous or previous[key] is None:
        return ""
    if current.get(key) is None:
        return ""
    diff = current[key] - previous[key]
    if abs(diff) < 1e-6:
        return " (stable)"
    return f" ({'↑' if diff > 0 else '↓'} {abs(diff):.4f})"


def format_summary(run, latest, previous_check, categories):
    status = run.state
    step = latest.get("_step", "?")

    lines = [
        f"# WandB Monitor — {run.name}",
        f"**Status**: {status} | **Step**: {step}",
        "",
        "## Metrics",
    ]

    for cat in categories:
        cat_keys = sorted(k for k in latest if k.startswith(cat) and latest[k] is not None)
        if not cat_keys:
            continue
        lines.append(f"\n### {cat.rstrip('/')}")
        for k in cat_keys:
            name = k.replace(cat, "")
            val = latest[k]
            t = trend(latest, previous_check, k)
            if isinstance(val, float):
                lines.append(f"- **{name}**: {val:.4f}{t}")
            else:
                lines.append(f"- **{name}**: {val}{t}")

    return "\n".join(lines)


def load_previous_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text()
    marker = "<!-- RAW_METRICS:"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.index("-->", start)
        return json.loads(text[start:end])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path: entity/project/runs/run_id")
    parser.add_argument("--previous", help="Path to previous monitor output for trend comparison")
    parser.add_argument("--categories", default="", help="Comma-separated metric category prefixes")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)
    run.update()
    latest = dict(run.summary)

    if not latest or "_step" not in latest:
        print("No metrics available yet.")
        sys.exit(0)

    previous_check = None
    if args.previous:
        previous_check = load_previous_metrics(args.previous)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else []
    if not categories:
        # Auto-discover categories from metric keys
        prefixes = set()
        for k in latest:
            if "/" in k and not k.startswith("_"):
                prefixes.add(k.rsplit("/", 1)[0] + "/")
        categories = sorted(prefixes)

    summary = format_summary(run, latest, previous_check, categories)
    print(summary)

    print(f"\n<!-- RAW_METRICS:{json.dumps({k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in latest.items() if v is not None})}-->")

    sys.exit(2 if run.state in ("killed", "crashed") else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x .claude/rl-training/scripts/monitor.py
python3 -m py_compile .claude/rl-training/scripts/monitor.py
```

- [ ] **Step 3: Commit**

```bash
git add .claude/rl-training/scripts/monitor.py
git commit -m "feat: add project-local monitor.py for rl-training"
```

---

### Task 8: Create project-local evaluate_policy.py

**Files:**
- Create: `.claude/rl-training/scripts/evaluate_policy.py`

- [ ] **Step 1: Write evaluate_policy.py**

This is the most project-specific script. For LeggySim it imports LeggyRlEnv, uses Leggy-specific test commands and joint names. The script is generated per-project by the setup agent — this is the LeggySim version.

```python
#!/usr/bin/env python3
"""Evaluate a trained policy headlessly — log metrics and record video.

Usage:
    uv run python .claude/rl-training/scripts/evaluate_policy.py <wandb-run-path> [--output-dir <dir>]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.wrappers import VideoRecorder
from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

import mjlab_leggy.tasks  # noqa: F401 — registers Mjlab-Leggy

TASK_ID = "Mjlab-Leggy"
JOINTS = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]

# (name, lin_vel_x, lin_vel_y, ang_vel_z, duration_steps)
TEST_COMMANDS = [
    ("stand", 0.0, 0.0, 0.0, 100),
    ("walk_forward", 0.5, 0.0, 0.0, 200),
    ("run_forward", 2.0, 0.0, 0.0, 200),
    ("turn_left", 0.5, 0.0, 1.0, 200),
    ("turn_right", 0.5, 0.0, -1.0, 200),
    ("side_step", 0.0, 0.5, 0.0, 200),
]


def evaluate(run_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    rl_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720
    env_cfg.viewer.distance = 1.0

    log_root = (Path("logs") / "rsl_rl" / rl_cfg.experiment_name).resolve()
    checkpoint_path, was_cached = get_wandb_checkpoint_path(log_root, Path(run_path))
    print(f"[INFO] Checkpoint: {checkpoint_path.name} ({'cached' if was_cached else 'downloaded'})")

    total_steps = sum(s for _, _, _, _, s in TEST_COMMANDS)
    env = LeggyRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env.metadata["render_fps"] = 30
    env = VideoRecorder(
        env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(checkpoint_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = wrapped.get_observations()
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    robot = env.unwrapped.scene["robot"]

    metrics = {}

    for cmd_name, vx, vy, vz, duration in TEST_COMMANDS:
        vel_errors_xy, vel_errors_yaw, torques, falls = [], [], [], 0

        for _ in range(duration):
            cmd_term.vel_command_b[:, 0] = vx
            cmd_term.vel_command_b[:, 1] = vy
            cmd_term.vel_command_b[:, 2] = vz
            cmd_term.time_left[:] = 1e9

            action = policy(obs)
            obs, _, dones, extras = wrapped.step(action)

            actual_vel = robot.data.root_link_lin_vel_b[0].cpu()
            actual_ang = robot.data.root_link_ang_vel_b[0, 2].cpu().item()
            torque = robot.data.actuator_force[0].cpu().abs().tolist()

            vel_errors_xy.append(((actual_vel[0].item() - vx)**2 + (actual_vel[1].item() - vy)**2)**0.5)
            vel_errors_yaw.append(abs(actual_ang - vz))
            torques.append(torque)

            if dones.any():
                falls += 1

        n = len(vel_errors_xy)
        avg_torques = [sum(t[i] for t in torques) / n for i in range(len(JOINTS))]
        metrics[cmd_name] = {
            "command": {"vx": vx, "vy": vy, "vz": vz},
            "rms_vel_error_xy": (sum(e**2 for e in vel_errors_xy) / n)**0.5,
            "rms_vel_error_yaw": (sum(e**2 for e in vel_errors_yaw) / n)**0.5,
            "falls": falls,
            "mean_torque": avg_torques,
        }

    env.close()

    lines = [f"# Policy Evaluation — {run_path}", ""]
    for name, m in metrics.items():
        cmd = m["command"]
        lines.append(f"## {name} (vx={cmd['vx']}, vy={cmd['vy']}, vz={cmd['vz']})")
        lines.append(f"- RMS velocity error XY: {m['rms_vel_error_xy']:.4f}")
        lines.append(f"- RMS velocity error yaw: {m['rms_vel_error_yaw']:.4f}")
        lines.append(f"- Falls: {m['falls']}")
        torque_str = ", ".join(f"{JOINTS[i]}={m['mean_torque'][i]:.3f}" for i in range(len(JOINTS)))
        lines.append(f"- Mean torque: {torque_str}")
        lines.append("")

    summary = "\n".join(lines)
    (output_dir / "eval_metrics.md").write_text(summary)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary)

    video_files = list(output_dir.glob("*.mp4"))
    video_path = video_files[0] if video_files else None
    print(f"\nVideo: {video_path}")
    print(f"Metrics: {output_dir / 'eval_metrics.md'}")
    return str(video_path), str(output_dir / 'eval_metrics.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path: entity/project/runs/run_id")
    parser.add_argument("--output-dir", default="logs/training_session/eval")
    args = parser.parse_args()
    evaluate(args.run_path, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x .claude/rl-training/scripts/evaluate_policy.py
python3 -m py_compile .claude/rl-training/scripts/evaluate_policy.py
```

- [ ] **Step 3: Commit**

```bash
git add .claude/rl-training/scripts/evaluate_policy.py
git commit -m "feat: add project-local evaluate_policy.py for rl-training"
```

---

### Task 9: Rewrite SKILL.md — frontmatter + architecture + SETUP phase

**Files:**
- Modify: `~/.claude/skills/rl-training/SKILL.md`

This is the core of the refactor. The new SKILL.md is fully generic. We rewrite it in 3 tasks (9, 10, 11) to keep changes manageable.

- [ ] **Step 1: Write the new SKILL.md header, architecture, prerequisites, and SETUP phase**

Replace the entire file. Start with everything up to and including the SETUP phase:

```markdown
---
name: rl-training
description: Manage RL training loops — modify tasks, launch training, monitor metrics, evaluate policies, iterate. Use when user asks to train, improve, or create an RL task. Use when user asks to "make the robot do X" or "improve the gait" or "train a new behavior". Works with any RL project after setup.
---

# RL Training Manager

Autonomously manage the full RL training loop using a multi-agent architecture. Works with any RL project — reads project-specific configuration from `.claude/rl-training/config.md`.

## Architecture

```
Main (this conversation)
  ├─ Phase 0: SETUP (if no config exists)
  ├─ Phase 1-2: Clarify + Code (directly)
  ├─ Phase 3: Spawn Run Agent (foreground) → starts training, returns run path
  ├─ Creates ONE recurring durable cron (every ~30 min)
  └─ Done — main conversation ends or waits

Recurring Monitor Cron (fires every ~30 min, reads session_state.json):
  phase=MONITOR → fetch metrics, eval, notify, decide keep/kill
  phase=ITERATE → diagnose, fix code, relaunch training, set phase=MONITOR
  phase=FINISHED → do nothing, exit
  phase=PAUSED → do nothing, exit
```

**Communication is file-based** — all agents read/write `logs/training_session/`:
- `session_state.json` — global state (phase, goal, run#, run path, etc.)
- `run_NNN/context.md` — what was changed and why (written before launch)
- `run_NNN/monitor_MMM.md` — monitoring output
- `run_NNN/result.md` — final summary when run is killed

## Prerequisites

- Project config at `.claude/rl-training/config.md` (generated by SETUP phase if missing)
- Infra memory at `~/.claude/projects/<hash>/memory/rl_training_infra.md`
- Project-local scripts at `.claude/rl-training/scripts/`

## Phase 0: SETUP

**Trigger**: `.claude/rl-training/config.md` does not exist, OR user explicitly asks to update training config.

If config exists and user wants to update, show current config sections and ask which to update. Only regenerate affected scripts.

### Step 1: Codebase Exploration (autonomous)

No user interaction. Scan the project:
- Read project structure, dependencies (`pyproject.toml`, `setup.py`, `requirements.txt`), entry points
- Identify: simulator, RL framework, algorithm, task names
- Look at existing training scripts, config files, reward definitions
- Check local hardware: `nvidia-smi` for GPU, `uname` for OS
- Produce a findings summary for the next steps

### Step 2: Robot & Objective (interactive)

Present findings from Step 1. Ask user to confirm or correct. Then ask:
- Robot type and key physical traits / constraints
- Actuated joints and any special mechanics
- Training objective — what should the robot learn?
- One question at a time.

### Step 3: Training Infrastructure (interactive)

- Local or remote training?
- If remote: SSH host, user, remote project path, tunnel command if needed → store in `rl_training_infra.md` (project memory, gitignored)
- If local: confirm GPU availability
- Dependency management command (e.g., `uv sync`, `pip install -e .`)
- Exact training launch command

### Step 4: Monitoring & Evaluation (interactive)

- What monitoring tool? (WandB, TensorBoard, local logs)
- If WandB: project path → store in `rl_training_infra.md`
- Key metric categories and prefixes to track
- What does "good" vs "bad" look like for this task?
- Evaluation: what scenarios to test, what metrics, record video?
- May iterate with user to refine eval strategy

### Step 5: Notifications (interactive)

- Does the user want notifications about training progress?
- If yes: via installed Claude Code skill (e.g., `discord-notify`) or custom script?
  - If skill: store skill name in config (`method: skill:<name>`)
  - If script: generate `.claude/rl-training/scripts/notify.sh` with the user's delivery method
- Which events trigger notifications? (training_started, monitor_update, eval_complete, training_killed, iteration_started, blocker)
- Store any credentials/webhooks in `rl_training_infra.md`

### Step 6: Generate (autonomous)

- Write `.claude/rl-training/config.md` from gathered info
- Write `rl_training_infra.md` to project memory
- Generate all scripts in `.claude/rl-training/scripts/`:
  - `init_session.sh` — session state management
  - `train.sh` — launch training (local or remote)
  - `kill_training.sh` — stop running training
  - `get_latest_run.py` — find active run (WandB or other)
  - `monitor.py` — fetch metrics and format markdown report
  - `evaluate_policy.py` — headless eval with video and metrics
  - `notify.sh` — (only if `method: script`) custom notification delivery. Generate based on user's chosen method:
    - Slack webhook: `curl -s -X POST -H 'Content-Type: application/json' -d "{\"text\":\"$1\"}" <webhook_url>`
    - Email: `echo "$1" | mail -s "RL Training" <email>`
    - Other webhook: adapt curl command to the API
    - Read credentials from `rl_training_infra.md` at runtime
    - Script interface is always: `notify.sh "<message>" [--file <path>]`
- Make all scripts executable
- Present summary of generated files to user for review
```

- [ ] **Step 2: Verify the file so far**

Read back `~/.claude/skills/rl-training/SKILL.md` and confirm the structure is correct: frontmatter, architecture, prerequisites, complete SETUP phase with all 6 steps.

- [ ] **Step 3: Commit**

```bash
cd ~/.claude/skills/rl-training && git add SKILL.md && git commit -m "feat: rewrite SKILL.md — generic header + SETUP phase"
```

Note: If the skill directory is not a git repo, just save and move on — we'll commit the final version.

---

### Task 10: Rewrite SKILL.md — CLARIFY, CODE, LAUNCH phases

**Files:**
- Modify: `~/.claude/skills/rl-training/SKILL.md`

- [ ] **Step 1: Append CLARIFY, CODE, and LAUNCH phases**

Add after the SETUP phase:

```markdown

## Phase 1: CLARIFY

Read `.claude/rl-training/config.md` for robot context, task details, and source files.

Ask the user until the goal is clear:
- What behavior should change?
- What does success look like?
- Constraints?

## Phase 2: CODE

### Step 1: INVESTIGATE

Read the relevant code before changing anything. Get source file paths from `config.md` → Source Files section.
- Read task config, rewards, curriculum, observations
- Previous run results if available (`logs/training_session/`)
- Write findings in `logs/training_session/run_NNN/analysis.md`

### Step 2: RESEARCH

Search the web for RL solutions relevant to the task if needed. Write approach in `analysis.md`.

### Step 3: IMPLEMENT

1. Branch: `git checkout -b claude/<short-description>`
2. Make targeted changes to files listed in `config.md` → Source Files
3. No dead code, no "just in case" code
4. Commit with clear message
5. Push: `git push -u origin claude/<short-description>`

## Phase 3: LAUNCH

Read `.claude/rl-training/config.md` and infra memory (`rl_training_infra.md`) for all parameters.

### Step 1: Write run context

Create `logs/training_session/run_NNN/context.md`:
```
# Run NNN Context
Goal: <user's goal>
Branch: claude/<name>
Changes: <what was modified and why>
Expected: <what we expect to see in training>
Previous failures: <summary of past runs if any>
```

### Step 2: Spawn Run Agent

Use the Agent tool (foreground). Read config.md and rl_training_infra.md, then build the agent prompt with the project-specific values:

```
You are a training launcher. Do these steps in order:

1. Initialize session:
   bash .claude/rl-training/scripts/init_session.sh "{{goal}}" "{{branch}}"

2. Launch training:
   bash .claude/rl-training/scripts/train.sh {{branch}} {{ssh_host}} {{remote_dir}} {{screen_name}} "{{train_command}}" "{{deps_command}}" "{{remote_env}}"
   (For local: use "local" as ssh_host)
   If SSH fails and config mentions a tunnel command, run it, wait 8s, retry.

3. Wait for run to appear:
   uv run .claude/rl-training/scripts/get_latest_run.py {{wandb_project}} --state running --wait 120
   Save the output path.

4. Update logs/training_session/session_state.json:
   - Set wandb_run_path to the path from step 3
   - Set phase to "MONITOR"
   - Set monitor_count to 0
   - Set consecutive_bad to 0

5. Send notification (if enabled in config):
   "Training Started — Run {{run_number}}\nBranch: {{branch}}\nGoal: {{goal}}"

6. Return: "Training started. Run path: <path>."
```

### Step 3: Create the monitoring cron

After the Run Agent returns, the MAIN CONVERSATION creates the cron:

```
CronCreate:
  recurring: true
  durable: true
  cron: "*/33 * * * *"
```

Use the MONITOR CRON PROMPT below as the prompt. Tell the user: "Training launched. Monitoring every ~30 min."
```

- [ ] **Step 2: Verify phases are appended correctly**

Read back the file and confirm CLARIFY → CODE → LAUNCH flow is complete, with config.md references replacing all hardcoded values.

- [ ] **Step 3: Commit**

```bash
cd ~/.claude/skills/rl-training && git add SKILL.md && git commit -m "feat: SKILL.md — generic CLARIFY, CODE, LAUNCH phases"
```

---

### Task 11: Rewrite SKILL.md — MONITOR CRON PROMPT

**Files:**
- Modify: `~/.claude/skills/rl-training/SKILL.md`

- [ ] **Step 1: Append the MONITOR CRON PROMPT**

This is the longest section. Add after LAUNCH:

````markdown

---

## MONITOR CRON PROMPT

Copy this as the `prompt` parameter for CronCreate. The cron reads config.md and session_state.json dynamically.

```
You are the autonomous training loop. Read project config and act based on current phase.

STEP 0: Read project context.
- Read .claude/rl-training/config.md → extract: task name, metric categories, key metrics, kill threshold, max iterations, evaluation config, notification config, source files.
- Read rl_training_infra.md from project memory → extract: SSH host, screen name, wandb project, remote details.
- Read logs/training_session/session_state.json → extract: phase, goal, branch, current_run, wandb_run_path, monitor_count, consecutive_bad, iterations.

STEP 1: Act based on phase.

=== IF phase = "FINISHED" or phase = "PAUSED" or phase = "CODE" ===
Do nothing. Exit immediately.

=== IF phase = "MONITOR" ===

1. Determine M = monitor_count + 1. Pad to 3 digits for filenames.
   Run directory: logs/training_session/run_{current_run padded to 3 digits}/
   Previous monitor: run_NNN/monitor_{M-1 padded}.md (if M > 1)

2. Fetch metrics:
   Read config.md → Monitoring.Tool and Monitoring.Metric categories.
   uv run .claude/rl-training/scripts/monitor.py <wandb_run_path> [--previous <prev_monitor>] [--categories <from config>]
   Save output to: run_NNN/monitor_{M padded}.md
   If this fails, notify "Monitor error — <error>" and exit.

3. Evaluate policy (if config says Video: true or Evaluation section exists):
   uv run python .claude/rl-training/scripts/evaluate_policy.py <wandb_run_path> --output-dir run_NNN/
   If eval fails, continue without video.

4. Send notification (if enabled and monitor_update in When list):
   Compose message: "Monitor M — Run N (step XXXX)\n<key metrics from config>\n<trend: improving/stable/degrading>"
   If notification method is skill:<name>, invoke that skill with the message.
   If notification method is script, call: bash .claude/rl-training/scripts/notify.sh "<message>" [--file run_NNN/rl-video-step-0.mp4]

5. DECIDE using key metrics from config:
   - Progressing well → KEEP
   - Converged (rewards plateaued, good eval) → FINISH
   - Bad (degrading metrics, poor tracking) → BAD

6. ACT:

   If KEEP:
   - Update session_state.json: monitor_count = M, consecutive_bad = 0

   If BAD (consecutive_bad < kill_threshold - 1):
   - Update session_state.json: monitor_count = M, consecutive_bad += 1

   If BAD (consecutive_bad >= kill_threshold - 1) → KILL:
   - Read config for SSH host and screen name from infra memory.
   - Kill training: bash .claude/rl-training/scripts/kill_training.sh <ssh_host> <screen_name>
   - Write run_NNN/result.md with: goal, kill step, metrics at death, eval results, trend, assessment.
   - Update session_state.json:
     - phase: "ITERATE"
     - Add to iterations: {run: N, result: "<one-line>"}
     - Increment current_run
     - mkdir -p logs/training_session/run_{new N padded}/
   - If current_run > max_iterations from config: notify "Max iterations reached — need user guidance", set phase: "PAUSED". Exit.
   - Notify: "Run N killed — <reason>. Will iterate on next cron fire."

   If FINISH:
   - Notify: "Training Complete — Run N\n<summary>"
   - Update session_state.json: phase: "FINISHED"

=== IF phase = "ITERATE" ===

A training run was killed. Diagnose, fix code, relaunch.

1. Read run_{previous_run}/result.md and run_{previous_run}/context.md
2. Read iterations array to understand what was tried before.

3. DIAGNOSE — read source files listed in config.md → Source Files section.
   Search the web if the problem is non-obvious.
   Use robot specificities from config.md → Robot section for informed decisions.

4. Write diagnosis: logs/training_session/run_{current_run}/analysis.md

5. Make targeted code changes. No dead code, no "just in case" code.

6. Commit: git add <files> && git commit -m "<what and why>"
7. Push: git push origin HEAD

8. Write logs/training_session/run_{current_run}/context.md

9. Launch training:
   Read config and infra for parameters.
   bash .claude/rl-training/scripts/train.sh <branch> <ssh_host> <remote_dir> <screen_name> "<train_command>" "<deps_command>" "<remote_env>"

10. Get run path:
    uv run .claude/rl-training/scripts/get_latest_run.py <wandb_project> --state running --wait 120

11. Update session_state.json: wandb_run_path, phase: "MONITOR", monitor_count: 0, consecutive_bad: 0

12. Notify: "Relaunching — Run {current_run}\nChanged: <summary>\nExpected: <what should improve>"

IMPORTANT:
- Read config.md Robot section for physical constraints and mechanics
- Read config.md Source Files for which files to modify
- Keep code concise, no verbose comments
- Only change what the diagnosis justifies
```
````

- [ ] **Step 2: Verify the cron prompt reads config.md everywhere**

Read the file and confirm: no hardcoded project values remain. All references go through config.md or rl_training_infra.md.

- [ ] **Step 3: Commit**

```bash
cd ~/.claude/skills/rl-training && git add SKILL.md && git commit -m "feat: SKILL.md — generic MONITOR CRON PROMPT"
```

---

### Task 12: Rewrite SKILL.md — Recovery, Decision Rules, References

**Files:**
- Modify: `~/.claude/skills/rl-training/SKILL.md`

- [ ] **Step 1: Append Recovery, Decision Rules, and Scripts sections**

Add after the MONITOR CRON PROMPT:

```markdown

---

## Recovery

If the conversation restarts or the user re-invokes the skill:

1. Check if `.claude/rl-training/config.md` exists. If not, run SETUP.
2. Read `logs/training_session/session_state.json`
3. Check if a monitoring cron exists (CronList). If not and phase is MONITOR or ITERATE, create one.
4. Based on phase:
   - **MONITOR**: Cron handles it. Ensure cron exists.
   - **ITERATE**: Cron handles it on next fire. Or trigger manually.
   - **FINISHED**: Show results, ask if user wants to iterate further.
   - **PAUSED**: Show why (max iterations), ask user for guidance.
   - **CODE**: Continue with Phase 2.

## Decision Rules

- **Cron interval**: ~30 min (use odd minutes like */33 to avoid collisions)
- **Kill threshold**: Read from `config.md` → Monitoring.Kill threshold (default: 2)
- **Max iterations**: Read from `config.md` → Monitoring.Max iterations (default: 10)
- **When in doubt**: Send notification and wait for user

## Notification Delivery

Read `config.md` → Notifications section:
- If `Enabled: false` → skip all notifications
- If `Method: skill:<name>` → invoke the named skill with the message text
- If `Method: script` → call `.claude/rl-training/scripts/notify.sh "<message>" [--file <path>]`
- Only notify for events listed in the `When` field

## Scripts

All scripts live in `.claude/rl-training/scripts/`. The SKILL.md agent reads config.md and infra memory to build the correct arguments.

| Script | Usage |
|--------|-------|
| `init_session.sh "<goal>" "<branch>"` | Initialize session directory + state |
| `train.sh <branch> <host> <dir> <screen> "<cmd>" ["<deps>"] ["<env>"]` | Launch training (local or remote) |
| `kill_training.sh <host> <screen>` | Kill training session |
| `get_latest_run.py <project> [--state S] [--wait N]` | Find active run |
| `monitor.py <run_path> [--previous file] [--categories cats]` | Fetch metrics → markdown |
| `evaluate_policy.py <run_path> [--output-dir dir]` | Headless eval with video + metrics |
| `notify.sh "<message>" [--file path]` | (optional) Custom notification delivery |
```

- [ ] **Step 2: Verify the complete SKILL.md**

Read the full file. Check:
- No Leggy-specific references remain (no "leggy", "lerobot", "LhipY", etc.)
- No hardcoded WandB project paths
- No hardcoded Discord/webhook references
- All phases reference config.md or infra memory for project-specific values
- Script table matches the actual scripts in Task 3-8

- [ ] **Step 3: Commit**

```bash
cd ~/.claude/skills/rl-training && git add SKILL.md && git commit -m "feat: SKILL.md — Recovery, Decision Rules, Notification, Scripts reference"
```

---

### Task 13: Final verification and project commit

**Files:**
- Verify: `.claude/rl-training/config.md`
- Verify: `.claude/rl-training/scripts/*`
- Verify: `~/.claude/skills/rl-training/SKILL.md`
- Verify: `~/.claude/projects/<hash>/memory/rl_training_infra.md`

- [ ] **Step 1: Verify all project-local files exist**

```bash
ls -la .claude/rl-training/config.md .claude/rl-training/scripts/
```

Expected: config.md + 6 scripts (init_session.sh, train.sh, kill_training.sh, get_latest_run.py, monitor.py, evaluate_policy.py). All scripts executable.

- [ ] **Step 2: Verify no Leggy-specific content in SKILL.md**

```bash
grep -i -E "leggy|lerobot|rabault|LhipY|Mjlab-Leggy|discord" ~/.claude/skills/rl-training/SKILL.md
```

Expected: No matches. If `discord` appears, it should only be in the context of "e.g., discord-notify" as an example.

- [ ] **Step 3: Verify config.md has all required sections**

```bash
grep "^## " .claude/rl-training/config.md
```

Expected output:
```
## Robot
## Task
## Training
## Monitoring
## Evaluation
## Notifications
## Source Files
```

- [ ] **Step 4: Verify Python scripts compile**

```bash
python3 -m py_compile .claude/rl-training/scripts/get_latest_run.py
python3 -m py_compile .claude/rl-training/scripts/monitor.py
python3 -m py_compile .claude/rl-training/scripts/evaluate_policy.py
```

Expected: No output (no syntax errors).

- [ ] **Step 5: Verify Bash scripts have no syntax errors**

```bash
bash -n .claude/rl-training/scripts/init_session.sh
bash -n .claude/rl-training/scripts/train.sh
bash -n .claude/rl-training/scripts/kill_training.sh
```

Expected: No output (no syntax errors).

- [ ] **Step 6: Commit all project-local files**

```bash
git add .claude/rl-training/
git commit -m "feat: complete LeggySim migration to generic rl-training skill

Project-local config.md, 6 scripts in .claude/rl-training/scripts/.
Infra details in project memory (gitignored).
SKILL.md is now fully generic."
```
