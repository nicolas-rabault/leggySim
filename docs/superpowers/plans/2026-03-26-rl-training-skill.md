# RL Training Management Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Code skill + helper scripts that let Claude autonomously manage RL training for leggySim — code changes, remote training, wandb monitoring, policy evaluation, Discord updates, and iterative improvement.

**Architecture:** A skill file (`~/.claude/skills/rl-training/SKILL.md`) describes the workflow and decision logic. Four helper scripts in `scripts/` handle mechanical operations (SSH training, wandb polling, headless evaluation, Discord notifications). File-based state in `logs/training_session/` enables context management via subagents and session recovery.

**Tech Stack:** Python (wandb API, MuJoCo offscreen rendering, moviepy), Bash (SSH, screen, git, curl), Claude Code skills system

---

## File Structure

| File | Responsibility |
|------|---------------|
| `~/.claude/skills/rl-training/SKILL.md` | Skill definition — workflow, decision logic, agent prompts |
| `scripts/notify_discord.sh` | Send text/file messages to Discord webhook |
| `scripts/train_remote.sh` | SSH to lerobot, check GPU, pull branch, launch training in screen |
| `scripts/monitor_wandb.py` | Fetch wandb metrics, compute trends, output markdown summary |
| `scripts/evaluate_policy.py` | Run policy headless, log metrics, record video |

---

### Task 1: Discord Notification Script

**Files:**
- Create: `scripts/notify_discord.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# Send a message (and optional file) to the project Discord webhook.
# Usage:
#   scripts/notify_discord.sh "message text"
#   scripts/notify_discord.sh "message text" --file path/to/video.mp4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEBHOOK_URL=$(grep '^WEBHOOK_URL:' "$PROJECT_ROOT/.claude-discord.md" | sed 's/^WEBHOOK_URL: //')

if [ -z "$WEBHOOK_URL" ]; then
    echo "ERROR: No webhook URL found in .claude-discord.md" >&2
    exit 1
fi

MESSAGE="$1"
shift

FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --file) FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -n "$FILE" ]; then
    curl -s -F "payload_json={\"content\":\"$MESSAGE\"}" \
         -F "file=@$FILE" \
         "$WEBHOOK_URL"
else
    curl -s -H "Content-Type: application/json" \
         -d "{\"content\":\"$MESSAGE\"}" \
         "$WEBHOOK_URL"
fi
```

- [ ] **Step 2: Make executable and test with a text message**

Run: `chmod +x scripts/notify_discord.sh && scripts/notify_discord.sh "Test from notify_discord.sh"`
Expected: Message appears in Discord channel, no errors.

- [ ] **Step 3: Test with a file attachment**

Create a small test file and send it:
Run: `echo "test" > /tmp/test.txt && scripts/notify_discord.sh "File test" --file /tmp/test.txt && rm /tmp/test.txt`
Expected: Message with file attachment appears in Discord.

- [ ] **Step 4: Commit**

```bash
git add scripts/notify_discord.sh
git commit -m "feat: add Discord notification script"
```

---

### Task 2: Remote Training Script

**Files:**
- Create: `scripts/train_remote.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# Launch training on lerobot GPU server.
# Usage: scripts/train_remote.sh <branch-name>
# Prerequisites: sft ssh lerobot must have been run at least once this session.

set -euo pipefail

BRANCH="$1"
REMOTE_DIR="~/leggySim"

echo "=== Checking SSH tunnel ==="
if ! ssh -o ConnectTimeout=5 lerobot "echo ok" &>/dev/null; then
    echo "SSH tunnel not active. Starting sft..."
    sft ssh lerobot &
    SFT_PID=$!
    sleep 8
    if ! ssh -o ConnectTimeout=5 lerobot "echo ok" &>/dev/null; then
        echo "ERROR: Cannot connect to lerobot after starting sft" >&2
        kill $SFT_PID 2>/dev/null
        exit 1
    fi
fi

echo "=== Checking GPU usage ==="
GPU_UTIL=$(ssh lerobot "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
GPU_MEM=$(ssh lerobot "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
echo "GPU utilization: ${GPU_UTIL}%, Memory used: ${GPU_MEM} MiB"
if [ "$GPU_UTIL" -gt 50 ]; then
    echo "ERROR: GPU utilization is ${GPU_UTIL}% — someone else may be using it" >&2
    exit 1
fi

echo "=== Pulling branch $BRANCH ==="
ssh lerobot "cd $REMOTE_DIR && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"

echo "=== Syncing dependencies ==="
ssh lerobot "cd $REMOTE_DIR && uv sync"

echo "=== Launching training in screen ==="
# Create or reuse screen session, send the training command
ssh lerobot "screen -ls | grep -q leggy-train && screen -S leggy-train -X stuff $'\003' || screen -dmS leggy-train"
sleep 1
ssh lerobot "screen -S leggy-train -X stuff 'cd $REMOTE_DIR && uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048\n'"

echo "=== Training launched ==="
echo "Monitor with: ssh lerobot 'screen -r leggy-train'"
echo "Check wandb for the new run at: https://wandb.ai/rabault-nicolas-leggy/mjlab"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/train_remote.sh`

- [ ] **Step 3: Test SSH connectivity check (dry run)**

Run: `ssh lerobot "echo ok && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"`
Expected: `ok` followed by GPU stats.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_remote.sh
git commit -m "feat: add remote training launch script"
```

---

### Task 3: WandB Monitor Script

**Files:**
- Create: `scripts/monitor_wandb.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Fetch wandb metrics and output a concise markdown summary.

Usage:
    uv run scripts/monitor_wandb.py <run-path> [--previous <file>]
    uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/abc123
    uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/abc123 --previous logs/training_session/run_001/monitor_001.md
"""

import argparse
import json
import sys
from pathlib import Path

import wandb


def get_run_status(run):
    return run.state  # "running", "finished", "killed", "crashed"


def fetch_latest_metrics(run, num_samples=5):
    rows = run.history(samples=num_samples, pandas=False)
    if not rows:
        return None, None
    latest = rows[-1]
    previous = rows[-2] if len(rows) >= 2 else None
    return latest, previous


def trend(current, previous, key):
    if previous is None or key not in previous or previous[key] is None:
        return ""
    if current.get(key) is None:
        return ""
    diff = current[key] - previous[key]
    if abs(diff) < 1e-6:
        return " (stable)"
    return f" ({'↑' if diff > 0 else '↓'} {abs(diff):.4f})"


def format_summary(run, latest, previous_check):
    status = get_run_status(run)
    step = latest.get("_step", "?")

    lines = [
        f"# WandB Monitor — {run.name}",
        f"**Status**: {status} | **Step**: {step}",
        "",
        "## Key Metrics",
    ]

    reward_keys = sorted(k for k in latest if k.startswith("Episode_Reward/") and latest[k] is not None)
    if reward_keys:
        lines.append("### Rewards")
        for k in reward_keys:
            name = k.replace("Episode_Reward/", "")
            val = latest[k]
            t = trend(latest, previous_check, k)
            lines.append(f"- **{name}**: {val:.4f}{t}")

    lines.append("")
    lines.append("### Training")
    for k in ["Train/mean_reward", "Train/mean_episode_length"]:
        if latest.get(k) is not None:
            t = trend(latest, previous_check, k)
            lines.append(f"- **{k.split('/')[-1]}**: {latest[k]:.2f}{t}")

    lines.append("")
    lines.append("### Terminations")
    term_keys = sorted(k for k in latest if k.startswith("Episode_Termination/") and latest[k] is not None)
    for k in term_keys:
        name = k.replace("Episode_Termination/", "")
        lines.append(f"- **{name}**: {latest[k]:.4f}")

    lines.append("")
    lines.append("### Curriculum")
    curr_keys = sorted(k for k in latest if k.startswith("Curriculum/") and latest[k] is not None)
    for k in curr_keys:
        name = k.replace("Curriculum/command_vel/", "")
        lines.append(f"- **{name}**: {latest[k]:.4f}")

    lines.append("")
    lines.append("### Torque")
    torque_keys = sorted(k for k in latest if k.startswith("Torque/") and latest[k] is not None)
    for k in torque_keys:
        name = k.replace("Torque/", "")
        lines.append(f"- **{name}**: {latest[k]:.4f}")

    lines.append("")
    lines.append("### Loss")
    for k in ["Loss/value_function", "Loss/surrogate", "Loss/entropy"]:
        if latest.get(k) is not None:
            t = trend(latest, previous_check, k)
            lines.append(f"- **{k.split('/')[-1]}**: {latest[k]:.6f}{t}")

    lines.append("")
    lines.append("### Velocity Tracking")
    for k in ["Metrics/twist/error_vel_xy", "Metrics/twist/error_vel_yaw"]:
        if latest.get(k) is not None:
            t = trend(latest, previous_check, k)
            lines.append(f"- **{k.split('/')[-1]}**: {latest[k]:.4f}{t}")

    return "\n".join(lines)


def load_previous_metrics(path):
    """Extract the raw metrics JSON from a previous monitor file if embedded."""
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
    parser.add_argument("run_path", help="wandb run path: entity/project/run_id")
    parser.add_argument("--previous", help="Path to previous monitor output for trend comparison")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    latest, wandb_previous = fetch_latest_metrics(run)
    if latest is None:
        print("No metrics available yet.")
        sys.exit(0)

    previous_check = None
    if args.previous:
        previous_check = load_previous_metrics(args.previous)
    if previous_check is None:
        previous_check = wandb_previous

    summary = format_summary(run, latest, previous_check)
    print(summary)

    # Embed raw metrics for next comparison
    print(f"\n<!-- RAW_METRICS:{json.dumps({{k: v for k, v in latest.items() if v is not None}})}-->")

    # Exit code based on status
    status = get_run_status(run)
    if status == "running":
        sys.exit(0)
    elif status == "finished":
        sys.exit(0)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test against a real run**

Run: `uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/xdvcvih3`
Expected: Formatted markdown summary with all metric categories, status "finished".

- [ ] **Step 3: Test trend comparison**

Run:
```bash
uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/xdvcvih3 > /tmp/monitor_check1.md
uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/xdvcvih3 --previous /tmp/monitor_check1.md
```
Expected: Second run shows trend arrows (↑/↓/stable) next to metrics.

- [ ] **Step 4: Commit**

```bash
git add scripts/monitor_wandb.py
git commit -m "feat: add wandb monitoring script with trend tracking"
```

---

### Task 4: Policy Evaluation Script

**Files:**
- Create: `scripts/evaluate_policy.py`

This is the most complex script. It runs the policy headlessly, applies test velocity commands, logs metrics, and saves video.

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Evaluate a trained policy headlessly — log metrics and record video.

Usage:
    uv run --active mjpython scripts/evaluate_policy.py <wandb-run-path> [--output-dir <dir>] [--num-steps 500]
    uv run --active mjpython scripts/evaluate_policy.py rabault-nicolas-leggy/mjlab/xdvcvih3
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from mjlab_leggy.leggy.leggy_env import LeggyRlEnv
from rsl_rl.runners import OnPolicyRunner


TASK_ID = "Mjlab-Leggy"

# Test scenarios: (name, lin_vel_x, lin_vel_y, ang_vel_z, duration_steps)
TEST_COMMANDS = [
    ("stand", 0.0, 0.0, 0.0, 100),
    ("walk_forward", 0.5, 0.0, 0.0, 200),
    ("run_forward", 2.0, 0.0, 0.0, 200),
    ("turn_left", 0.5, 0.0, 1.0, 200),
    ("turn_right", 0.5, 0.0, -1.0, 200),
    ("side_step", 0.0, 0.5, 0.0, 200),
]


def get_checkpoint_path(run_path):
    import wandb as wb
    api = wb.Api()
    run = api.run(run_path)
    files = [f for f in run.files() if f.name.endswith(".pt")]
    if not files:
        raise RuntimeError(f"No .pt checkpoint found in run {run_path}")
    best = sorted(files, key=lambda f: f.name)[-1]
    local = best.download(replace=True)
    return Path(local.name)


def evaluate(run_path, output_dir, num_steps_override=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    rl_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1

    checkpoint_path = get_checkpoint_path(run_path)

    # Create env with video recording
    env = LeggyRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    total_steps = sum(s for _, _, _, _, s in TEST_COMMANDS)
    env = VideoRecorder(
        env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    # Load policy
    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(checkpoint_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs, _ = wrapped.reset()
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    robot = env.unwrapped.scene["robot"]

    metrics = {}
    global_step = 0

    for cmd_name, vx, vy, vz, duration in TEST_COMMANDS:
        cmd_metrics = {"vel_errors_xy": [], "vel_errors_yaw": [], "falls": 0, "torques": []}

        for step in range(duration):
            cmd_term.vel_command_b[:, 0] = vx
            cmd_term.vel_command_b[:, 1] = vy
            cmd_term.vel_command_b[:, 2] = vz
            cmd_term.time_left[:] = 1e9

            action = policy(obs)
            obs, _, terminated, truncated, info = wrapped.step(action)

            actual_vel = robot.data.root_link_lin_vel_b[0].cpu()
            actual_ang = robot.data.root_link_ang_vel_b[0, 2].cpu().item()
            torque = robot.data.actuator_force[0].cpu().abs().tolist()

            vel_err_xy = ((actual_vel[0].item() - vx)**2 + (actual_vel[1].item() - vy)**2)**0.5
            vel_err_yaw = abs(actual_ang - vz)

            cmd_metrics["vel_errors_xy"].append(vel_err_xy)
            cmd_metrics["vel_errors_yaw"].append(vel_err_yaw)
            cmd_metrics["torques"].append(torque)

            if terminated.any():
                cmd_metrics["falls"] += 1

            global_step += 1

        n = len(cmd_metrics["vel_errors_xy"])
        avg_torques = [sum(t[i] for t in cmd_metrics["torques"]) / n for i in range(6)]
        metrics[cmd_name] = {
            "command": {"vx": vx, "vy": vy, "vz": vz},
            "rms_vel_error_xy": (sum(e**2 for e in cmd_metrics["vel_errors_xy"]) / n)**0.5,
            "rms_vel_error_yaw": (sum(e**2 for e in cmd_metrics["vel_errors_yaw"]) / n)**0.5,
            "falls": cmd_metrics["falls"],
            "mean_torque": avg_torques,
        }

    env.close()

    # Write metrics summary
    lines = [f"# Policy Evaluation — {run_path}", ""]
    for name, m in metrics.items():
        cmd = m["command"]
        lines.append(f"## {name} (vx={cmd['vx']}, vy={cmd['vy']}, vz={cmd['vz']})")
        lines.append(f"- RMS velocity error XY: {m['rms_vel_error_xy']:.4f}")
        lines.append(f"- RMS velocity error yaw: {m['rms_vel_error_yaw']:.4f}")
        lines.append(f"- Falls: {m['falls']}")
        joints = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
        torque_str = ", ".join(f"{joints[i]}={m['mean_torque'][i]:.3f}" for i in range(6))
        lines.append(f"- Mean torque: {torque_str}")
        lines.append("")

    summary = "\n".join(lines)
    summary_path = output_dir / "eval_metrics.md"
    summary_path.write_text(summary)
    print(summary)

    # Also save raw JSON
    json_path = output_dir / "eval_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    # Find the video file
    video_files = list(output_dir.glob("*.mp4"))
    video_path = video_files[0] if video_files else None

    print(f"\nVideo: {video_path}")
    print(f"Metrics: {summary_path}")
    return str(video_path), str(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path: entity/project/run_id")
    parser.add_argument("--output-dir", default="logs/training_session/eval", help="Output directory")
    parser.add_argument("--num-steps", type=int, default=None, help="Override total steps")
    args = parser.parse_args()
    evaluate(args.run_path, args.output_dir, args.num_steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test against a finished run**

Run: `uv run --active mjpython scripts/evaluate_policy.py rabault-nicolas-leggy/mjlab/xdvcvih3 --output-dir /tmp/eval_test`
Expected: Prints metrics summary, saves `eval_metrics.md`, `eval_metrics.json`, and an `.mp4` video in the output dir.

- [ ] **Step 3: Verify video file is valid**

Run: `ls -la /tmp/eval_test/*.mp4`
Expected: A non-zero-size mp4 file exists.

- [ ] **Step 4: Test sending video to Discord**

Run: `scripts/notify_discord.sh "Policy eval test" --file /tmp/eval_test/*.mp4`
Expected: Video appears in Discord channel.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_policy.py
git commit -m "feat: add headless policy evaluation with metrics and video"
```

---

### Task 5: Skill File

**Files:**
- Create: `~/.claude/skills/rl-training/SKILL.md`

- [ ] **Step 1: Create the skill file**

```markdown
---
name: rl-training
description: Manage RL training for leggySim — modify tasks, launch remote training, monitor wandb, evaluate policies, iterate, notify via Discord. Use when user asks to train, improve, or create a task for the Leggy robot. Use when user asks to "make Leggy do X" or "improve the gait" or "train a new behavior".
---

# RL Training Manager — leggySim

Autonomously manage the full RL training loop for the Leggy biped robot.

## Prerequisites

- Discord webhook configured in `.claude-discord.md`
- SSH access to lerobot (run `sft ssh lerobot` first if tunnel is down)
- wandb authenticated on both machines

## Workflow

### Phase 1: CLARIFY

Ask the user questions until the goal is completely clear:
- What behavior should change?
- What does success look like?
- Any constraints (don't break existing gaits, keep specific rewards, etc.)?

### Phase 2: CODE

1. Create a new branch: `git checkout -b claude/<short-description>`
2. Make changes to the task — rewards, observations, curriculum, hyperparameters in:
   - `src/mjlab_leggy/tasks/leggy_run.py` (task config)
   - `src/mjlab_leggy/leggy/leggy_rewards.py` (custom rewards)
   - `src/mjlab_leggy/leggy/leggy_curriculums.py` (curriculum)
   - `src/mjlab_leggy/leggy/leggy_observations.py` (observations)
   - `src/mjlab_leggy/leggy/leggy_actions.py` (actions)
3. Commit with clear message explaining what and why
4. Push: `git push -u origin claude/<short-description>`

### Phase 3: TRAIN

Use an Agent to run training:

```
Spawn Agent (TRAIN):
  1. Run: scripts/train_remote.sh <branch-name>
  2. Wait for wandb run to appear
  3. Return the wandb run path
```

After launch, notify Discord:
```bash
scripts/notify_discord.sh "**Training Started**\nBranch: <branch>\nRun: <wandb-url>\nGoal: <what we're trying to achieve>"
```

### Phase 4: MONITOR + EVALUATE

**Initialize session state** in `logs/training_session/session_state.json`:
```json
{
  "goal": "<user's goal>",
  "branch": "claude/<name>",
  "current_run": 1,
  "wandb_run_path": "<path>",
  "phase": "MONITOR",
  "monitor_count": 0,
  "consecutive_bad": 0,
  "iterations": []
}
```

**Monitoring loop** (repeat every 30+ minutes):

1. Spawn a MONITOR Agent:
   ```
   Run: uv run scripts/monitor_wandb.py <run_path> [--previous <last_monitor_file>]
   Save output to: logs/training_session/run_NNN/monitor_MMM.md
   ```

2. Read the monitor summary. Decide:
   - **Too early**: Metrics just starting, wait longer
   - **Progressing well**: Wait for next check
   - **Concerning**: Run an evaluation to confirm
   - **Converged**: Run final evaluation

3. When evaluation is needed, spawn an EVALUATE Agent:
   ```
   Run: uv run --active mjpython scripts/evaluate_policy.py <run_path> --output-dir logs/training_session/run_NNN/
   Save: eval_metrics.md, eval_metrics.json, video.mp4
   ```

4. Send video to Discord:
   ```bash
   scripts/notify_discord.sh "**Eval — Run N, Check M**\n<brief metrics summary>" --file logs/training_session/run_NNN/eval_NNN.mp4
   ```

5. After evaluation, decide:
   - **Good**: Keep training or finish
   - **Bad (1st time)**: Note concern, wait for next check
   - **Bad (2nd consecutive)**: Kill training, move to ITERATE

**To kill a training:**
```bash
ssh lerobot "screen -S leggy-train -X stuff $'\003'"
```

### Phase 5: ITERATE

1. Analyze what went wrong from monitor + eval files
2. Update code on the same branch (or create a new commit)
3. Commit and push
4. Notify Discord: `"**Relaunching** — Changed X because Y"`
5. Go back to Phase 3: TRAIN
6. Update `session_state.json` with iteration details

**After 10 iterations**: Write `logs/training_session/summary.md` with:
- What was tried in each run
- What worked and what didn't
- Current best result
Send summary to Discord and ask user for guidance.

### Phase 6: FINISH

When a policy looks good:
1. Run final evaluation with video
2. Send to Discord: `"**Training Complete** — <summary of result>"`
3. Ask user to validate on Discord
4. Wait for user feedback

## Decision Rules

- **Minimum wait**: 30 min between wandb checks (longer is OK early on)
- **Kill threshold**: 2 consecutive bad evaluations
- **Max iterations**: 10 runs before escalating to user
- **When in doubt**: Ask user on Discord and wait

## Context Management

This skill runs in long sessions. To avoid context overload:

- **Delegate to subagents**: Each TRAIN, MONITOR, EVALUATE, NOTIFY operation spawns a fresh agent
- **File-based state**: All results go to `logs/training_session/` — agents read/write files, not conversation context
- **Session recovery**: If conversation restarts, read `session_state.json` to resume

## Key Reference

- **Train command** (on lerobot): `uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048`
- **Play command** (on mac): `uv run --active mjpython scripts/evaluate_policy.py <run_path>`
- **WandB project**: `rabault-nicolas-leggy/mjlab`
- **Actuator order**: `[LhipY, LhipX, Lknee, RhipY, RhipX, Rknee]`
- **Motor-to-knee**: `knee = motor - hipX`
```

- [ ] **Step 2: Verify the skill is detected**

Run: `ls ~/.claude/skills/rl-training/SKILL.md`
Expected: File exists.

Note: The skill file lives outside the repo (`~/.claude/skills/`) so it's not committed. Scripts were already committed in Tasks 1-4.

---

### Task 6: Session State Initialization

**Files:**
- Create: `scripts/init_session.sh`

- [ ] **Step 1: Create session init script**

```bash
#!/usr/bin/env bash
# Initialize a training session directory.
# Usage: scripts/init_session.sh "<goal>" "<branch>"

set -euo pipefail

GOAL="$1"
BRANCH="$2"

SESSION_DIR="logs/training_session"
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

- [ ] **Step 2: Make executable and test**

Run: `chmod +x scripts/init_session.sh && scripts/init_session.sh "test goal" "claude/test" && cat logs/training_session/session_state.json`
Expected: Valid JSON with goal and branch fields.

- [ ] **Step 3: Clean up test and commit**

Run: `rm -rf logs/training_session`

```bash
git add scripts/init_session.sh
git commit -m "feat: add session state initialization script"
```

---

### Task 7: End-to-End Validation

- [ ] **Step 1: Verify all scripts are executable**

Run: `ls -la scripts/*.sh scripts/*.py`
Expected: All scripts have execute permission, all files exist.

- [ ] **Step 2: Test notify_discord.sh**

Run: `scripts/notify_discord.sh "**Validation** — All scripts installed and working.\nProject: leggySim"`
Expected: Message in Discord.

- [ ] **Step 3: Test monitor_wandb.py against a real run**

Run: `uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/xdvcvih3`
Expected: Formatted markdown output with metrics.

- [ ] **Step 4: Test evaluate_policy.py (if a checkpoint is available)**

Run: `uv run --active mjpython scripts/evaluate_policy.py rabault-nicolas-leggy/mjlab/xdvcvih3 --output-dir /tmp/eval_validation`
Expected: Video saved, metrics printed.

- [ ] **Step 5: Test full Discord flow with video**

Run: `scripts/notify_discord.sh "**Validation Complete** — eval video attached" --file /tmp/eval_validation/*.mp4`
Expected: Video in Discord.

- [ ] **Step 6: Verify skill file**

Run: `cat ~/.claude/skills/rl-training/SKILL.md | head -5`
Expected: Shows frontmatter with name and description.

- [ ] **Step 7: Clean up validation artifacts**

Run: `rm -rf /tmp/eval_test /tmp/eval_validation /tmp/monitor_check1.md`
