# Parallel RL Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multiple independent RL training sessions to run in parallel with git worktrees, per-host launch/kill scripts, session isolation by branch name, and fixed Discord notifications.

**Architecture:** Evolve the single-session rl-training skill into a multi-session one. Each session is keyed by branch name, has its own log directory, worktree (local + remote), host assignment, and monitoring cron. Hosts are configured individually with their own launch/kill scripts.

**Tech Stack:** Bash scripts, Python (wandb API), git worktrees, Claude Code crons, Discord webhooks

---

## File Structure

### New files
- `.claude/rl-training/hosts/lerobot/host.md` — host config (migrated from rl_training_infra.md)
- `.claude/rl-training/hosts/lerobot/launch.sh` — SSH + GPU check + worktree + screen launch
- `.claude/rl-training/hosts/lerobot/kill.sh` — kill screen session on this host

### Modified files
- `.claude/rl-training/scripts/init_session.sh` — session dir under `logs/sessions/<branch>/`
- `.claude/rl-training/scripts/notify.sh` — add `--branch` flag
- `.claude/rl-training/scripts/monitor.py` — add `--session-dir` argument
- `.claude/rl-training/scripts/evaluate_policy.py` — add `--output-dir` default change
- `.claude/rl-training/config.md` — add `## Hosts` section
- `.claude/rl-training/SKILL.md` — full rewrite for multi-session architecture
- `~/.claude/projects/.../memory/rl_training_infra.md` — remove host-specific fields (moved to host.md)

### Removed files
- `.claude/rl-training/scripts/train.sh` — replaced by per-host launch.sh
- `.claude/rl-training/scripts/kill_training.sh` — replaced by per-host kill.sh

### Migrated data
- `logs/training_session/` → `logs/sessions/claude--improve-velocity-tracking/`

---

### Task 1: Migrate existing session data

**Files:**
- Move: `logs/training_session/` → `logs/sessions/claude--improve-velocity-tracking/`

- [ ] **Step 1: Create new sessions directory and move data**

```bash
mkdir -p logs/sessions
mv logs/training_session logs/sessions/claude--improve-velocity-tracking
```

- [ ] **Step 2: Verify migration**

```bash
ls logs/sessions/claude--improve-velocity-tracking/session_state.json
```

Expected: file exists

- [ ] **Step 3: Commit**

```bash
git add logs/
git commit -m "refactor: migrate training_session to sessions/claude--improve-velocity-tracking"
```

---

### Task 2: Update init_session.sh for branch-based sessions

**Files:**
- Modify: `.claude/rl-training/scripts/init_session.sh`

- [ ] **Step 1: Rewrite init_session.sh**

Replace the entire file with:

```bash
#!/usr/bin/env bash
# Initialize or reset a training session directory.
# Usage: .claude/rl-training/scripts/init_session.sh "<goal>" "<branch>"
#
# Creates logs/sessions/<branch-sanitized>/ with session_state.json.
# Branch sanitization: / → --

set -euo pipefail

GOAL="$1"
BRANCH="$2"
BRANCH_SANITIZED="${BRANCH//\//--}"

SESSION_DIR="logs/sessions/$BRANCH_SANITIZED"

if [ -f "$SESSION_DIR/session_state.json" ]; then
    BACKUP="$SESSION_DIR/session_state.$(date +%Y%m%d_%H%M%S).json"
    cp "$SESSION_DIR/session_state.json" "$BACKUP"
    echo "Previous state backed up to $BACKUP"
fi

NEXT_RUN=1
if [ -d "$SESSION_DIR" ]; then
    LAST_RUN=$(ls -d "$SESSION_DIR"/run_* 2>/dev/null | sort -V | tail -1 | grep -oE '[0-9]+$' || echo "0")
    NEXT_RUN=$((10#$LAST_RUN + 1))
fi

RUN_DIR=$(printf "run_%03d" "$NEXT_RUN")
mkdir -p "$SESSION_DIR/$RUN_DIR"

jq -n \
  --arg goal "$GOAL" \
  --arg branch "$BRANCH" \
  --arg branch_sanitized "$BRANCH_SANITIZED" \
  --argjson run "$NEXT_RUN" \
  '{goal: $goal, branch: $branch, branch_sanitized: $branch_sanitized, current_run: $run, wandb_run_path: "", host: "", phase: "CODE", monitor_count: 0, consecutive_bad: 0, iterations: []}' \
  > "$SESSION_DIR/session_state.json"

echo "Session initialized at $SESSION_DIR"
echo "Run directory: $SESSION_DIR/$RUN_DIR"
echo "State: $SESSION_DIR/session_state.json"
```

- [ ] **Step 2: Test it**

```bash
bash .claude/rl-training/scripts/init_session.sh "test goal" "claude/test-branch"
cat logs/sessions/claude--test-branch/session_state.json
```

Expected: JSON with `branch_sanitized: "claude--test-branch"`, `host: ""`

- [ ] **Step 3: Clean up test data**

```bash
rm -rf logs/sessions/claude--test-branch
```

- [ ] **Step 4: Commit**

```bash
git add .claude/rl-training/scripts/init_session.sh
git commit -m "refactor: init_session.sh uses branch-based session dirs"
```

---

### Task 3: Update notify.sh with --branch flag and newline fix

**Files:**
- Modify: `.claude/rl-training/scripts/notify.sh`

- [ ] **Step 1: Rewrite notify.sh**

Replace the entire file with:

```bash
#!/usr/bin/env bash
# Send a notification via Discord webhook.
# Usage: .claude/rl-training/scripts/notify.sh "<message>" [--branch <name>] [--file <path>]
#
# --branch: prepends [<branch>] to the message
# --file: attaches a file (e.g. video)
#
# Reads WEBHOOK_URL from .claude-discord.md in the project root.
# Messages are truncated to 2000 chars (Discord limit).
# IMPORTANT: pass actual newlines in the message, not literal \n.
# Use printf or $'...' syntax to compose multi-line messages.

set -euo pipefail

MESSAGE="$1"
shift

FILE=""
BRANCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file) FILE="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -n "$BRANCH" ]; then
    MESSAGE="[$BRANCH] $MESSAGE"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DISCORD_CONFIG="$PROJECT_ROOT/.claude-discord.md"

if [ ! -f "$DISCORD_CONFIG" ]; then
    echo "WARNING: No .claude-discord.md found — skipping notification" >&2
    exit 0
fi

WEBHOOK_URL=$(grep -E "^WEBHOOK_URL:" "$DISCORD_CONFIG" | head -1 | sed 's/^WEBHOOK_URL:[[:space:]]*//')

if [ -z "$WEBHOOK_URL" ]; then
    echo "WARNING: No WEBHOOK_URL found in .claude-discord.md — skipping notification" >&2
    exit 0
fi

MESSAGE="${MESSAGE:0:2000}"

if [ -n "$FILE" ] && [ -f "$FILE" ]; then
    curl -s \
        -F "payload_json={\"content\":$(printf '%s' "$MESSAGE" | jq -Rs .)}" \
        -F "file=@$FILE" \
        "$WEBHOOK_URL" > /dev/null
else
    curl -s -H "Content-Type: application/json" \
        -d "{\"content\":$(printf '%s' "$MESSAGE" | jq -Rs .)}" \
        "$WEBHOOK_URL" > /dev/null
fi
```

- [ ] **Step 2: Test with actual newlines**

```bash
bash .claude/rl-training/scripts/notify.sh "$(printf 'Test notification\nLine 2\nLine 3')" --branch "claude/test"
```

Expected: Discord message shows `[claude/test] Test notification` with actual line breaks

- [ ] **Step 3: Commit**

```bash
git add .claude/rl-training/scripts/notify.sh
git commit -m "fix: notify.sh --branch flag and newline documentation"
```

---

### Task 4: Create per-host directory for lerobot

**Files:**
- Create: `.claude/rl-training/hosts/lerobot/host.md`
- Create: `.claude/rl-training/hosts/lerobot/launch.sh`
- Create: `.claude/rl-training/hosts/lerobot/kill.sh`

- [ ] **Step 1: Create host.md**

```markdown
# Host: lerobot
- Type: direct
- SSH: lerobot
- Remote dir: ~/leggySim
- Tunnel: sft ssh lerobot
- GPU check: nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
- GPU threshold: 50
- PATH setup: export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
- Dependencies: uv sync
```

- [ ] **Step 2: Create launch.sh**

```bash
#!/usr/bin/env bash
# Launch training on lerobot (direct SSH + GPU + screen).
# Usage: launch.sh <branch> <branch-sanitized> <train-command> [<deps-command>]
# Exit 0: training started. Exit 1: host unavailable.
# Reads host config from host.md in the same directory.

set -euo pipefail

BRANCH="$1"
BRANCH_SANITIZED="$2"
TRAIN_CMD="$3"
DEPS_CMD="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Read host config
SSH_HOST=$(grep "^- SSH:" "$SCRIPT_DIR/host.md" | sed 's/^- SSH:[[:space:]]*//')
REMOTE_DIR=$(grep "^- Remote dir:" "$SCRIPT_DIR/host.md" | sed 's/^- Remote dir:[[:space:]]*//')
TUNNEL=$(grep "^- Tunnel:" "$SCRIPT_DIR/host.md" | sed 's/^- Tunnel:[[:space:]]*//')
GPU_CHECK=$(grep "^- GPU check:" "$SCRIPT_DIR/host.md" | sed 's/^- GPU check:[[:space:]]*//')
GPU_THRESHOLD=$(grep "^- GPU threshold:" "$SCRIPT_DIR/host.md" | sed 's/^- GPU threshold:[[:space:]]*//')
PATH_SETUP=$(grep "^- PATH setup:" "$SCRIPT_DIR/host.md" | sed 's/^- PATH setup:[[:space:]]*//')
HOST_DEPS=$(grep "^- Dependencies:" "$SCRIPT_DIR/host.md" | sed 's/^- Dependencies:[[:space:]]*//')

SCREEN_NAME="leggy-$BRANCH_SANITIZED"
WORKTREE_DIR="${REMOTE_DIR%/*}/leggySim-wt-$BRANCH_SANITIZED"

echo "=== Checking SSH connection to $SSH_HOST ==="
if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
    if [ -n "$TUNNEL" ]; then
        echo "SSH failed, trying tunnel: $TUNNEL"
        eval "$TUNNEL" &
        TUNNEL_PID=$!
        sleep 8
        if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
            kill "$TUNNEL_PID" 2>/dev/null || true
            echo "ERROR: Cannot connect to $SSH_HOST even with tunnel" >&2
            exit 1
        fi
    else
        echo "ERROR: Cannot connect to $SSH_HOST" >&2
        exit 1
    fi
fi

if [ -n "$GPU_CHECK" ]; then
    echo "=== Checking GPU usage ==="
    GPU_MAX=$(ssh "$SSH_HOST" "$GPU_CHECK" | awk '{if($1+0 > max) max=$1+0} END{print max}')
    echo "GPU max utilization: ${GPU_MAX}%"
    if [ "$GPU_MAX" -gt "$GPU_THRESHOLD" ]; then
        echo "ERROR: GPU utilization ${GPU_MAX}% > threshold ${GPU_THRESHOLD}%" >&2
        exit 1
    fi
fi

echo "=== Setting up remote worktree ==="
ssh "$SSH_HOST" "cd $REMOTE_DIR && git fetch origin && (git worktree add $WORKTREE_DIR $BRANCH 2>/dev/null || (cd $WORKTREE_DIR && git checkout $BRANCH && git pull origin $BRANCH))"

if [ -n "${DEPS_CMD:-$HOST_DEPS}" ]; then
    ACTUAL_DEPS="${DEPS_CMD:-$HOST_DEPS}"
    echo "=== Syncing dependencies ==="
    if [ -n "$PATH_SETUP" ]; then
        ssh "$SSH_HOST" "$PATH_SETUP && cd $WORKTREE_DIR && $ACTUAL_DEPS"
    else
        ssh "$SSH_HOST" "cd $WORKTREE_DIR && $ACTUAL_DEPS"
    fi
fi

echo "=== Launching training in screen $SCREEN_NAME ==="
ssh "$SSH_HOST" "screen -ls | grep -q '$SCREEN_NAME' && screen -S '$SCREEN_NAME' -X stuff \$'\\003' || screen -dmS '$SCREEN_NAME'"
sleep 1

FULL_CMD="cd $WORKTREE_DIR && $TRAIN_CMD"
[ -n "$PATH_SETUP" ] && FULL_CMD="$PATH_SETUP && $FULL_CMD"
ssh "$SSH_HOST" "screen -S '$SCREEN_NAME' -X stuff '$FULL_CMD\n'"

echo "=== Training launched on $SSH_HOST ==="
echo "Worktree: $WORKTREE_DIR"
echo "Screen: $SCREEN_NAME"
echo "Monitor with: ssh $SSH_HOST 'screen -r $SCREEN_NAME'"
```

- [ ] **Step 3: Create kill.sh**

```bash
#!/usr/bin/env bash
# Kill training on lerobot for a given session.
# Usage: kill.sh <branch-sanitized>
# Reads host config from host.md in the same directory.

set -euo pipefail

BRANCH_SANITIZED="$1"
SCREEN_NAME="leggy-$BRANCH_SANITIZED"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_HOST=$(grep "^- SSH:" "$SCRIPT_DIR/host.md" | sed 's/^- SSH:[[:space:]]*//')

echo "Sending Ctrl-C to $SCREEN_NAME on $SSH_HOST..."
ssh "$SSH_HOST" "screen -S '$SCREEN_NAME' -X stuff \$'\\003'"
sleep 2

if ssh "$SSH_HOST" "screen -ls | grep -q '$SCREEN_NAME'" 2>/dev/null; then
    echo "Screen still exists, sending quit..."
    ssh "$SSH_HOST" "screen -S '$SCREEN_NAME' -X quit" 2>/dev/null || true
fi

echo "Training killed on $SSH_HOST."
```

- [ ] **Step 4: Make scripts executable**

```bash
chmod +x .claude/rl-training/hosts/lerobot/launch.sh .claude/rl-training/hosts/lerobot/kill.sh
```

- [ ] **Step 5: Commit**

```bash
git add .claude/rl-training/hosts/
git commit -m "feat: add per-host scripts for lerobot"
```

---

### Task 5: Remove old train.sh and kill_training.sh

**Files:**
- Remove: `.claude/rl-training/scripts/train.sh`
- Remove: `.claude/rl-training/scripts/kill_training.sh`

- [ ] **Step 1: Remove old scripts**

```bash
rm .claude/rl-training/scripts/train.sh .claude/rl-training/scripts/kill_training.sh
```

- [ ] **Step 2: Commit**

```bash
git add -u .claude/rl-training/scripts/
git commit -m "refactor: remove train.sh and kill_training.sh (replaced by per-host scripts)"
```

---

### Task 6: Update monitor.py with --session-dir

**Files:**
- Modify: `.claude/rl-training/scripts/monitor.py`

- [ ] **Step 1: Update monitor.py**

Change the argparse section and remove any hardcoded paths. The script already takes a run_path and outputs to stdout, so the only change is documenting that the caller is responsible for saving output to the correct session directory. No hardcoded path exists in monitor.py currently — verify and confirm no changes needed.

Actually, reading the code again: monitor.py has no hardcoded `logs/training_session` path. It takes `run_path` as argument, outputs to stdout, and the caller redirects. No change needed.

- [ ] **Step 2: Verify no hardcoded paths**

```bash
grep -n "training_session" .claude/rl-training/scripts/monitor.py
```

Expected: no matches

---

### Task 7: Update evaluate_policy.py default output dir

**Files:**
- Modify: `.claude/rl-training/scripts/evaluate_policy.py:175`

- [ ] **Step 1: Change default output-dir**

In `evaluate_policy.py` line 175, change:
```python
    parser.add_argument("--output-dir", default="logs/training_session/eval")
```
to:
```python
    parser.add_argument("--output-dir", required=True, help="Session run directory for output")
```

The caller (cron prompt) must now always pass `--output-dir` explicitly pointing to the correct session's run directory.

- [ ] **Step 2: Commit**

```bash
git add .claude/rl-training/scripts/evaluate_policy.py
git commit -m "refactor: evaluate_policy.py requires explicit --output-dir"
```

---

### Task 8: Update config.md with Hosts section

**Files:**
- Modify: `.claude/rl-training/config.md`

- [ ] **Step 1: Add Hosts section to config.md**

Add after the `## Training` section:

```markdown
## Hosts
Order: [lerobot]
```

- [ ] **Step 2: Remove screen name from Training section**

The screen name is now derived from branch name (`leggy-<branch-sanitized>`), so remove `- Screen name: leggy` from the Training section.

- [ ] **Step 3: Commit**

```bash
git add .claude/rl-training/config.md
git commit -m "refactor: add Hosts section to config.md, remove fixed screen name"
```

---

### Task 9: Update rl_training_infra.md memory

**Files:**
- Modify: `~/.claude/projects/-Users-nicolasrabault-Projects-LeParkour-leggySim/memory/rl_training_infra.md`

- [ ] **Step 1: Remove host-specific fields**

Keep only WandB info. Remove SSH host, remote path, tunnel command, GPU check, screen session name, PATH setup — all migrated to `.claude/rl-training/hosts/lerobot/host.md`.

New content:
```markdown
---
name: rl_training_infra
description: WandB project and monitoring details for RL training
type: project
---

## WandB
- Project: rabault-nicolas-leggy/mjlab
- Dashboard: https://wandb.ai/rabault-nicolas-leggy/mjlab
```

- [ ] **Step 2: No commit needed** (memory files are not in git)

---

### Task 10: Rewrite SKILL.md for multi-session architecture

**Files:**
- Modify: `~/.claude/skills/rl-training/SKILL.md`

This is the largest task. The SKILL.md needs to be rewritten to support:
- Multiple parallel sessions keyed by branch name
- Per-host launch/kill scripts in `.claude/rl-training/hosts/<name>/`
- Git worktrees on local and remote
- Session dirs under `logs/sessions/<branch-sanitized>/`
- Screen names: `leggy-<branch-sanitized>`
- Notifications with `--branch` flag
- Recovery that scans all sessions

- [ ] **Step 1: Rewrite SKILL.md**

Full replacement — the new SKILL.md content:

```markdown
---
name: rl-training
description: Manage RL training loops — modify tasks, launch training, monitor metrics, evaluate policies, iterate. Use when user asks to train, improve, or create an RL task. Use when user asks to "make the robot do X" or "improve the gait" or "train a new behavior". Also use when user wants to resume training, check training status, kill a run, review training metrics, or says "the training looks bad". Works with any RL project after setup.
---

# RL Training Manager

Autonomously manage the full RL training loop using a multi-agent architecture. Supports multiple parallel training sessions, each on its own branch, worktree, and host. Reads project-specific configuration from `.claude/rl-training/config.md`.

## Architecture

```
Main (this conversation)
  ├─ Phase 0: SETUP (if no config exists)
  ├─ Phase 1-2: Clarify + Code (directly, in a local worktree)
  ├─ Phase 3: Spawn Run Agent (foreground) → starts training, returns run path
  ├─ Creates ONE recurring durable cron per session (every ~30 min)
  └─ Done — main conversation ends or waits

Recurring Monitor Cron (per session, fires every ~30 min):
  phase=MONITOR → fetch metrics, eval, notify, decide keep/kill
  phase=ITERATE → diagnose, fix code in worktree, relaunch training
  phase=FINISHED → do nothing, exit
  phase=PAUSED → do nothing, exit
```

**Sessions are keyed by branch name.** Each session has:
- Local worktree: `../leggySim-wt-<branch-sanitized>`
- Remote worktree: `<remote-parent>/leggySim-wt-<branch-sanitized>`
- Logs: `logs/sessions/<branch-sanitized>/`
- Screen: `leggy-<branch-sanitized>`
- Its own recurring monitoring cron

**Branch sanitization:** `/` → `--` (e.g. `claude/improve-gait` → `claude--improve-gait`)

**Communication is file-based** — all agents read/write `logs/sessions/<branch-sanitized>/`:
- `session_state.json` — global state (phase, goal, run#, host, branch, etc.)
- `run_NNN/context.md` — what was changed and why (written before launch)
- `run_NNN/monitor_MMM.md` — monitoring output
- `run_NNN/result.md` — final summary when run is killed

## Prerequisites

- Project config at `.claude/rl-training/config.md` (generated by SETUP phase if missing)
- Per-host scripts at `.claude/rl-training/hosts/<name>/` (host.md, launch.sh, kill.sh)
- Infra memory at `~/.claude/projects/<hash>/memory/rl_training_infra.md` (WandB details)
- Project-local scripts at `.claude/rl-training/scripts/`

## Phase 0: SETUP

**Trigger**: `.claude/rl-training/config.md` does not exist, OR user explicitly asks to update training config.

If config exists and user wants to update, show current config sections and ask which to update. Only regenerate affected scripts.

### Step 1: Codebase Exploration (autonomous)

No user interaction. Scan the project:
- Read project structure, dependencies (`pyproject.toml`, `setup.py`, `requirements.txt`), entry points
- Identify: simulator, RL framework, algorithm, task names
- Look at existing training scripts, config files, reward definitions
- Produce a findings summary for the next steps

### Step 2: Robot & Objective (interactive)

Present findings from Step 1. Ask user to confirm or correct. Then ask:
- Robot type and key physical traits / constraints
- Actuated joints and any special mechanics
- Training objective — what should the robot learn?
- One question at a time.

### Step 3: Training Hosts (interactive)

Ask user to add hosts one by one. For each host:
- Name (identifier, e.g. `lerobot`, `cluster1`, `local`)
- Type: direct SSH, SLURM cluster, or local
- Connection details (SSH alias, remote dir, tunnel command)
- GPU check command and threshold
- PATH setup if needed
- Cluster-specific params if applicable (partition, GPU type, allocation command)
- Dependencies command for this host

Generate per-host directory `.claude/rl-training/hosts/<name>/`:
- `host.md` — host config
- `launch.sh` — availability check + worktree setup + training launch
- `kill.sh` — stop training for a session

Ask "Add another host?" until done.

Write `## Hosts` section in config.md with ordered host list.

### Step 4: Monitoring & Evaluation (interactive)

- What monitoring tool? (WandB, TensorBoard, local logs)
- If WandB: project path → store in `rl_training_infra.md`
- Key metric categories and prefixes to track
- What does "good" vs "bad" look like for this task?
- Evaluation: what scenarios to test, what metrics, record video?
- May iterate with user to refine eval strategy

### Step 5: Notifications (interactive)

- Does the user want notifications about training progress?
- If yes: ask how — Discord webhook, Slack webhook, email, custom?
  - Look at existing project config files (e.g., `.claude-discord.md`) for webhook URLs
  - Store any credentials/webhooks in `rl_training_infra.md`
- Which events trigger notifications? (training_started, monitor_update, eval_complete, training_killed, iteration_started, blocker)
- **Always generate `.claude/rl-training/scripts/notify.sh`** — cron agents cannot invoke Claude Code skills, so all notification delivery must go through a bash script.

### Step 6: Generate (autonomous)

- Write `.claude/rl-training/config.md` from gathered info
- Write `rl_training_infra.md` to project memory (WandB details only)
- Generate shared scripts in `.claude/rl-training/scripts/`:
  - `init_session.sh` — session state management (branch-based)
  - `get_latest_run.py` — find active run (WandB or other)
  - `monitor.py` — fetch metrics and format markdown report
  - `evaluate_policy.py` — headless eval with video and metrics
  - `notify.sh` — notification delivery with `--branch` support
- Per-host scripts already generated in Step 3
- Make all scripts executable
- Present summary of generated files to user for review

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
- Previous run results if available (`logs/sessions/`)
- Write findings in `logs/sessions/<branch-sanitized>/run_NNN/analysis.md`

### Step 2: RESEARCH

Search the web for RL solutions relevant to the task if needed. Write approach in `analysis.md`.

Common RL issues to consider when diagnosing problems:
- **Reward scale**: rewards too large cause instability, too small cause slow learning. Check reward weights relative to each other — a single dominant reward term drowns out others.
- **Observation normalization**: unnormalized observations (especially velocities, positions) can destabilize training. Check if the framework provides running normalization.
- **Curriculum pacing**: if the curriculum advances too fast, the policy collapses. If too slow, the policy never sees harder conditions. Check curriculum thresholds vs actual performance.
- **Action space**: clipping too tight limits expressiveness, too loose causes jitter. Check action scale and limits against the robot's physical range.
- **Termination conditions**: overly aggressive terminations (e.g., small tilt → reset) prevent the policy from learning recovery. Check if the robot has room to explore.
- **Reward conflicts**: contradictory reward terms (e.g., "move fast" + "minimize energy") need careful balancing. Look for terms that fight each other.

### Step 3: IMPLEMENT

1. Branch: `git checkout -b claude/<short-description>`
2. Create local worktree: `git worktree add ../leggySim-wt-<branch-sanitized> <branch>`
3. Make targeted changes in the worktree directory
4. No dead code, no "just in case" code
5. Commit from the worktree: `cd ../leggySim-wt-<branch-sanitized> && git add <files> && git commit -m "<what and why>"`
6. Push: `git push -u origin <branch>`

## Phase 3: LAUNCH

Read `.claude/rl-training/config.md` and infra memory (`rl_training_infra.md`) for all parameters.

### Step 1: Initialize session

```bash
bash .claude/rl-training/scripts/init_session.sh "<goal>" "<branch>"
```

This creates `logs/sessions/<branch-sanitized>/` with `session_state.json`.

### Step 2: Spawn Run Agent

Use the Agent tool (foreground). The agent:

```
You are a training launcher. Do these steps in order:

1. Read config and host info:
   - .claude/rl-training/config.md → task name, training command, host order
   - rl_training_infra.md from project memory → wandb project
   - Read host.md for each host in the order listed in config.md

2. Select a host:
   Read host order from config.md → Hosts section.
   For each host in order, run its launch.sh:
     bash .claude/rl-training/hosts/<host>/launch.sh "<branch>" "<branch-sanitized>" "<train-command>" ["<deps-command>"]
   If exit 0: host found. If exit 1: try next host.
   If no host available: notify "No host available" via notify.sh and exit with error.

3. Update logs/sessions/<branch-sanitized>/session_state.json:
   - Set host to the selected host identifier
   - Set phase to "MONITOR"
   - Set monitor_count to 0
   - Set consecutive_bad to 0

4. Wait for run to appear:
   uv run .claude/rl-training/scripts/get_latest_run.py <wandb-project> --state running --wait 120
   Save the output path.

5. Update session_state.json:
   - Set wandb_run_path to the path from step 4

6. Send notification (if enabled in config):
   bash .claude/rl-training/scripts/notify.sh "$(printf 'Training Started — Run %d\nBranch: %s\nHost: %s\nGoal: %s' "$RUN" "$BRANCH" "$HOST" "$GOAL")" --branch "<branch>"

7. Return: "Training started on <host>. Run path: <path>."
```

### Step 3: Create the monitoring cron

After the Run Agent returns, the MAIN CONVERSATION creates the cron:

```
CronCreate:
  recurring: true
  durable: true
  cron: "*/33 * * * *"
```

Use the MONITOR CRON PROMPT below as the prompt, with `<BRANCH>` and `<BRANCH_SANITIZED>` replaced. Tell the user: "Training launched on <host>. Monitoring every ~30 min."

---

## MONITOR CRON PROMPT

Copy this as the `prompt` parameter for CronCreate, replacing `<BRANCH>` and `<BRANCH_SANITIZED>` with actual values.

```
You are the autonomous training loop for session <BRANCH>.

STEP 0: Read project context.
- Read .claude/rl-training/config.md → extract: task name, training command, metric categories, key metrics, kill threshold, max iterations, decision criteria, evaluation config, notification config, source files, host order.
- Read rl_training_infra.md from project memory → extract: wandb project.
- Read logs/sessions/<BRANCH_SANITIZED>/session_state.json. If missing or corrupted, notify via notify.sh --branch "<BRANCH>" "Monitor error — session_state.json missing" and exit.
- Extract: phase, goal, branch, host, current_run, wandb_run_path, monitor_count, consecutive_bad, iterations.
- Read .claude/rl-training/hosts/<host>/host.md for host details.

STEP 1: Act based on phase.

=== IF phase = "FINISHED" or phase = "PAUSED" or phase = "CODE" ===
Do nothing. Exit immediately.

=== IF phase = "MONITOR" ===

1. Determine M = monitor_count + 1. Pad to 3 digits for filenames.
   Session dir: logs/sessions/<BRANCH_SANITIZED>/
   Run directory: <session_dir>/run_{current_run padded to 3 digits}/
   Previous monitor: run_NNN/monitor_{M-1 padded}.md (if M > 1)

2. Fetch metrics:
   Read config.md → Monitoring.Tool and Monitoring.Metric categories.
   uv run .claude/rl-training/scripts/monitor.py <wandb_run_path> [--previous <prev_monitor>] [--categories <from config>]
   Save output to: run_NNN/monitor_{M padded}.md
   Check exit code: if exit code is 2, the run has crashed/been killed externally — treat as BAD immediately (skip to KILL in step 6).
   If this fails for other reasons, notify via notify.sh --branch "<BRANCH>" "Monitor error — <error>" and exit.

3. Evaluate policy (if config says Video: true or Evaluation section exists):
   uv run .claude/rl-training/scripts/evaluate_policy.py <wandb_run_path> --output-dir run_NNN/ --config .claude/rl-training/config.md
   If eval fails, continue without video.

4. Send notification (if enabled and monitor_update in When list):
   Compose message with actual newlines using printf:
   MSG=$(printf 'Monitor %d — Run %d (step %s)\n%s\nTrend: %s\nProject: leggySim' "$M" "$RUN" "$STEP" "$KEY_METRICS" "$TREND")
   bash .claude/rl-training/scripts/notify.sh "$MSG" --branch "<BRANCH>" [--file run_NNN/rl-video-step-0.mp4]

5. DECIDE by comparing current key metrics against the previous monitor file:
   Compare each key metric (from config.md → Key metrics) to its previous value.
   If config.md has a "Decision criteria" section, use those thresholds. Otherwise use these defaults:
   - **KEEP**: at least one key metric improved OR all are stable (changed < 5%) AND training step is advancing
   - **FINISH**: main reward metric plateaued (< 2% change over last 3 monitors) AND eval metrics are acceptable (low error, few falls)
   - **BAD**: main reward metric degraded > 10% from its peak, OR key tracking metrics worsening for 2+ consecutive monitors, OR training step hasn't advanced (stalled)
   When in doubt between KEEP and BAD, choose KEEP — false kills waste more time than extra monitoring.

6. ACT:

   If KEEP:
   - Update session_state.json: monitor_count = M, consecutive_bad = 0

   If BAD (consecutive_bad < kill_threshold - 1):
   - Update session_state.json: monitor_count = M, consecutive_bad += 1

   If BAD (consecutive_bad >= kill_threshold - 1) → KILL:
   - Kill training: bash .claude/rl-training/hosts/<host>/kill.sh <BRANCH_SANITIZED>
   - Write run_NNN/result.md with: goal, kill step, metrics at death, eval results, trend, assessment.
   - Update session_state.json:
     - phase: "ITERATE"
     - Add to iterations: {run: N, result: "<one-line>"}
     - Increment current_run
     - mkdir -p logs/sessions/<BRANCH_SANITIZED>/run_{new N padded}/
   - If current_run > max_iterations from config: notify via notify.sh --branch "<BRANCH>" "Max iterations reached — need user guidance", set phase: "PAUSED". Exit.
   - Notify via notify.sh --branch "<BRANCH>": "Run N killed — <reason>. Will iterate on next cron fire."

   If FINISH:
   - Notify via notify.sh --branch "<BRANCH>": "Training Complete — Run N\n<summary>"
   - Update session_state.json: phase: "FINISHED"

=== IF phase = "ITERATE" ===

A training run was killed. Diagnose, fix code, relaunch.

1. Read run_{previous_run}/result.md and run_{previous_run}/context.md from logs/sessions/<BRANCH_SANITIZED>/
2. Read iterations array to understand what was tried before.

3. DIAGNOSE — read source files listed in config.md → Source Files section.
   Work in the local worktree: ../leggySim-wt-<BRANCH_SANITIZED>
   Search the web if the problem is non-obvious.
   Use robot specificities from config.md → Robot section for informed decisions.

4. Write diagnosis: logs/sessions/<BRANCH_SANITIZED>/run_{current_run}/analysis.md

5. Before making changes, check for uncommitted work in the worktree:
   Run: cd ../leggySim-wt-<BRANCH_SANITIZED> && git status --porcelain
   If there are uncommitted changes not from this agent, set phase to "PAUSED", notify via notify.sh --branch "<BRANCH>" "Git has uncommitted changes — need user to resolve", and exit.

6. Make targeted code changes in ../leggySim-wt-<BRANCH_SANITIZED>. No dead code, no "just in case" code.

7. Commit: cd ../leggySim-wt-<BRANCH_SANITIZED> && git add <files> && git commit -m "<what and why>"
8. Push: cd ../leggySim-wt-<BRANCH_SANITIZED> && git push origin HEAD
   If push fails (e.g., diverged branch), set phase to "PAUSED", notify via notify.sh --branch "<BRANCH>" "Push failed — branch may have diverged", and exit.

9. Write logs/sessions/<BRANCH_SANITIZED>/run_{current_run}/context.md

10. Launch training:
    Read host order from config.md. Try each host's launch.sh in order:
    bash .claude/rl-training/hosts/<host>/launch.sh "<BRANCH>" "<BRANCH_SANITIZED>" "<train-cmd>" ["<deps-cmd>"]
    Use first that succeeds. Update session_state.json host field.

11. Get run path:
    uv run .claude/rl-training/scripts/get_latest_run.py <wandb-project> --state running --wait 120

12. Update session_state.json: wandb_run_path, phase: "MONITOR", monitor_count: 0, consecutive_bad: 0

13. Notify via notify.sh --branch "<BRANCH>": printf "Relaunching — Run %d\nHost: %s\nChanged: %s\nExpected: %s" "$RUN" "$HOST" "$SUMMARY" "$EXPECTED"

IMPORTANT:
- Read config.md Robot section for physical constraints and mechanics
- Read config.md Source Files for which files to modify
- Keep code concise, no verbose comments
- Only change what the diagnosis justifies
- Always use --branch "<BRANCH>" when calling notify.sh
- Always use printf with actual newlines for multi-line messages, never literal \n strings
- Work in the local worktree ../leggySim-wt-<BRANCH_SANITIZED>, not the main repo
```

---

## Recovery

If the conversation restarts or the user re-invokes the skill:

1. Check if `.claude/rl-training/config.md` exists. If not, run SETUP.
2. Scan `logs/sessions/*/session_state.json` for all sessions.
3. For each active session (phase MONITOR or ITERATE):
   - Check if a monitoring cron exists (CronList, match by branch name in prompt)
   - If no cron, create one
4. Show summary of all sessions: branch, phase, host, current run, last monitor
5. Based on user intent:
   - View status: show session details
   - New training: proceed to Phase 1 (CLARIFY)
   - Resume specific session: act on its current phase

## Cleanup

When a session reaches FINISHED or user requests cleanup:

1. Remove local worktree: `git worktree remove ../leggySim-wt-<branch-sanitized>`
2. Remove remote worktree: `ssh <host> "cd <remote-dir> && git worktree remove ../leggySim-wt-<branch-sanitized>"`
3. Session logs in `logs/sessions/<branch-sanitized>/` are preserved
4. Branch is preserved (can be checked out again)

## Decision Rules

- **Cron interval**: ~30 min (use odd minutes like */33 to avoid collisions)
- **Kill threshold**: Read from `config.md` → Monitoring.Kill threshold (default: 2)
- **Max iterations**: Read from `config.md` → Monitoring.Max iterations (default: 10)
- **When in doubt**: Send notification and wait for user

## Notification Delivery

Read `config.md` → Notifications section:
- If `Enabled: false` → skip all notifications
- Only notify for events listed in the `When` field
- **Always use the script**: call `.claude/rl-training/scripts/notify.sh "<message>" --branch "<branch>" [--file <path>]`
- **Always use printf or $'...'** to compose multi-line messages — never pass literal \n strings
- The main conversation can also invoke notification skills directly if available, but cron agents must always use the script

## Scripts

Shared scripts in `.claude/rl-training/scripts/`:

| Script | Usage |
|--------|-------|
| `init_session.sh "<goal>" "<branch>"` | Initialize session directory + state |
| `get_latest_run.py <project> [--state S] [--wait N]` | Find active run |
| `monitor.py <run_path> [--previous file] [--categories cats]` | Fetch metrics → markdown |
| `evaluate_policy.py <run_path> --output-dir <dir>` | Headless eval with video + metrics |
| `notify.sh "<message>" [--branch name] [--file path]` | Notification delivery |

Per-host scripts in `.claude/rl-training/hosts/<name>/`:

| Script | Usage |
|--------|-------|
| `launch.sh <branch> <branch-sanitized> <train-cmd> [<deps-cmd>]` | Check availability + launch training |
| `kill.sh <branch-sanitized>` | Kill training for a session |
```

- [ ] **Step 2: Commit SKILL.md**

Note: SKILL.md is outside the repo (in `~/.claude/skills/`), so no git commit needed.

---

### Task 11: Update config.md SETUP instructions for host generation

**Files:**
- Already covered by Task 10 (SKILL.md Phase 0 Step 3)

This task is handled within the SKILL.md rewrite. The SETUP phase in SKILL.md now describes iterative host addition. No separate code needed — the skill agent follows the instructions to generate host directories during setup.

---

### Task 12: Verify end-to-end

- [ ] **Step 1: Verify file structure**

```bash
ls -la .claude/rl-training/hosts/lerobot/
ls -la .claude/rl-training/scripts/
ls -la logs/sessions/
```

Expected:
- `hosts/lerobot/` has host.md, launch.sh, kill.sh
- `scripts/` has init_session.sh, notify.sh, monitor.py, evaluate_policy.py, get_latest_run.py (no train.sh, no kill_training.sh)
- `sessions/` has `claude--improve-velocity-tracking/`

- [ ] **Step 2: Verify init_session.sh works with sanitization**

```bash
bash .claude/rl-training/scripts/init_session.sh "verify test" "test/verify-branch"
cat logs/sessions/test--verify-branch/session_state.json | jq .
rm -rf logs/sessions/test--verify-branch
```

Expected: JSON with `branch_sanitized: "test--verify-branch"`, `host: ""`

- [ ] **Step 3: Verify notify.sh --branch works**

```bash
bash .claude/rl-training/scripts/notify.sh "$(printf 'Verification test\nMultiple lines\nWorking!')" --branch "test/branch"
```

Expected: Discord shows `[test/branch] Verification test` with actual line breaks

- [ ] **Step 4: Final commit if any loose changes**

```bash
git status
```

If clean, done. If not, commit remaining changes.
