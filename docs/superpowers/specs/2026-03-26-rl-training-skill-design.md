# RL Training Management Skill — Design Spec

## Goal

A Claude Code skill that allows Claude to autonomously manage RL training for leggySim: modify tasks, launch training on a remote GPU server, monitor wandb curves, evaluate policies, iterate, and keep the user informed via Discord.

## Scope

- Specific to leggySim project (not generic)
- Handles the full loop: clarify → code → train → monitor → evaluate → iterate → report
- Up to 10 training runs per user request before escalating
- Long-running sessions with context management via subagents and file-based state

## Infrastructure

| Resource | Details |
|----------|---------|
| Dev machine | Mac M1, code editing and policy evaluation |
| GPU server | Linux + RTX 4090, accessed via `sft ssh lerobot` then `ssh lerobot` |
| leggySim path (lerobot) | `~/leggySim` |
| WandB | Entity: `rabault-nicolas-leggy`, Project: `mjlab`, API via `wandb.Api()` |
| Discord | Webhook in `.claude-discord.md`, for updates, videos, and blockers |
| Git | Separate branch per user request, commits between iterations |

## Workflow

```
User Request ("make Leggy run faster")
    │
    ├─ 1. CLARIFY — Ask questions until goal is crystal clear
    │
    ├─ 2. CODE — Create branch, modify task/rewards/curriculum/obs
    │     └─ Commit & push
    │
    ├─ 3. TRAIN — SSH to lerobot, pull branch, launch in screen
    │     └─ Discord: "Training started — branch: X, run: Y"
    │
    ├─ 4. MONITOR — Poll wandb every 30+ min
    │     ├─ Check key curves (rewards, terminations, curriculum)
    │     ├─ When curves suggest enough progress (or concern):
    │     │     └─ EVALUATE latest checkpoint locally (headless)
    │     │         ├─ Log metrics (velocity tracking, gait, contacts)
    │     │         ├─ Record video, send to Discord
    │     │         └─ Decide: keep training / kill & iterate
    │     ├─ If bad over 2+ consecutive checks+evals → kill → ITERATE
    │     └─ If converged + eval looks good → FINISH
    │
    ├─ 5. ITERATE (up to 10 runs)
    │     ├─ Analyze what went wrong from wandb + eval metrics
    │     ├─ Tweak code, commit, push, retrain
    │     └─ Discord: "Relaunching — changed X because Y"
    │
    └─ 6. REPORT — After success or 10 runs
          ├─ Discord: summary + video of best policy
          └─ Ask user for feedback & wait
```

### Decision Rules

- **Minimum wait before first check**: 30 minutes (can wait longer if early curriculum stages)
- **Kill a run**: Only if metrics look bad over 2+ consecutive observations
- **Max iterations**: 10 runs, then report to user with summary
- **User escalation**: When in doubt or struggling, ask user on Discord and wait for response

## Helper Scripts

### `scripts/train_remote.sh <branch>`
1. Check if `sft` tunnel is active, open it if not (`sft ssh lerobot &`, wait, then `ssh lerobot`)
2. `ssh lerobot` — check GPU usage via `nvidia-smi`, abort if heavy usage detected
3. `cd ~/leggySim && git fetch && git checkout <branch> && git pull`
4. `uv sync`
5. Attach to existing screen session or create one named `leggy-train`
6. Inside screen: `uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048`
7. Parse and output the wandb run path from training logs

### `scripts/monitor_wandb.py <run_path> [--previous <file>]`
- Fetch latest metrics from wandb API: `Episode_Reward/*`, `Train/*`, `Loss/*`, `Torque/*`, `Curriculum/*`, `Episode_Termination/*`, `Metrics/*`
- Compare with previous check if `--previous` provided (trend: improving/stable/degrading)
- Output concise markdown summary to stdout
- Report training status: running / finished / crashed
- Report latest checkpoint iteration number

### `scripts/evaluate_policy.py <run_path> [--output-dir <dir>]`
- Run policy headless using `--wandb-run-path` (loads latest checkpoint automatically) with `render_mode="rgb_array"` + `VideoRecorder`
- Apply a set of velocity commands (forward, lateral, turning, mixed) across multiple episodes
- Log metrics: actual vs commanded velocity, contact patterns, gait symmetry, step frequency, stability (fall rate)
- Save mp4 video and metrics summary markdown
- Output: paths to video file and metrics file

### `scripts/notify_discord.sh <message> [--file <path>]`
- Read webhook URL from `.claude-discord.md`
- Send text message via curl
- If `--file` provided, upload file (video) as attachment via multipart form
- All messages prefixed with `Project: leggySim`

## Context Management

### Orchestrator Pattern

The main conversation acts as a lightweight orchestrator. Heavy work is delegated to subagents:

| Agent | Responsibility | Input | Output |
|-------|---------------|-------|--------|
| CODE | Make code changes | Goal + eval feedback | Committed branch |
| TRAIN | SSH + launch training | Branch name | Run path |
| MONITOR | Poll wandb | Run path | `monitor_XXX.md` |
| EVALUATE | Run policy + record | Run path | `eval_XXX.md` + video |
| NOTIFY | Discord messages/files | Message + optional file | Sent |

Each agent is disposable — spawned fresh with a focused prompt, results written to files.

### File-Based State: `logs/training_session/`

```
logs/training_session/
├── session_state.json        # Current state machine
├── run_001/
│   ├── changes.md            # What was changed and why
│   ├── monitor_001.md        # Wandb check summaries
│   ├── monitor_002.md
│   ├── eval_001.md           # Evaluation metrics
│   └── eval_001.mp4          # Evaluation video
├── run_002/
│   └── ...
└── summary.md                # Running cross-iteration summary
```

#### `session_state.json` schema:
```json
{
  "goal": "Make Leggy run faster",
  "branch": "claude/run-faster",
  "current_run": 3,
  "wandb_run_path": "rabault-nicolas-leggy/mjlab/abc123",
  "phase": "MONITOR",
  "monitor_count": 2,
  "consecutive_bad": 0,
  "iterations": [
    {
      "run": 1,
      "branch": "claude/run-faster",
      "wandb_path": "rabault-nicolas-leggy/mjlab/xyz789",
      "outcome": "killed — reward plateaued early, velocity tracking poor",
      "changes": "Increased track_linear_velocity weight from 2.0 to 3.0"
    }
  ]
}
```

### Recovery

If conversation dies mid-session, a new conversation can:
1. Read `session_state.json` to understand current state
2. Read latest `monitor_*.md` and `eval_*.md` for progress
3. Check wandb run status (still running? finished?)
4. Resume from the correct phase

## Skill File Location

`~/.claude/skills/rl-training/rl-training.md`

## Evaluation Metrics (logged during play)

Per-episode, across multiple velocity commands:
- **Velocity tracking**: RMS error between commanded and actual (x, y, yaw)
- **Stability**: fall rate, episode length distribution
- **Gait quality**: step frequency, contact duty cycle, gait symmetry (left vs right)
- **Torque**: mean, peak, saturation ratio per joint group
- **Smoothness**: action rate, jerk

## Git Convention

- Branch naming: `claude/<short-description>` (e.g. `claude/faster-running`)
- One commit per code change with clear message explaining what and why
- Push before each training run
- Never force push, never push to main
