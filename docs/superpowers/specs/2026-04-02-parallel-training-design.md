# Parallel RL Training — Design Spec

## Goal

Enable multiple independent RL training sessions to run in parallel, each on its own branch, worktree, and host. Fix Discord notification formatting. Identify each notification by branch name.

## Architecture Overview

The existing single-session architecture evolves into a multi-session one. Each session is independent: own branch, own worktree (local + remote), own host, own cron, own log directory. No central orchestrator — sessions don't coordinate with each other.

```
Main conversation
  ├─ Session A (branch: claude/improve-gait)
  │   ├─ Local worktree: ../leggySim-wt-claude--improve-gait
  │   ├─ Remote worktree: ~/leggySim-wt-claude--improve-gait @ lerobot
  │   ├─ Logs: logs/sessions/claude--improve-gait/
  │   ├─ Screen: leggy-claude--improve-gait
  │   └─ Recurring cron (monitor/iterate)
  └─ Session B (branch: claude/add-jumping)
      ├─ Local worktree: ../leggySim-wt-claude--add-jumping
      ├─ Remote worktree: ~/leggySim-wt-claude--add-jumping @ cluster1
      ├─ Logs: logs/sessions/claude--add-jumping/
      ├─ Screen: leggy-claude--add-jumping
      └─ Recurring cron (monitor/iterate)
```

## 1. Host Configuration

### Per-host directory structure

```
.claude/rl-training/hosts/
├── lerobot/
│   ├── host.md          # config: ssh alias, remote dir, tunnel, type=direct
│   ├── launch.sh        # SSH + GPU check + worktree + screen
│   └── kill.sh          # kill screen session
├── cluster1/
│   ├── host.md          # config: ssh alias, remote dir, type=slurm
│   ├── launch.sh        # SSH + sbatch/salloc + GPU allocation
│   └── kill.sh          # scancel
└── local/
    ├── host.md          # type=local
    ├── launch.sh        # direct launch
    └── kill.sh          # kill process
```

### host.md format

```markdown
# Host: lerobot
- Type: direct (direct | slurm | local)
- SSH: lerobot
- Remote dir: ~/leggySim
- Tunnel: sft ssh lerobot
- GPU check: nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
- GPU threshold: 50
```

### Host ordering

Hosts are tried in directory listing order (alphabetical). `config.md` gets a `## Hosts` section listing preferred order:

```markdown
## Hosts
Order: [lerobot, cluster1, local]
```

### SETUP changes

Step 3 (Training Infrastructure) becomes iterative: ask user to add hosts one by one. For each host, ask:
- Name (identifier)
- Type: direct SSH, SLURM cluster, or local
- Connection details (SSH alias, remote dir, tunnel command)
- Cluster-specific params if applicable (partition, GPU type, allocation command)
- GPU check command and threshold

Generate `host.md`, `launch.sh`, `kill.sh` per host.

`rl_training_infra.md` retains only non-host info: WandB project, dashboard URL.

## 2. Session Isolation

### Directory structure

```
logs/sessions/
├── claude--improve-gait/
│   ├── session_state.json
│   ├── run_001/
│   │   ├── context.md
│   │   ├── analysis.md
│   │   ├── monitor_001.md
│   │   ├── monitor_002.md
│   │   └── result.md
│   └── run_002/
│       └── ...
└── claude--add-jumping/
    ├── session_state.json
    └── run_001/
        └── ...
```

### Branch name sanitization

`/` → `--` (e.g. `claude/improve-gait` → `claude--improve-gait`). Used for directory names, worktree paths, and screen session names.

### session_state.json additions

New fields added to the existing schema:
- `host`: identifier of the machine running this training (e.g. `lerobot`)
- `branch_sanitized`: sanitized branch name for paths

### Migration

Existing `logs/training_session/` content moves to `logs/sessions/<current-branch-sanitized>/`. `init_session.sh` updated to create under `logs/sessions/`.

## 3. Git Worktrees

### Local worktree

Created when launching a new training:

```bash
git worktree add ../leggySim-wt-<branch-sanitized> <branch>
```

Lives as a sibling directory to the main repo. Code changes for a session (Phase 2 and ITERATE phase) happen in the worktree.

### Remote worktree

Created by the host's `launch.sh`:

```bash
ssh <host> "cd <remote-dir> && git fetch origin && git worktree add ../leggySim-wt-<branch-sanitized> <branch>"
```

Training runs from the worktree directory. Dependencies installed there.

### Screen session

Named `leggy-<branch-sanitized>` so multiple trainings on the same host don't collide.

### Cleanup

When a session reaches FINISHED or is manually stopped:

```bash
# Local
git worktree remove ../leggySim-wt-<branch-sanitized>

# Remote
ssh <host> "cd <remote-dir> && git worktree remove ../leggySim-wt-<branch-sanitized>"
```

Branches persist after worktree removal.

## 4. Host Selection

Each host's `launch.sh` handles its own availability check and training launch. The selection loop in the skill:

1. Read host order from `config.md`
2. For each host, run its `launch.sh` with args: branch, worktree path, train command, deps command
3. `launch.sh` internally: checks connection, checks GPU/availability, creates remote worktree, installs deps, starts training in screen
4. If `launch.sh` exits 0: host claimed, write to session_state.json
5. If `launch.sh` exits non-zero: try next host
6. If no host available: fail with clear error

### launch.sh interface (all host types)

```bash
# Usage: launch.sh <branch> <branch-sanitized> <train-command> [<deps-command>]
# Exit 0: training started successfully
# Exit 1: host unavailable or error
# Stdout: status messages
#
# Host-specific details (SSH alias, remote dir, tunnel, PATH setup, GPU threshold)
# are read from host.md in the same directory — not passed as arguments.
```

Each host type implements this differently:
- **direct**: SSH + GPU check + git worktree + screen
- **slurm**: SSH + sbatch/salloc + GPU allocation
- **local**: local GPU check + worktree + background process

### kill.sh interface (all host types)

```bash
# Usage: kill.sh <branch-sanitized>
# Kills the training for the given session
```

## 5. Monitoring Cron

One recurring durable cron per session. Created after training launches.

### Changes from current cron prompt

- Session-scoped: reads `logs/sessions/<branch-sanitized>/session_state.json`
- Host-aware: reads `host` from session state, uses `.claude/rl-training/hosts/<host>/kill.sh` for kills
- Worktree-aware: ITERATE phase works in local worktree `../leggySim-wt-<branch-sanitized>`, pushes from there
- Screen name: `leggy-<branch-sanitized>`
- Notifications prefixed with `[<branch-name>]`

### Cron lifecycle

- Created: after Phase 3 (LAUNCH) succeeds
- Runs: every ~33 minutes (recurring, durable)
- Ends: when session phase becomes FINISHED or PAUSED, the cron exits without action

## 6. Discord Notification Fix

### Problem

Callers pass literal `\n` strings to `notify.sh`. Even though `notify.sh` uses `jq -Rs` for JSON encoding, the input is already wrong — it contains backslash-n characters, not newlines.

### Fix

Update the cron prompt to compose messages with actual newlines using printf or `$'...'` bash syntax. Example:

```bash
MSG=$(printf "[%s] Monitor %d — Run %d (step %d)\nStage: %d | ema_lin: %.2f\nmean_reward: %.1f" \
  "$BRANCH" "$M" "$RUN" "$STEP" "$STAGE" "$EMA" "$REWARD")
bash .claude/rl-training/scripts/notify.sh "$MSG"
```

### Branch prefix

`notify.sh` gains an optional `--branch <name>` argument. When provided, it prepends `[<branch>] ` to the message. All cron and skill notifications pass `--branch`.

## 7. SKILL.md Phase Updates

### Phase 0 (SETUP)

- Step 3 becomes iterative host addition (per-host scripts generated)
- Existing single-host config migrated

### Phase 2 (CODE)

- Branch created in main repo
- Local worktree created for the branch
- Code changes happen in worktree

### Phase 3 (LAUNCH)

- `init_session.sh` creates `logs/sessions/<branch-sanitized>/`
- Host selection loop iterates `.claude/rl-training/hosts/*/launch.sh`
- Session state records selected host
- Screen named `leggy-<branch-sanitized>`
- Creates one recurring cron scoped to this session

### MONITOR CRON PROMPT

- Reads from `logs/sessions/<branch-sanitized>/`
- Uses host-specific kill.sh
- ITERATE: works in local worktree, pushes from there
- Notifications prefixed with `[branch]`

### Recovery

- Scans `logs/sessions/*/session_state.json` for all active sessions
- Checks each has a running cron (CronList), recreates if missing
- Shows summary of all sessions and their states

### Cleanup (new)

- When session FINISHED: remove local + remote worktrees
- Session logs preserved for reference
- Branches preserved

## 8. Script Changes Summary

| Script | Change |
|--------|--------|
| `init_session.sh` | Takes branch name, creates under `logs/sessions/<branch-sanitized>/` |
| `train.sh` | Removed — replaced by per-host `launch.sh` scripts |
| `kill_training.sh` | Removed — replaced by per-host `kill.sh` scripts |
| `notify.sh` | Add `--branch` flag, fix newline documentation in cron prompt |
| `monitor.py` | Add `--session-dir` argument instead of hardcoded path |
| `evaluate_policy.py` | Add `--session-dir` argument |
| `get_latest_run.py` | No change (already takes wandb path as argument) |

## 9. Migration Plan

1. Move `logs/training_session/` → `logs/sessions/<current-branch-sanitized>/`
2. Migrate single-host config from `rl_training_infra.md` into `.claude/rl-training/hosts/lerobot/`
3. Update `config.md` with `## Hosts` section
4. Update all scripts for new session directory structure
5. Update SKILL.md with new architecture
6. Update cron prompt template
