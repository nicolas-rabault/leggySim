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

# Find the next run number by checking existing run_NNN directories
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
  --argjson run "$NEXT_RUN" \
  '{goal: $goal, branch: $branch, current_run: $run, wandb_run_path: "", phase: "CODE", monitor_count: 0, consecutive_bad: 0, iterations: []}' \
  > "$SESSION_DIR/session_state.json"

echo "Session initialized at $SESSION_DIR"
echo "Run directory: $SESSION_DIR/$RUN_DIR"
echo "State: $SESSION_DIR/session_state.json"
