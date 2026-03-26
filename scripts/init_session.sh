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
