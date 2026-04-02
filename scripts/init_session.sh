#!/usr/bin/env bash
# Initialize or reset a training session directory.
# Usage: scripts/init_session.sh "<goal>" "<branch>"

set -euo pipefail

GOAL="$1"
BRANCH="$2"

SESSION_DIR="logs/training_session"

# If session_state.json exists, archive it
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
