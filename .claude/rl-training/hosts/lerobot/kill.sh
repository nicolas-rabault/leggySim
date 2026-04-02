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
