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
