#!/usr/bin/env bash
# Kill the remote training screen session on lerobot.
# Usage: scripts/kill_training.sh

set -euo pipefail

echo "Sending Ctrl-C to leggy-train screen..."
ssh lerobot "screen -S leggy-train -X stuff $'\003'"
sleep 2

if ssh lerobot "screen -ls | grep -q leggy-train" 2>/dev/null; then
    echo "Screen still exists, sending quit..."
    ssh lerobot "screen -S leggy-train -X quit" 2>/dev/null || true
fi

echo "Training killed."
