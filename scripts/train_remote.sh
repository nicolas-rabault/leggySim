#!/usr/bin/env bash
# Launch training on lerobot GPU server.
# Usage: scripts/train_remote.sh <branch-name>
# Prerequisites: sft ssh lerobot must have been run at least once this session.

set -euo pipefail

BRANCH="$1"
REMOTE_DIR="~/leggySim"
# Non-interactive SSH doesn't load .profile, so prepend common user paths
REMOTE_ENV="export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"

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
ssh lerobot "$REMOTE_ENV && cd $REMOTE_DIR && uv sync"

echo "=== Launching training in screen ==="
# Create or reuse screen session, send the training command
ssh lerobot "screen -ls | grep -q leggy-train && screen -S leggy-train -X stuff $'\003' || screen -dmS leggy-train"
sleep 1
ssh lerobot "screen -S leggy-train -X stuff 'export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH && cd $REMOTE_DIR && uv run leggy-train Mjlab-Leggy --env.scene.num-envs 2048\n'"

echo "=== Training launched ==="
echo "Monitor with: ssh lerobot 'screen -r leggy-train'"
echo "Check wandb for the new run at: https://wandb.ai/success-hf/Leggy"
