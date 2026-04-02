#!/usr/bin/env bash
# Launch training — remote or local.
# Usage: .claude/rl-training/scripts/train.sh <branch> <ssh-host> <remote-dir> <screen-name> <train-command> [<deps-command>] [<remote-path-setup>]
#
# For local training:
#   .claude/rl-training/scripts/train.sh <branch> local "" "" "<train-command>" [<deps-command>]

set -euo pipefail

BRANCH="$1"
SSH_HOST="$2"
REMOTE_DIR="$3"
SCREEN_NAME="$4"
TRAIN_CMD="$5"
DEPS_CMD="${6:-}"
REMOTE_ENV="${7:-}"

if [ "$SSH_HOST" = "local" ]; then
    echo "=== Local training ==="
    git checkout "$BRANCH"
    [ -n "$DEPS_CMD" ] && eval "$DEPS_CMD"
    echo "=== Launching training ==="
    eval "$TRAIN_CMD" &
    echo "Training PID: $!"
    exit 0
fi

echo "=== Checking SSH connection ==="
if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot connect to $SSH_HOST" >&2
    exit 1
fi

echo "=== Checking GPU usage ==="
GPU_MAX=$(ssh "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" | awk '{if($1+0 > max) max=$1+0} END{print max}')
echo "GPU max utilization: ${GPU_MAX}%"
if [ "$GPU_MAX" -gt 50 ]; then
    echo "ERROR: GPU max utilization is ${GPU_MAX}% — someone else may be using it" >&2
    exit 1
fi

echo "=== Pulling branch $BRANCH ==="
ssh "$SSH_HOST" "cd $REMOTE_DIR && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"

if [ -n "$DEPS_CMD" ]; then
    echo "=== Syncing dependencies ==="
    if [ -n "$REMOTE_ENV" ]; then
        ssh "$SSH_HOST" "$REMOTE_ENV && cd $REMOTE_DIR && $DEPS_CMD"
    else
        ssh "$SSH_HOST" "cd $REMOTE_DIR && $DEPS_CMD"
    fi
fi

echo "=== Launching training in screen ==="
ssh "$SSH_HOST" "screen -ls | grep -q $SCREEN_NAME && screen -S $SCREEN_NAME -X stuff \$'\\003' || screen -dmS $SCREEN_NAME"
sleep 1

FULL_CMD="cd $REMOTE_DIR && $TRAIN_CMD"
[ -n "$REMOTE_ENV" ] && FULL_CMD="$REMOTE_ENV && $FULL_CMD"
ssh "$SSH_HOST" "screen -S $SCREEN_NAME -X stuff '$FULL_CMD\n'"

echo "=== Training launched ==="
echo "Monitor with: ssh $SSH_HOST 'screen -r $SCREEN_NAME'"
