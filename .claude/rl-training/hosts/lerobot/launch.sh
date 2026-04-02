#!/usr/bin/env bash
# Launch training on lerobot (direct SSH + GPU + screen).
# Usage: launch.sh <branch> <branch-sanitized> <train-command> [<deps-command>]
# Exit 0: training started. Exit 1: host unavailable.
# Reads host config from host.md in the same directory.

set -euo pipefail

BRANCH="$1"
BRANCH_SANITIZED="$2"
TRAIN_CMD="$3"
DEPS_CMD="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Read host config
SSH_HOST=$(grep "^- SSH:" "$SCRIPT_DIR/host.md" | sed 's/^- SSH:[[:space:]]*//')
REMOTE_DIR=$(grep "^- Remote dir:" "$SCRIPT_DIR/host.md" | sed 's/^- Remote dir:[[:space:]]*//')
TUNNEL=$(grep "^- Tunnel:" "$SCRIPT_DIR/host.md" | sed 's/^- Tunnel:[[:space:]]*//')
GPU_CHECK=$(grep "^- GPU check:" "$SCRIPT_DIR/host.md" | sed 's/^- GPU check:[[:space:]]*//')
GPU_THRESHOLD=$(grep "^- GPU threshold:" "$SCRIPT_DIR/host.md" | sed 's/^- GPU threshold:[[:space:]]*//')
PATH_SETUP=$(grep "^- PATH setup:" "$SCRIPT_DIR/host.md" | sed 's/^- PATH setup:[[:space:]]*//')
HOST_DEPS=$(grep "^- Dependencies:" "$SCRIPT_DIR/host.md" | sed 's/^- Dependencies:[[:space:]]*//')

SCREEN_NAME="leggy-$BRANCH_SANITIZED"
WORKTREE_DIR="${REMOTE_DIR%/*}/leggySim-wt-$BRANCH_SANITIZED"

echo "=== Checking SSH connection to $SSH_HOST ==="
if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
    if [ -n "$TUNNEL" ]; then
        echo "SSH failed, trying tunnel: $TUNNEL"
        eval "$TUNNEL" &
        TUNNEL_PID=$!
        sleep 8
        if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
            kill "$TUNNEL_PID" 2>/dev/null || true
            echo "ERROR: Cannot connect to $SSH_HOST even with tunnel" >&2
            exit 1
        fi
    else
        echo "ERROR: Cannot connect to $SSH_HOST" >&2
        exit 1
    fi
fi

if [ -n "$GPU_CHECK" ]; then
    echo "=== Checking GPU usage ==="
    GPU_MAX=$(ssh "$SSH_HOST" "$GPU_CHECK" | awk '{if($1+0 > max) max=$1+0} END{print max}')
    echo "GPU max utilization: ${GPU_MAX}%"
    if [ "$GPU_MAX" -gt "$GPU_THRESHOLD" ]; then
        echo "ERROR: GPU utilization ${GPU_MAX}% > threshold ${GPU_THRESHOLD}%" >&2
        exit 1
    fi
fi

echo "=== Setting up remote worktree ==="
ssh "$SSH_HOST" "cd $REMOTE_DIR && git fetch origin && (git worktree add $WORKTREE_DIR $BRANCH 2>/dev/null || (cd $WORKTREE_DIR && git checkout $BRANCH && git pull origin $BRANCH))"

if [ -n "${DEPS_CMD:-$HOST_DEPS}" ]; then
    ACTUAL_DEPS="${DEPS_CMD:-$HOST_DEPS}"
    echo "=== Syncing dependencies ==="
    if [ -n "$PATH_SETUP" ]; then
        ssh "$SSH_HOST" "$PATH_SETUP && cd $WORKTREE_DIR && $ACTUAL_DEPS"
    else
        ssh "$SSH_HOST" "cd $WORKTREE_DIR && $ACTUAL_DEPS"
    fi
fi

echo "=== Launching training in screen $SCREEN_NAME ==="
ssh "$SSH_HOST" "screen -ls | grep -q '$SCREEN_NAME' && screen -S '$SCREEN_NAME' -X stuff \$'\\003' || screen -dmS '$SCREEN_NAME'"
sleep 1

FULL_CMD="cd $WORKTREE_DIR && $TRAIN_CMD"
[ -n "$PATH_SETUP" ] && FULL_CMD="$PATH_SETUP && $FULL_CMD"
ssh "$SSH_HOST" "screen -S '$SCREEN_NAME' -X stuff '$FULL_CMD\n'"

echo "=== Training launched on $SSH_HOST ==="
echo "Worktree: $WORKTREE_DIR"
echo "Screen: $SCREEN_NAME"
echo "Monitor with: ssh $SSH_HOST 'screen -r $SCREEN_NAME'"
