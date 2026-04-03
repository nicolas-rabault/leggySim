#!/usr/bin/env bash
# Kill training on lerobot for a given session.
# Usage: kill.sh <branch-sanitized>
# Reads host config from host.md in the same directory.

set -euo pipefail

[ $# -lt 1 ] && { echo "Usage: kill.sh <branch-sanitized>" >&2; exit 1; }

BRANCH_SANITIZED="$1"
SCREEN_NAME="leggy-$BRANCH_SANITIZED"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_HOST=$(grep "^- SSH:" "$SCRIPT_DIR/host.md" | sed 's/^- SSH:[[:space:]]*//')
TUNNEL=$(grep "^- Tunnel:" "$SCRIPT_DIR/host.md" | sed 's/^- Tunnel:[[:space:]]*//')

cleanup_tunnel() {
    [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup_tunnel EXIT

if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
    if [ -n "$TUNNEL" ]; then
        eval "$TUNNEL" &
        TUNNEL_PID=$!
        for i in $(seq 1 10); do
            sleep 2
            ssh -o ConnectTimeout=3 "$SSH_HOST" "echo ok" &>/dev/null && break
        done
        if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "echo ok" &>/dev/null; then
            echo "ERROR: Cannot connect to $SSH_HOST even with tunnel" >&2
            exit 1
        fi
    else
        echo "ERROR: Cannot connect to $SSH_HOST" >&2
        exit 1
    fi
fi

if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "screen -ls | grep -q \"$SCREEN_NAME\"" 2>/dev/null; then
    echo "No screen session '$SCREEN_NAME' found on $SSH_HOST — nothing to kill."
    exit 0
fi

echo "Sending Ctrl-C to $SCREEN_NAME on $SSH_HOST..."
ssh "$SSH_HOST" "screen -S \"$SCREEN_NAME\" -X stuff \$'\\003'" || true
sleep 2

if ssh -o ConnectTimeout=5 "$SSH_HOST" "screen -ls | grep -q \"$SCREEN_NAME\"" 2>/dev/null; then
    echo "Screen still exists, sending quit..."
    ssh "$SSH_HOST" "screen -S \"$SCREEN_NAME\" -X quit" 2>/dev/null || true
fi

echo "Training killed on $SSH_HOST."
