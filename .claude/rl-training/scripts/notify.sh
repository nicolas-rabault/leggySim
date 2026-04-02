#!/usr/bin/env bash
# Send a notification via Discord webhook.
# Usage: .claude/rl-training/scripts/notify.sh "<message>" [--branch <name>] [--file <path>]
#
# --branch: prepends [<branch>] to the message
# --file: attaches a file (e.g. video)
#
# Reads WEBHOOK_URL from .claude-discord.md in the project root.
# Messages are truncated to 2000 chars (Discord limit).
# IMPORTANT: pass actual newlines in the message, not literal \n.
# Use printf or $'...' syntax to compose multi-line messages.

set -euo pipefail

MESSAGE="$1"
shift

FILE=""
BRANCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file) FILE="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -n "$BRANCH" ]; then
    MESSAGE="[$BRANCH] $MESSAGE"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DISCORD_CONFIG="$PROJECT_ROOT/.claude-discord.md"

if [ ! -f "$DISCORD_CONFIG" ]; then
    echo "WARNING: No .claude-discord.md found — skipping notification" >&2
    exit 0
fi

WEBHOOK_URL=$(grep -E "^WEBHOOK_URL:" "$DISCORD_CONFIG" | head -1 | sed 's/^WEBHOOK_URL:[[:space:]]*//')

if [ -z "$WEBHOOK_URL" ]; then
    echo "WARNING: No WEBHOOK_URL found in .claude-discord.md — skipping notification" >&2
    exit 0
fi

MESSAGE="${MESSAGE:0:2000}"

if [ -n "$FILE" ] && [ -f "$FILE" ]; then
    curl -s \
        -F "payload_json={\"content\":$(printf '%s' "$MESSAGE" | jq -Rs .)}" \
        -F "file=@$FILE" \
        "$WEBHOOK_URL" > /dev/null
else
    curl -s -H "Content-Type: application/json" \
        -d "{\"content\":$(printf '%s' "$MESSAGE" | jq -Rs .)}" \
        "$WEBHOOK_URL" > /dev/null
fi
