#!/usr/bin/env bash
# Send a message (and optional file) to the project Discord webhook.
# Usage:
#   scripts/notify_discord.sh "message text"
#   scripts/notify_discord.sh "message text" --file path/to/video.mp4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEBHOOK_URL=$(grep '^WEBHOOK_URL:' "$PROJECT_ROOT/.claude-discord.md" | sed 's/^WEBHOOK_URL: //')

if [ -z "$WEBHOOK_URL" ]; then
    echo "ERROR: No webhook URL found in .claude-discord.md" >&2
    exit 1
fi

MESSAGE=$(printf '%b' "$1")
shift

FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --file) FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

PAYLOAD=$(jq -n --arg msg "$MESSAGE" '{content: $msg}')

if [ -n "$FILE" ]; then
    curl -s -F "payload_json=$PAYLOAD" \
         -F "file1=@$FILE" \
         "$WEBHOOK_URL"
else
    curl -s -H "Content-Type: application/json" \
         -d "$PAYLOAD" \
         "$WEBHOOK_URL"
fi
