#!/usr/bin/env bash
# vast_pull.sh — Download outputs from a running vast.ai instance
#
# Usage:
#   ./scripts/vast_pull.sh [instance_id] [ssh_host] [ssh_port]
#   or just:
#   ./scripts/vast_pull.sh          (reads .vast_instance file)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_WORKDIR="/workspace/gat_shearjamming"
LOCAL_OUT="$PROJECT_DIR/outputs"

if [[ $# -ge 3 ]]; then
    INSTANCE_ID="$1"; SSH_HOST="$2"; SSH_PORT="$3"
elif [[ -f "$PROJECT_DIR/.vast_instance" ]]; then
    read -r INSTANCE_ID SSH_HOST SSH_PORT < "$PROJECT_DIR/.vast_instance"
else
    echo "Usage: $0 <instance_id> <ssh_host> <ssh_port>"
    exit 1
fi

mkdir -p "$LOCAL_OUT"

echo "==> Pulling outputs from instance $INSTANCE_ID ($SSH_HOST:$SSH_PORT)..."
rsync -avz --progress \
    -e "ssh -p $SSH_PORT" \
    "root@$SSH_HOST:$REMOTE_WORKDIR/outputs/" \
    "$LOCAL_OUT/"

echo "==> Done. Files in: $LOCAL_OUT"
echo ""
echo "To destroy the instance and stop billing:"
echo "  vastai destroy instance $INSTANCE_ID"
