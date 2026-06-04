#!/usr/bin/env bash
# =============================================================================
# vast_deploy.sh — Rent a GPU on vast.ai, upload data + code, run training
# =============================================================================
# Prerequisites:
#   pip install vastai          (vast.ai CLI)
#   vastai set api-key <YOUR_KEY>
#   Docker installed locally (for building the image)
#   Docker Hub account (or any registry vast.ai can pull from)
#
# Usage:
#   ./scripts/vast_deploy.sh [data_dir]
#
# The script:
#   1. Searches for a cheap GPU instance (RTX 3090 / A100 / H100)
#   2. Creates the instance
#   3. Waits for it to be ready
#   4. Rsyncs your data and source code
#   5. Launches training inside a tmux session
#   6. Prints a command to stream logs
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR="${1:-./data}"          # local data directory with .dat files
DOCKER_IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"  # pre-built base
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_WORKDIR="/workspace/gat_shearjamming"

# GPU search parameters (adjust to taste / budget)
GPU_MIN_VRAM=16                  # GB — A100/H100 for large N=2000 graphs
MAX_PRICE=0.50                   # $/hr ceiling
GPU_NAME="RTX_3090"              # or "A100" / "H100" / "" for any

# Training hyperparameters (passed to train.py)
EPOCHS=200
LR=0.0001
NEWFDIM=10

# ── 1. Find a cheap instance ──────────────────────────────────────────────────
echo "==> Searching for GPU instance (>= ${GPU_MIN_VRAM}GB VRAM, <= \$${MAX_PRICE}/hr)..."

OFFERS=$(vastai search offers \
    "gpu_ram >= ${GPU_MIN_VRAM} \
     dph_total <= ${MAX_PRICE} \
     num_gpus = 1 \
     cuda_vers >= 12.1 \
     inet_up >= 200 \
     inet_down >= 200" \
    --order dph_total \
    --raw \
    2>/dev/null | head -5)

echo "Top offers:"
echo "$OFFERS" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
for o in offers[:5]:
    print(f\"  ID={o['id']}  GPU={o['gpu_name']}  VRAM={o['gpu_ram']}GB  \${o['dph_total']:.3f}/hr  {o['geolocation']}\")
"

# Take the cheapest
INSTANCE_ID=$(echo "$OFFERS" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
print(offers[0]['id'])
")

echo "==> Selected offer ID: $INSTANCE_ID"
read -rp "Proceed? [y/N] " confirm
[[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "Aborted."; exit 0; }

# ── 2. Create instance ────────────────────────────────────────────────────────
echo "==> Creating instance..."
CREATED=$(vastai create instance "$INSTANCE_ID" \
    --image "$DOCKER_IMAGE" \
    --disk 40 \
    --raw)

RUNNING_ID=$(echo "$CREATED" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['new_contract'])")
echo "Instance created: $RUNNING_ID"

# ── 3. Wait for ready ─────────────────────────────────────────────────────────
echo "==> Waiting for instance to become ready (this takes 1-3 min)..."
while true; do
    STATUS=$(vastai show instance "$RUNNING_ID" --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['actual_status'])")
    echo "  Status: $STATUS"
    [[ "$STATUS" == "running" ]] && break
    sleep 15
done

# Get SSH details
SSH_INFO=$(vastai show instance "$RUNNING_ID" --raw | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['ssh_host'], d['ssh_port'])
")
SSH_HOST=$(echo "$SSH_INFO" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_INFO" | awk '{print $2}')
SSH_CMD="ssh -p $SSH_PORT root@$SSH_HOST"

echo "==> Instance ready!"
echo "    SSH:  $SSH_CMD"

# Accept host key automatically (vast.ai instances have ephemeral keys)
ssh-keyscan -p "$SSH_PORT" "$SSH_HOST" >> ~/.ssh/known_hosts 2>/dev/null || true
sleep 5  # let SSH daemon fully start

# ── 4. Upload code + data ─────────────────────────────────────────────────────
echo "==> Uploading source code..."
$SSH_CMD "mkdir -p $REMOTE_WORKDIR/src $REMOTE_WORKDIR/configs $REMOTE_WORKDIR/data $REMOTE_WORKDIR/outputs"

rsync -avz --progress \
    -e "ssh -p $SSH_PORT" \
    "$PROJECT_DIR/src/" \
    "root@$SSH_HOST:$REMOTE_WORKDIR/src/"

rsync -avz --progress \
    -e "ssh -p $SSH_PORT" \
    "$PROJECT_DIR/requirements.txt" \
    "root@$SSH_HOST:$REMOTE_WORKDIR/"

echo "==> Uploading data files..."
rsync -avz --progress \
    -e "ssh -p $SSH_PORT" \
    "$DATA_DIR/" \
    "root@$SSH_HOST:$REMOTE_WORKDIR/data/"

# ── 5. Install deps & launch training ─────────────────────────────────────────
echo "==> Installing Python dependencies on instance..."
$SSH_CMD "cd $REMOTE_WORKDIR && pip install -q -r requirements.txt"

echo "==> Launching training in tmux session 'train'..."
$SSH_CMD "tmux new-session -d -s train -x 220 -y 50 \
    'cd $REMOTE_WORKDIR && python src/train.py \
        --data_dir data \
        --output_dir outputs \
        --epochs $EPOCHS \
        --lr $LR \
        --newfdim $NEWFDIM \
        2>&1 | tee outputs/training.log; echo DONE'"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Training launched!                                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Attach to live logs:                                        ║"
echo "║  $SSH_CMD -t 'tmux attach -t train'"
echo "║                                                              ║"
echo "║  Stream log file:                                            ║"
echo "║  $SSH_CMD 'tail -f $REMOTE_WORKDIR/outputs/training.log'"
echo "║                                                              ║"
echo "║  Download outputs when done:                                 ║"
echo "║  ./scripts/vast_pull.sh $RUNNING_ID $SSH_HOST $SSH_PORT     ║"
echo "║                                                              ║"
echo "║  Destroy instance when done (SAVES MONEY):                  ║"
echo "║  vastai destroy instance $RUNNING_ID                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Save instance info for pull script
echo "$RUNNING_ID $SSH_HOST $SSH_PORT" > "$PROJECT_DIR/.vast_instance"
