#!/bin/bash
# Deploy BracketDiffusion on a Vast.ai GPU instance
#
# === Vast.ai instance setup ===
#
# 1. Choose a GPU instance (RTX 3090+ with >= 24 GB VRAM recommended)
# 2. Use a PyTorch template image (e.g. pytorch/pytorch:2.x-cuda12.x-runtime)
# 3. In "Docker options", add:  -p 8002:8002
# 4. Set disk space to at least 10 GB
#
# === On the instance ===
#
# SSH in, then:
#   git clone --recurse-submodules <repo-url>
#   cd bracketdiffusion-service/service
#   ./deploy_vastai.sh
#
# === Access ===
#
# Option A: Click "Open" on the instance card
# Option B: Use direct IP:port from "IP Port Info" popup

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== BracketDiffusion Vast.ai Deployment ==="
echo ""

# 1. Install Python dependencies
echo "[1/3] Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"

# 2. Download model if needed
MODEL_FILE="$REPO_DIR/vendor/BracketDiffusion/unconditional/models/imagenet256.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "[2/3] Downloading model checkpoint (~1.5 GB)..."
    "$SCRIPT_DIR/download_weights.sh"
else
    echo "[2/3] Model already present, skipping download."
fi

# 3. Start the service
echo "[3/3] Starting service on port 8002..."
echo ""

if [ -n "$VAST_TCP_PORT_8002" ]; then
    echo "Direct access: http://$(hostname -I | awk '{print $1}'):$VAST_TCP_PORT_8002"
fi
echo "Local: http://0.0.0.0:8002"
echo ""

cd "$SCRIPT_DIR/backend"
export BRACKET_MODEL_DIR="$REPO_DIR/vendor/BracketDiffusion/unconditional/models"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

exec uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 1
