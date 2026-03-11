#!/bin/bash
# Run BracketDiffusion service natively on Mac (with MPS/CPU)
#
# Prerequisites:
#   pip install -r backend/requirements.txt
#   ./download_weights.sh
#
# Usage: ./run_mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Check model exists
MODEL_FILE="$REPO_DIR/vendor/BracketDiffusion/unconditional/models/imagenet256.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Model not found. Downloading..."
    "$SCRIPT_DIR/download_weights.sh"
fi

export BRACKET_MODEL_DIR="$REPO_DIR/vendor/BracketDiffusion/unconditional/models"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

echo "Starting BracketDiffusion on http://localhost:8003"
echo "Backend: PyTorch (CPU — MPS not supported by this diffusion model)"
echo "Model: $MODEL_FILE"
echo ""

cd "$SCRIPT_DIR/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 1
