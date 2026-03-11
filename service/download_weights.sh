#!/bin/bash
# Download BracketDiffusion model checkpoint (ImageNet 256x256 unconditional diffusion)
#
# This downloads the pre-trained DDPM model from OpenAI's guided-diffusion.
# File: ~1.5 GB
#
# Usage: ./download_weights.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$REPO_DIR/vendor/BracketDiffusion/unconditional/models"

mkdir -p "$MODEL_DIR"

MODEL_FILE="$MODEL_DIR/imagenet256.pt"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already present at $MODEL_FILE"
    exit 0
fi

echo "Downloading ImageNet 256x256 unconditional diffusion model..."
echo "Destination: $MODEL_FILE"
echo ""

# OpenAI's guided-diffusion model (256x256 unconditional)
URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"

if command -v wget &>/dev/null; then
    wget -O "$MODEL_FILE" "$URL"
elif command -v curl &>/dev/null; then
    curl -L -o "$MODEL_FILE" "$URL"
else
    echo "ERROR: Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo ""
echo "Download complete: $(du -h "$MODEL_FILE" | cut -f1)"
