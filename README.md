# BracketDiffusion Service

Web service for LDR-to-HDR image conversion using [BracketDiffusion](https://github.com/m-bemana/BracketDiffusion) (Eurographics 2025).

The model generates multiple exposure brackets via diffusion posterior sampling, then merges them into an HDR image.

## Structure

- `service/` — Web application (FastAPI backend + static frontend)
- `vendor/BracketDiffusion/` — Original BracketDiffusion repository (git submodule)

## Quick Start (Mac)

```bash
git clone --recurse-submodules <repo-url>
cd bracketdiffusion-service

# Install dependencies
cd service/backend
pip install -r requirements.txt

# Download model (~1.5 GB) + start service
cd ..
./run_mac.sh
```

Open `http://localhost:8003`.

## Quick Start (Vast.ai GPU)

```bash
git clone --recurse-submodules <repo-url>
cd bracketdiffusion-service/service
./deploy_vastai.sh
```

## Features

- Diffusion-based LDR-to-HDR via exposure bracket generation
- All inference parameters exposed (brackets, EV steps, diffusion steps, DPS weights, CRF, seed)
- FIFO job queue with real-time progress (SSE)
- Client-side tone mapping (ACES, Reinhard, Linear)
- A/B comparison slider (SDR vs HDR)
- Input/output image analysis (histogram, dynamic range, luminance)
- EXR export
- In-memory storage (no temp files from inference)
- CUDA (GPU) + CPU fallback on Mac (MPS not supported by this diffusion model)

## Limitations

- Output resolution is **256x256** (model architecture constraint)
- Inference is slow (~5-15 min on GPU depending on diffusion steps)
- GPU memory: ~23 GB VRAM for 5 brackets (24+ GB VRAM recommended)
- On MPS, gradient computation may fall back to CPU

See [INSTALL.md](INSTALL.md) for detailed setup, API reference, and troubleshooting.
