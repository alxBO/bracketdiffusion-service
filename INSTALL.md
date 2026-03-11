# BracketDiffusion Service - Installation Guide

## Table of contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Native installation (Mac)](#native-installation-mac)
- [Vast.ai deployment](#vastai-deployment)
- [Parameters](#parameters)
- [Environment variables](#environment-variables)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────────────────┐
│  FastAPI (port 8002)        │
│  - Static files (HTML/JS)   │
│  - REST API                 │
│  - SSE (progress)           │
│  - GPU/MPS/CPU inference    │
│  - In-memory storage        │
└─────────────────────────────┘
```

The backend uses **PyTorch** with CUDA (NVIDIA), MPS (Apple Silicon), or CPU.

The BracketDiffusion model operates at **256x256** resolution. Input images are automatically resized.

Results are kept in memory (no disk writes for inference) and purged after `JOB_TTL_HOURS`.

---

## Prerequisites

### Clone with submodule

```bash
git clone --recurse-submodules <repo-url>
cd bracketdiffusion-service

# Or if already cloned
git submodule update --init
```

### Model checkpoint

The model uses OpenAI's pre-trained ImageNet 256x256 unconditional diffusion model (~1.5 GB).

```bash
cd service
./download_weights.sh
```

This downloads the checkpoint to `vendor/BracketDiffusion/unconditional/models/imagenet256.pt`.

---

## Native installation (Mac)

### 1. Create a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Or with conda:

```bash
conda create -n bracketdiff python=3.10
conda activate bracketdiff
```

### 2. Install dependencies

```bash
cd service/backend
pip install -r requirements.txt
```

> **Note**: `OpenEXR` may require:
> ```bash
> brew install openexr
> ```

### 3. Run

```bash
cd service
./run_mac.sh
```

The web UI is available at `http://localhost:8002`.

### Device selection

- **CUDA**: auto-detected if available. No CPU fallback.
- **CPU**: used on Mac / non-CUDA systems. MPS is not used because BracketDiffusion requires adaptive pooling (non-divisible sizes) and `autograd.grad` for DPS conditioning, both unsupported on MPS.

---

## Vast.ai deployment

### 1. Create the instance

- Pick a GPU with **>= 24 GB VRAM** (RTX 3090, RTX 4090, A100 recommended)
- Use a **PyTorch** template image (e.g. `pytorch/pytorch:2.x-cuda12.x-runtime`)
- In **Docker options**, add: `-p 8002:8002`
- Disk space: **10 GB minimum**

> **Important**: 5 brackets at 256x256 require ~23 GB VRAM. Use 3 brackets to reduce to ~14 GB.

### 2. Deploy

```bash
ssh -p <PORT> root@<IP>

git clone --recurse-submodules <repo-url>
cd bracketdiffusion-service/service
./deploy_vastai.sh
```

The script will:
1. Install Python dependencies
2. Download the model checkpoint (~1.5 GB)
3. Start uvicorn on port 8002

### 3. Access

| Method | How |
|--------|-----|
| **Cloudflare tunnel** | Click **"Open"** on the instance card |
| **Direct IP** | Use `http://<PUBLIC_IP>:<EXTERNAL_PORT>` from "IP Port Info" |

---

## Parameters

### Bracket Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Brackets | 5 | 3-9 (odd) | Number of exposure brackets. More = better HDR, more VRAM, slower |
| EV Step | 4 | 1-8 | Exposure difference between adjacent brackets. Higher = wider dynamic range |

### Diffusion Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Steps | 1000 | 50-1000 | Diffusion sampling steps. More = better quality, slower |
| Seed | -1 | -1 to 999999 | Random seed (-1 = random). Same seed = reproducible results |

### Advanced (DPS conditioning)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Lambda (guidance) | 6.0 | 0.1-20 | DPS posterior sampling weight. Higher = stronger guidance |
| Lambda (saturation) | 1.0 | 0-10 | Weight for saturation constraint on overexposed brackets |
| Lambda (darkness) | 2.0 | 0-10 | Weight for darkness constraint on underexposed brackets |
| Noise sigma | 0.05 | 0-0.5 | Gaussian noise intensity in measurement model |
| CRF Type | complex | complex/gamma | Camera response function. "complex" is more accurate |

### Bracket generation

With `num_brackets=5` and `ev_steps=4`:

```
EV-4  EV-2  EV+0  EV+2  EV+4
(1/16x) (1/4x) (1x) (4x) (16x) exposure
```

The middle bracket (EV+0) is the input image. The model generates the other brackets via diffusion and merges all into HDR.

### Speed estimates

| Platform | 1000 steps | 250 steps | 100 steps |
|----------|-----------|-----------|-----------|
| RTX 3090 (CUDA) | ~10 min | ~3 min | ~1 min |
| RTX 4090 (CUDA) | ~7 min | ~2 min | ~45s |
| Apple M2 Pro (MPS) | ~30 min | ~8 min | ~3 min |
| CPU | ~2h+ | ~30 min | ~15 min |

*Approximate times with 5 brackets. Fewer brackets are proportionally faster.*

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BRACKET_MODEL_DIR` | *(auto)* | Directory containing `imagenet256.pt` |
| `JOB_TTL_HOURS` | `24` | How long completed results are kept in memory |
| `MAX_MEGAPIXELS` | `50` | Maximum resolution accepted at upload |

---

## API Reference

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload LDR image (multipart) |
| `POST` | `/api/generate/{job_id}` | Start HDR generation (queued) |
| `POST` | `/api/cancel/{job_id}` | Cancel a queued or running job |
| `GET` | `/api/status/{job_id}` | SSE progress stream |
| `GET` | `/api/result/{job_id}` | Result metadata |
| `GET` | `/api/hdr-raw/{job_id}` | Raw HDR data (float32) for tone mapping |
| `GET` | `/api/download/{job_id}` | Download EXR file |

### Example with curl

```bash
# Upload
curl -s -F "file=@photo.jpg" http://localhost:8002/api/upload

# Generate (5 brackets, 250 steps for faster inference)
curl -s -X POST http://localhost:8002/api/generate/<job_id> \
  -H "Content-Type: application/json" \
  -d '{"num_brackets": 5, "ev_steps": 4, "diffusion_steps": 250}'

# Progress (SSE)
curl -N http://localhost:8002/api/status/<job_id>

# Download result
curl -o photo_hdr.exr http://localhost:8002/api/download/<job_id>
```

### Request body for `/api/generate`

```json
{
  "num_brackets": 5,
  "ev_steps": 4.0,
  "diffusion_steps": 1000,
  "lambda_0": 6.0,
  "lambda_s": 1.0,
  "lambda_d": 2.0,
  "noise_sigma": 0.05,
  "crf_type": "complex",
  "seed": -1
}
```

---

## Troubleshooting

### Model checkpoint not found

```
Model checkpoint not found at .../imagenet256.pt
```

Run `./download_weights.sh` from the `service/` directory.

### Out of GPU memory

Reduce `num_brackets` from 5 to 3. This cuts VRAM usage from ~23 GB to ~14 GB.

You can also reduce `diffusion_steps` (does not affect VRAM, only speed/quality).

### OpenEXR won't install on Mac

```bash
brew install openexr
pip install OpenEXR
```

### Slow inference

BracketDiffusion is a diffusion model — inference is inherently slow (many denoising steps). To speed up:

1. Reduce `diffusion_steps` (250 is a good balance)
2. Reduce `num_brackets` to 3
3. Use a GPU with more VRAM and compute
