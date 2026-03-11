"""BracketDiffusion inference pipeline - supports CUDA, MPS (with CPU fallback), and CPU."""

import logging
import os
import sys
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, float, str], None]]

# Add vendor BracketDiffusion to path
# Derive from BRACKET_MODEL_DIR if set (points to .../unconditional/models),
# otherwise compute relative to this file.
_env_model_dir = os.environ.get("BRACKET_MODEL_DIR")
if _env_model_dir:
    VENDOR_DIR = os.path.normpath(os.path.join(_env_model_dir, ".."))
    MODEL_DIR = _env_model_dir
else:
    VENDOR_DIR = os.path.normpath(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', 'vendor', 'BracketDiffusion', 'unconditional')
    )
    MODEL_DIR = os.path.join(VENDOR_DIR, "models")


def _get_device():
    """Auto-detect best available device.

    MPS is NOT used: BracketDiffusion requires adaptive pooling (non-divisible sizes)
    and autograd.grad for DPS conditioning, both unsupported on MPS.
    On Apple Silicon, inference runs on CPU (fallback per spec).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS skipped — too many unsupported ops for this diffusion model
    return torch.device("cpu")


def _install_stub_modules():
    """Install stub modules for unused vendor dependencies (clip, open_clip, bkse).
    These are top-level imports in the vendor code but never used in the HDR pipeline."""
    import types
    for name in ("clip", "open_clip", "bkse"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _patch_cuda_calls():
    """Monkey-patch .cuda() to be a no-op on non-CUDA systems.
    The vendor code hardcodes .cuda() in measurements.py and utils.
    This makes those calls safely return the tensor on its current device."""
    if not torch.cuda.is_available():
        _original_cuda = torch.Tensor.cuda

        def _safe_cuda(self, *args, **kwargs):
            return self  # no-op: keep tensor on current device

        torch.Tensor.cuda = _safe_cuda


def _load_vendor_modules():
    """Import BracketDiffusion modules from vendor directory."""
    _install_stub_modules()
    _patch_cuda_calls()

    # Force our vendor dir at the front of sys.path
    if VENDOR_DIR not in sys.path:
        logger.info("Adding vendor dir to sys.path: %s", VENDOR_DIR)
    # Always ensure it's first (ahead of any site-packages with same name)
    if sys.path[0] != VENDOR_DIR:
        while VENDOR_DIR in sys.path:
            sys.path.remove(VENDOR_DIR)
        sys.path.insert(0, VENDOR_DIR)

    # Debug: verify the vendor directory contents
    gd_path = os.path.join(VENDOR_DIR, "guided_diffusion")
    logger.info("VENDOR_DIR exists: %s, guided_diffusion exists: %s",
                os.path.isdir(VENDOR_DIR), os.path.isdir(gd_path))
    if os.path.isdir(VENDOR_DIR):
        logger.info("VENDOR_DIR contents: %s", os.listdir(VENDOR_DIR))
    if os.path.isdir(gd_path):
        logger.info("guided_diffusion contents: %s", os.listdir(gd_path))
    logger.info("sys.path[0:3]: %s", sys.path[:3])

    # Evict any pre-existing guided_diffusion from sys.modules
    # (e.g. from a pip-installed package) so our vendor version is used
    for key in list(sys.modules.keys()):
        if key == "guided_diffusion" or key.startswith("guided_diffusion."):
            del sys.modules[key]

    import importlib
    try:
        from guided_diffusion.unet import create_model
        from guided_diffusion.gaussian_diffusion import create_sampler
        from guided_diffusion.measurements import get_operator, get_noise
        from guided_diffusion.condition_methods import get_conditioning_method
    except ModuleNotFoundError:
        import traceback
        logger.error("Import failed. Full traceback:\n%s", traceback.format_exc())
        raise

    return create_model, create_sampler, get_operator, get_noise, get_conditioning_method


# Default configs matching the BracketDiffusion repo
MODEL_CONFIG = {
    "image_size": 256,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": "32,16,8",
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
}

DIFFUSION_CONFIG = {
    "sampler": "ddpm",
    "steps": 1000,
    "noise_schedule": "linear",
    "model_mean_type": "epsilon",
    "model_var_type": "learned_range",
    "dynamic_threshold": False,
    "clip_denoised": True,
    "rescale_timesteps": False,
}


class BracketDiffusionPipeline:
    """BracketDiffusion inference pipeline for LDR-to-HDR conversion."""

    def __init__(self):
        self.device = _get_device()
        logger.info("BracketDiffusion pipeline using device: %s", self.device)

        # Load vendor modules
        create_model, _, _, _, _ = _load_vendor_modules()

        # Load model
        model_path = os.path.join(MODEL_DIR, "imagenet256.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                f"Run download_weights.sh first."
            )

        config = {**MODEL_CONFIG, "model_path": model_path}
        self.model = create_model(**config)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.lock = threading.Lock()
        logger.info("BracketDiffusion pipeline ready (model resolution: 256x256).")

    def run(
        self,
        img_bytes: bytes,
        progress_cb: ProgressCallback = None,
        num_brackets: int = 5,
        ev_steps: float = 4.0,
        diffusion_steps: int = 1000,
        lambda_0: float = 6.0,
        lambda_s: float = 1.0,
        lambda_d: float = 2.0,
        noise_sigma: float = 0.05,
        crf_type: str = "complex",
        seed: int = -1,
    ) -> np.ndarray:
        """Run LDR-to-HDR inference. Returns HDR float32 numpy array (H, W, 3)."""
        with self.lock:
            return self._run_inner(
                img_bytes, progress_cb, num_brackets, ev_steps, diffusion_steps,
                lambda_0, lambda_s, lambda_d, noise_sigma, crf_type, seed,
            )

    def _run_inner(
        self, img_bytes, progress_cb, num_brackets, ev_steps, diffusion_steps,
        lambda_0, lambda_s, lambda_d, noise_sigma, crf_type, seed,
    ) -> np.ndarray:
        _, create_sampler, get_operator, get_noise, get_conditioning_method = _load_vendor_modules()

        if progress_cb:
            progress_cb("preprocessing", 0.02, "Preprocessing image...")

        # Ensure num_brackets is odd
        if num_brackets % 2 == 0:
            num_brackets += 1

        # Decode and prepare input image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot decode image")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        # Transform to model input: resize to 256x256, normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ref_img = transform(img_rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        ref_img = F.interpolate(ref_img, (256, 256), mode="area")

        # Generate EV values
        evs = [ev_steps ** i for i in range(-(num_brackets // 2), num_brackets // 2 + 1)]
        ev_log2 = np.int16(np.log2(evs))
        ev_labels = ["EV+%d" % ii if ii > 0 else "EV%d" % ii for ii in ev_log2]
        logger.info("Brackets: %s", ev_labels)

        if progress_cb:
            progress_cb("preprocessing", 0.05, f"Generating {num_brackets} brackets: {', '.join(ev_labels)}...")

        # Create operator, noise, and conditioning method
        operator = get_operator(name="HDR", device=self.device, CRF_Type=crf_type)
        noiser = get_noise(name="gaussian", sigma=noise_sigma)
        cond_method = get_conditioning_method(
            "ps", operator, noiser,
            lambda_0=lambda_0, lambda_s=lambda_s, lambda_d=lambda_d,
        )
        measurement_cond_fn = cond_method.conditioning

        # Create sampler with requested diffusion steps
        diffusion_cfg = {**DIFFUSION_CONFIG, "timestep_respacing": str(diffusion_steps)}
        sampler = create_sampler(**diffusion_cfg)

        # Set seed
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if progress_cb:
            progress_cb("diffusion", 0.06, "Starting diffusion sampling...")

        # Run custom sampling loop (no file I/O)
        brackets = self._sample_loop(
            sampler=sampler,
            model=self.model,
            ref_img=ref_img,
            measurement_cond_fn=measurement_cond_fn,
            operator=operator,
            num_brackets=num_brackets,
            ev_steps=ev_steps,
            progress_cb=progress_cb,
        )

        if progress_cb:
            progress_cb("merging", 0.90, "Merging all brackets to HDR...")

        # Merge ALL brackets into HDR (vendor only uses 3, we use all)
        # brackets is (num_brackets, 3, 256, 256) in [-1, 1]
        brackets_01 = (brackets + 1.0) / 2.0
        brackets_01 = brackets_01.clamp(0.0, 1.0)

        hdr_256 = self._merge_all_brackets(operator, brackets_01, evs)
        # hdr_256 is (3, 256, 256) linear HDR

        if progress_cb:
            progress_cb("upscaling", 0.93, "Upscaling HDR to original resolution...")

        # Gain-map upsampling: apply HDR luminance ratios to full-res input
        orig_h, orig_w = img_rgb.shape[:2]
        hdr_np = self._gainmap_upsample(hdr_256, img_rgb, operator, orig_h, orig_w)

        del brackets, brackets_01, hdr_256, ref_img
        self._clear_device_cache()

        return hdr_np

    @staticmethod
    def _merge_all_brackets(operator, brackets_01, evs):
        """Merge all brackets using Debevec-style weighted average.
        Unlike vendor code which uses only 3, this uses all N brackets."""
        n = len(brackets_01)
        device = brackets_01.device

        # Apply inverse CRF to linearize each bracket
        linear = operator.invCRF(brackets_01)

        # Weight function: hat-shaped (pixels near 0 or 1 are unreliable)
        # w(z) = z for dark, (1-z) for bright, triangle for middle
        weights = torch.zeros_like(brackets_01)
        for i in range(n):
            z = brackets_01[i]
            # Triangle weight: peaks at 0.5, zero at 0 and 1
            w = torch.where(z <= 0.5, z, 1.0 - z)
            weights[i] = w.clamp(min=1e-6)

        # Divide by exposure to normalize to scene radiance
        evs_tensor = torch.tensor(evs, device=device, dtype=torch.float32)[:, None, None, None]
        linear = linear / evs_tensor

        # Weighted average
        hdr = (linear * weights).sum(0) / weights.sum(0).clamp(min=1e-8)
        return hdr.clamp(min=0.0)

    @staticmethod
    def _gainmap_upsample(hdr_256, img_rgb_fullres, operator, orig_h, orig_w):
        """Upsample 256x256 HDR to original resolution using gain map.

        1. Compute low-res input luminance (linearized via invCRF)
        2. Compute gain = HDR / input_linear at 256x256
        3. Upscale gain map to full resolution (bilateral-aware)
        4. Apply gain to full-res linearized input
        """
        # hdr_256 is (3, 256, 256) tensor, img_rgb_fullres is (H, W, 3) numpy [0, 1]

        # Linearize the 256x256 input via invCRF
        input_256 = cv2.resize(img_rgb_fullres, (256, 256), interpolation=cv2.INTER_AREA)
        input_256_t = torch.from_numpy(input_256).permute(2, 0, 1).to(hdr_256.device)
        input_linear_256 = operator.invCRF(input_256_t.unsqueeze(0))[0]

        # Compute per-channel gain map at 256x256
        eps = 1e-6
        gain_256 = hdr_256 / (input_linear_256 + eps)
        gain_256 = gain_256.clamp(0.0, 100.0)  # cap extreme gains

        # Convert gain map to numpy for upscaling
        gain_np = gain_256.cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

        # Upscale gain map to original resolution
        # Use bilateral filter to preserve edges, then resize
        gain_smooth = cv2.bilateralFilter(gain_np.astype(np.float32), d=9, sigmaColor=0.5, sigmaSpace=5)
        gain_fullres = cv2.resize(gain_smooth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Linearize full-res input
        input_linear_full = input_256_t  # we need to linearize the full-res, not the 256 version
        input_full_t = torch.from_numpy(img_rgb_fullres).permute(2, 0, 1).to(hdr_256.device)
        input_linear_full = operator.invCRF(input_full_t.unsqueeze(0))[0]
        input_linear_full_np = input_linear_full.cpu().numpy().transpose(1, 2, 0)

        # Apply gain map to full-res linearized input
        hdr_fullres = input_linear_full_np * gain_fullres
        hdr_fullres = np.maximum(hdr_fullres, 0.0).astype(np.float32)

        return hdr_fullres

    def _sample_loop(
        self, sampler, model, ref_img, measurement_cond_fn,
        operator, num_brackets, ev_steps, progress_cb,
    ) -> torch.Tensor:
        """Custom sampling loop with progress reporting. No file I/O."""
        device = self.device
        batch_size = num_brackets

        # Initial noise: (num_brackets - 1) channels (middle bracket is the measurement)
        x_start = torch.randn(
            batch_size - 1, 3, 256, 256, device=device
        ).requires_grad_()

        img = x_start
        num_timesteps = sampler.num_timesteps
        timesteps = list(range(num_timesteps))[::-1]

        for idx_i, idx in enumerate(timesteps):
            img = img.requires_grad_()
            time_t = torch.tensor([idx] * img.shape[0], device=device)
            scale = (1 - idx / (num_timesteps - 1)) ** 2

            out = sampler.p_sample(x=img, t=time_t, model=model)

            x_0_hat = out["pred_xstart"]
            x_0_hat = torch.cat(
                (x_0_hat[:img.shape[0] // 2], ref_img[0:1], x_0_hat[img.shape[0] // 2:]), 0
            )

            img = measurement_cond_fn(
                x_t=out["sample"], measurement=ref_img,
                noisy_measurement=None, x_prev=img,
                scale=scale, EV=ev_steps, x_0_hat=x_0_hat,
            )
            img = img.detach_()

            # Progress reporting
            if progress_cb and idx_i % 10 == 0:
                pct = 0.06 + 0.84 * (idx_i / num_timesteps)
                step_msg = f"Diffusion step {idx_i + 1}/{num_timesteps} (t={idx})"
                progress_cb("diffusion", round(pct, 3), step_msg)

        # Final: insert measurement as middle bracket
        img = torch.cat((img[:len(img) // 2], ref_img[0:1], img[len(img) // 2:]), 0)

        return img

    @staticmethod
    def _clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def close(self):
        del self.model
        self._clear_device_cache()


# --- I/O utilities ---

def save_exr(filepath: str, img: np.ndarray):
    """Save a float32 RGB image as OpenEXR."""
    import OpenEXR
    import Imath

    h, w, _ = img.shape
    img = img.astype(np.float32)
    header = OpenEXR.Header(w, h)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'R': float_chan, 'G': float_chan, 'B': float_chan}
    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels({
        'R': img[:, :, 0].tobytes(),
        'G': img[:, :, 1].tobytes(),
        'B': img[:, :, 2].tobytes(),
    })
    out.close()
