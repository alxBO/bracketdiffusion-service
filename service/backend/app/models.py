"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    width: int
    height: int
    file_size_bytes: int
    format: str
    histogram: Dict[str, List[int]]
    dynamic_range_ev: float
    mean_brightness: float
    median_brightness: float
    clipping_percent: float
    mean_luminance_linear: float = 0.0
    peak_luminance_linear: float = 0.0
    contrast_ratio: float = 0.0
    min_luminance_linear: float = 0.0


class GenerateRequest(BaseModel):
    num_brackets: int = Field(default=5, ge=3, le=9, description="Number of brackets (must be odd)")
    ev_steps: float = Field(default=4.0, ge=1.0, le=8.0, description="EV step size between brackets")
    diffusion_steps: int = Field(default=1000, ge=50, le=1000, description="Number of diffusion steps")
    lambda_0: float = Field(default=6.0, ge=0.1, le=20.0, description="DPS guidance weight")
    lambda_s: float = Field(default=1.0, ge=0.0, le=10.0, description="Saturation guidance weight")
    lambda_d: float = Field(default=2.0, ge=0.0, le=10.0, description="Darkness guidance weight")
    noise_sigma: float = Field(default=0.05, ge=0.0, le=0.5, description="Gaussian noise sigma")
    crf_type: Literal["complex", "gamma"] = Field(default="complex", description="Camera response function")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")


class ProgressEvent(BaseModel):
    stage: str
    progress: float
    message: str
    queue_position: int = 0


class HdrAnalysis(BaseModel):
    dynamic_range_ev: float
    contrast_ratio: float
    min_luminance: float = 0.0
    peak_luminance: float
    mean_luminance: float
    luminance_percentiles: Dict[str, float]
    hdr_histogram: dict


class ResultResponse(BaseModel):
    job_id: str
    download_url: str
    analysis: HdrAnalysis
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
