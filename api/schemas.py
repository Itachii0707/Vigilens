"""Pydantic schemas for FastAPI endpoints."""

from pydantic import BaseModel, Field

from veyraxis.inference.engine import DetectionResult, SingleDetection


class HealthResponse(BaseModel):
    """Health status and system telemetry response."""

    status: str = Field(default="healthy", description="Service health state")
    model_name: str = Field(..., description="Loaded model weights file")
    device: str = Field(..., description="Active compute device (cuda or cpu)")
    cuda_available: bool = Field(..., description="NVIDIA GPU availability flag")
    gpu_memory_allocated_mb: float = Field(..., description="Current GPU VRAM allocation")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")
    classes: list[str] = Field(default_factory=list, description="Target object class list")


class PredictResponse(BaseModel):
    """Single image detection response payload."""

    request_id: str = Field(..., description="Tracing request identifier")
    image_id: str = Field(..., description="Identifier of the processed frame")
    detections: list[SingleDetection] = Field(default_factory=list, description="List of detected objects")
    detections_count: int = Field(..., description="Number of detected objects")
    inference_latency_ms: float = Field(..., description="Time taken for inference in milliseconds")


class BatchPredictResponse(BaseModel):
    """Multi-image batch detection response."""

    request_id: str = Field(..., description="Tracing request identifier")
    total_images: int = Field(..., description="Total images in batch")
    results: list[DetectionResult] = Field(..., description="Per-image detection outputs")
    total_latency_ms: float = Field(..., description="Total batch processing latency")


class MetricsResponse(BaseModel):
    """Telemetry and latency metrics for monitoring systems."""

    total_requests: int = Field(..., description="Cumulative inference requests served")
    total_detections_made: int = Field(..., description="Cumulative objects identified")
    failed_requests: int = Field(..., description="Total rejected or failed requests")
    average_latency_ms: float = Field(..., description="Rolling average inference latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency in milliseconds")
    active_requests: int = Field(..., description="Currently in-flight requests")


class ErrorResponse(BaseModel):
    """Structured error payload for failed requests."""

    request_id: str = Field(..., description="Tracing request identifier")
    error: str = Field(..., description="Short error title")
    detail: str = Field(..., description="Actionable explanation of why request was rejected")
