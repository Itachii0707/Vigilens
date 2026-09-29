"""Configuration management using Pydantic and PyYAML for Veyraxis Sentinel."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseModel):
    """Detection model configuration."""

    name: str = Field(default="yolo11n.pt", description="Base model weight file or architecture name")
    fallback_name: str = Field(
        default="yolov8n.pt", description="Fallback model if primary architecture is unavailable"
    )
    num_classes: int = Field(default=5, description="Number of object detection classes")
    classes: list[str] = Field(
        default_factory=lambda: ["person", "vehicle", "helmet", "fire", "damaged_component"],
        description="List of target class names in index order",
    )
    imgsz: int = Field(default=640, description="Input image size (pixels)")
    device: str = Field(default="auto", description="Execution device: 'cuda', 'cpu', 'mps', or 'auto'")
    half: bool = Field(default=True, description="Enable FP16 automatic mixed precision on GPU")


class TrainingSettings(BaseModel):
    """Training hyperparameter configuration."""

    epochs: int = Field(default=50, description="Total training epochs")
    batch_size: int = Field(default=16, description="Mini-batch size")
    workers: int = Field(default=4, description="Dataloader worker processes")
    optimizer: str = Field(default="AdamW", description="Optimizer: SGD, Adam, AdamW")
    lr0: float = Field(default=0.001, description="Initial learning rate")
    lrf: float = Field(default=0.01, description="Final learning rate factor (cosine decay)")
    momentum: float = Field(default=0.937, description="Momentum or beta1")
    weight_decay: float = Field(default=0.0005, description="Weight decay penalty")
    warmup_epochs: float = Field(default=3.0, description="Warmup epochs")
    patience: int = Field(default=15, description="Early stopping patience (epochs without improvement)")
    seed: int = Field(default=42, description="Reproducible random seed")
    save_period: int = Field(default=5, description="Checkpoint save frequency")
    freeze_backbone_epochs: int = Field(default=0, description="Number of initial epochs to freeze backbone")
    pretrained: bool = Field(default=True, description="Use transfer learning from pretrained weights")


class AugmentationSettings(BaseModel):
    """Data augmentation configuration for Albumentations and training."""

    hsv_h: float = Field(default=0.015, description="HSV-Hue fraction")
    hsv_s: float = Field(default=0.7, description="HSV-Saturation fraction")
    hsv_v: float = Field(default=0.4, description="HSV-Value fraction")
    degrees: float = Field(default=0.0, description="Image rotation (+/- deg)")
    translate: float = Field(default=0.1, description="Image translation fraction")
    scale: float = Field(default=0.5, description="Image scale gain")
    fliplr: float = Field(default=0.5, description="Horizontal flip probability")
    flipud: float = Field(default=0.0, description="Vertical flip probability")
    mosaic: float = Field(default=1.0, description="Mosaic augmentation probability")
    mixup: float = Field(default=0.0, description="Mixup augmentation probability")


class InferenceSettings(BaseModel):
    """Runtime inference configuration."""

    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence score threshold")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="NMS IoU threshold")
    max_det: int = Field(default=300, description="Maximum detections per frame")
    classes: list[int] | None = Field(default=None, description="Optional filter list of class IDs to detect")
    enable_tracking: bool = Field(default=False, description="Enable object tracking across frames")
    tracker_type: str = Field(default="bytetrack", description="Tracker algorithm: 'bytetrack' or 'botsort'")


class TrackingSettings(BaseModel):
    """MLflow and experiment tracking configuration."""

    enabled: bool = Field(default=True, description="Enable MLflow tracking")
    tracking_uri: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI")
    experiment_name: str = Field(default="veyraxis-sentinel", description="MLflow experiment name")
    run_name: str | None = Field(default=None, description="Descriptive run name")
    log_artifacts: bool = Field(default=True, description="Log model artifacts and plots to MLflow")


class APISettings(BaseSettings):
    """FastAPI service configuration with environment variable support."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_", case_sensitive=False)

    host: str = Field(default="0.0.0.0", description="API bind host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=1, description="Number of worker processes")
    model_path: str = Field(default="runs/checkpoints/best.pt", description="Path to trained model weights")
    device: str = Field(default="auto", description="Device override: 'cpu' or 'cuda'")
    conf_threshold: float = Field(default=0.25, description="Default confidence threshold")
    max_upload_size_mb: int = Field(default=15, description="Maximum upload file size in megabytes")
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/bmp"],
        description="Allowed input image MIME types",
    )
    request_timeout_seconds: int = Field(default=30, description="Inference request timeout")
    log_json: bool = Field(default=False, description="Log in JSON format")


class SentinelConfig(BaseModel):
    """Unified configuration schema for Veyraxis Sentinel."""

    project_name: str = Field(default="veyraxis-sentinel")
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    augmentation: AugmentationSettings = Field(default_factory=AugmentationSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    api: APISettings = Field(default_factory=APISettings)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file safely."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data
