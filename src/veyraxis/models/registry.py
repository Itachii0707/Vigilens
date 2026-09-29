"""Model architecture registry and weight resolution policy."""

from dataclasses import dataclass

from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.models.registry")


@dataclass
class ModelProfile:
    name: str
    target_tier: str  # 'edge', 'balanced', 'high_accuracy'
    parameters_m: float
    gflops: float
    recommended_batch: int
    fallback_weights: str


class ModelRegistry:
    """
    Model policy manager mapping deploy targets to optimal YOLO architectures.
    Provides automatic fallback to latest compatible Ultralytics stable weights (YOLO11 / YOLOv8)
    whenever forward-looking weights (e.g. YOLO26) are requested.
    """

    PROFILES: dict[str, ModelProfile] = {
        # Small / Edge Deployment
        "yolo26n": ModelProfile("yolo26n", "edge", 2.6, 6.5, 32, "yolo11n.pt"),
        "yolo26s": ModelProfile("yolo26s", "edge", 9.4, 21.5, 16, "yolo11s.pt"),
        "yolo11n": ModelProfile("yolo11n", "edge", 2.6, 6.5, 32, "yolo11n.pt"),
        "yolo11s": ModelProfile("yolo11s", "edge", 9.4, 21.5, 16, "yolo11s.pt"),
        "yolov8n": ModelProfile("yolov8n", "edge", 3.2, 8.7, 32, "yolov8n.pt"),
        # Balanced Server Deployment
        "yolo26m": ModelProfile("yolo26m", "balanced", 20.1, 68.0, 16, "yolo11m.pt"),
        "yolo11m": ModelProfile("yolo11m", "balanced", 20.1, 68.0, 16, "yolo11m.pt"),
        "yolov8m": ModelProfile("yolov8m", "balanced", 25.9, 78.9, 16, "yolov8m.pt"),
        # High Accuracy GPU Deployment
        "yolo26l": ModelProfile("yolo26l", "high_accuracy", 25.3, 86.9, 8, "yolo11l.pt"),
        "yolo26x": ModelProfile("yolo26x", "high_accuracy", 56.9, 194.9, 4, "yolo11x.pt"),
        "yolo11l": ModelProfile("yolo11l", "high_accuracy", 25.3, 86.9, 8, "yolo11l.pt"),
        "yolo11x": ModelProfile("yolo11x", "high_accuracy", 56.9, 194.9, 4, "yolo11x.pt"),
    }

    @classmethod
    def get_profile(cls, model_name: str) -> ModelProfile | None:
        clean_name = model_name.replace(".pt", "").lower()
        return cls.PROFILES.get(clean_name)


def resolve_model_weights(requested_name: str) -> str:
    """
    Resolve requested model name to an available pretrained weight file,
    substituting stable compatible weights when experimental models are not yet distributed.
    """
    clean_name = requested_name.replace(".pt", "").lower()

    if clean_name.startswith("yolo26"):
        profile = ModelRegistry.get_profile(clean_name)
        fallback = profile.fallback_weights if profile else "yolo11n.pt"
        logger.info(
            f"Requested '{requested_name}'. In accordance with stack policy, using latest stable compatible model: '{fallback}'"
        )
        return fallback

    if not requested_name.endswith(".pt") and not requested_name.endswith(".onnx"):
        return f"{requested_name}.pt"

    return requested_name
