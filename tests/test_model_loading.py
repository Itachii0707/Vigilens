"""Unit tests for ModelRegistry, weight resolution, and detector loading."""

from veyraxis.models.detector import VeyraxisDetector
from veyraxis.models.registry import ModelRegistry, resolve_model_weights


def test_model_registry_profiles():
    """Verify registry has profiles for required deployment tiers."""
    assert ModelRegistry.get_profile("yolo26n") is not None
    assert ModelRegistry.get_profile("yolo26s") is not None
    assert ModelRegistry.get_profile("yolo26m") is not None
    assert ModelRegistry.get_profile("yolo26l") is not None
    assert ModelRegistry.get_profile("yolo11n") is not None


def test_resolve_model_weights_fallback():
    """Verify YOLO26 automatically falls back to latest stable compatible model (YOLO11)."""
    assert resolve_model_weights("yolo26n") == "yolo11n.pt"
    assert resolve_model_weights("yolo26s") == "yolo11s.pt"
    assert resolve_model_weights("yolo26m") == "yolo11m.pt"
    assert resolve_model_weights("yolo26l") == "yolo11l.pt"
    assert resolve_model_weights("yolo11n.pt") == "yolo11n.pt"


def test_detector_initialization_cpu_fallback():
    """Test explicit CPU device placement."""
    detector = VeyraxisDetector(model_path="yolo11n.pt", device="cpu")
    assert detector.device == "cpu"
    assert detector.half is False
    assert len(detector.names) > 0


def test_detector_warmup(sample_image):
    """Test warmup execution on dummy image."""
    detector = VeyraxisDetector(model_path="yolo11n.pt", device="cpu")
    detector.warmup(imgsz=320)
