"""Unit tests for InferenceEngine prediction schemas and thresholding."""

import numpy as np
import pytest

from veyraxis.inference.engine import DetectionResult, InferenceEngine


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    return InferenceEngine(model_path="yolo11n.pt", device="cpu", conf_threshold=0.25)


def test_predict_image_schema(engine: InferenceEngine, sample_image: np.ndarray):
    """Verify inference returns validated DetectionResult schema."""
    result = engine.predict_image(sample_image, image_id="test_frame_01")

    assert isinstance(result, DetectionResult)
    assert result.image_id == "test_frame_01"
    assert result.latency_ms is not None
    assert result.latency_ms > 0

    for det in result.detections:
        assert isinstance(det.class_id, int)
        assert isinstance(det.class_name, str)
        assert 0.0 <= det.confidence <= 1.0
        assert det.bbox.x1 <= det.bbox.x2
        assert det.bbox.y1 <= det.bbox.y2


def test_predict_image_confidence_override(engine: InferenceEngine, sample_image: np.ndarray):
    """Overriding confidence to 0.99 should filter out low-confidence detections."""
    result_low = engine.predict_image(sample_image, conf_override=0.1)
    result_high = engine.predict_image(sample_image, conf_override=0.99)

    assert len(result_high.detections) <= len(result_low.detections)


def test_predict_batch(engine: InferenceEngine, sample_image: np.ndarray):
    """Verify batch inference across multiple images."""
    batch = [sample_image, sample_image]
    results = engine.predict_batch(batch, image_ids=["img_1", "img_2"])

    assert len(results) == 2
    assert results[0].image_id == "img_1"
    assert results[1].image_id == "img_2"
