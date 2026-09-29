"""End-to-end integration test: Input Image -> Preprocessing -> Model Inference -> Postprocessing -> JSON Result."""

import json

import numpy as np

from veyraxis.data.preprocessor import LetterboxPreprocessor
from veyraxis.inference.engine import DetectionResult, InferenceEngine


def test_complete_end_to_end_pipeline(sample_image: np.ndarray):
    """
    Mandated End-to-End Pipeline Test:
    1. Input raw image array (H, W, 3)
    2. Aspect-ratio preserving letterbox preprocessing
    3. Model forward inference
    4. Postprocessing (NMS, confidence filtering, inverse coordinate scaling)
    5. Output structured JSON result conforming to Veyraxis Sentinel specification
    """
    orig_h, orig_w = sample_image.shape[:2]

    # Step 1 & 2: Preprocessing
    preprocessor = LetterboxPreprocessor(target_size=(640, 640))
    padded_img, ratio, pad = preprocessor(sample_image)
    assert padded_img.shape == (640, 640, 3)

    # Step 3: Model Inference & Step 4: Postprocessing (encapsulated in InferenceEngine)
    engine = InferenceEngine(model_path="yolo11n.pt", device="cpu", conf_threshold=0.20)
    result: DetectionResult = engine.predict_image(sample_image, image_id="e2e_frame_0001")

    # Step 5: JSON Output Verification
    json_str = result.model_dump_json(indent=2)
    parsed = json.loads(json_str)

    assert parsed["image_id"] == "e2e_frame_0001"
    assert "detections" in parsed
    assert isinstance(parsed["detections"], list)
    assert parsed["latency_ms"] is not None
    assert parsed["latency_ms"] > 0

    # Verify detection bounding box constraints if any detections occurred
    for d in parsed["detections"]:
        assert "class_id" in d
        assert "class_name" in d
        assert "confidence" in d
        assert "bbox" in d
        bbox = d["bbox"]
        assert 0 <= bbox["x1"] <= orig_w
        assert 0 <= bbox["y1"] <= orig_h
        assert 0 <= bbox["x2"] <= orig_w
        assert 0 <= bbox["y2"] <= orig_h
