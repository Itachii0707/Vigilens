"""Lightweight ONNX Runtime inference runner with integrated NMS."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from veyraxis.data.preprocessor import LetterboxPreprocessor
from veyraxis.inference.engine import BoundingBox, DetectionResult, SingleDetection
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.deployment.onnx_runner")


class ONNXRunner:
    """
    Pure ONNX Runtime inference engine.
    Independent of heavy PyTorch dependencies for edge and lightweight server deployment.
    """

    def __init__(
        self,
        onnx_model_path: str | Path,
        device: str = "cpu",
        class_names: dict[int, str] | None = None,
    ):
        self.model_path = Path(onnx_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        # Configure execution providers
        providers = ["CPUExecutionProvider"]
        if device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        logger.info(f"Loaded ONNX model on provider: {self.active_provider}")

        # Input metadata
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic dimensions
        self.input_h = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_w = input_shape[3] if isinstance(input_shape[3], int) else 640

        self.class_names = class_names or {}
        self.preprocessor = LetterboxPreprocessor(target_size=(self.input_h, self.input_w))

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_id: str = "onnx_frame_001",
    ) -> DetectionResult:
        """Run complete ONNX pipeline: preprocessing -> inference -> NMS -> coordinate scaling."""
        orig_h, orig_w = image.shape[:2]

        # 1. Letterbox Preprocessing
        padded_img, ratio, (pad_w, pad_h) = self.preprocessor(image)

        # Convert BGR to RGB, normalize [0, 1], transpose to (1, 3, H, W)
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        # 2. Run Inference
        outputs = self.session.run(None, {self.input_name: blob})
        raw_preds = outputs[0]  # Shape: (1, 4 + num_classes, num_boxes) or (1, num_boxes, 4 + num_classes)

        if raw_preds.ndim == 3 and raw_preds.shape[1] < raw_preds.shape[2]:
            raw_preds = np.transpose(raw_preds, (0, 2, 1))

        preds = raw_preds[0]  # Shape: (num_boxes, 4 + num_classes)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        best_classes = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= conf_threshold
        if not np.any(mask):
            return DetectionResult(image_id=image_id, detections=[])

        filtered_boxes_xywh = boxes_xywh[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = best_classes[mask]

        # Convert xywh to xyxy on padded image space
        x_c, y_c, bw, bh = (
            filtered_boxes_xywh[:, 0],
            filtered_boxes_xywh[:, 1],
            filtered_boxes_xywh[:, 2],
            filtered_boxes_xywh[:, 3],
        )
        x1 = x_c - bw / 2.0
        y1 = y_c - bh / 2.0
        x2 = x_c + bw / 2.0
        y2 = y_c + bh / 2.0
        boxes_xyxy_padded = np.stack([x1, y1, x2, y2], axis=1)

        # Rescale boxes back to original image space
        rescaled_boxes = LetterboxPreprocessor.scale_coords(boxes_xyxy_padded, ratio, (pad_w, pad_h), (orig_h, orig_w))

        # 3. Non-Maximum Suppression via cv2.dnn
        cv_boxes = []
        for b in rescaled_boxes:
            bx1, by1, bx2, by2 = b
            cv_boxes.append([int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)])

        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=filtered_scores.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )

        detections: list[SingleDetection] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                rb = rescaled_boxes[idx]
                cid = int(filtered_classes[idx])
                cname = self.class_names.get(cid, f"class_{cid}")
                bbox = BoundingBox(
                    x1=int(round(rb[0])),
                    y1=int(round(rb[1])),
                    x2=int(round(rb[2])),
                    y2=int(round(rb[3])),
                )
                detections.append(
                    SingleDetection(
                        class_id=cid,
                        class_name=cname,
                        confidence=round(float(filtered_scores[idx]), 4),
                        bbox=bbox,
                    )
                )

        return DetectionResult(image_id=image_id, detections=detections)
