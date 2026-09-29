"""Core inference engine and output data structures for Veyraxis Sentinel."""

import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from veyraxis.models.detector import VeyraxisDetector
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.inference.engine")


class BoundingBox(BaseModel):
    """Absolute pixel coordinate bounding box."""

    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")


class SingleDetection(BaseModel):
    """Individual object prediction."""

    class_id: int = Field(..., description="Zero-indexed class ID")
    class_name: str = Field(..., description="Semantic class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates in pixels")
    track_id: int | None = Field(default=None, description="Persistent tracking ID if enabled")


class DetectionResult(BaseModel):
    """Frame detection payload matching Veyraxis specification."""

    image_id: str = Field(..., description="Unique frame or image identifier")
    detections: list[SingleDetection] = Field(default_factory=list, description="List of detected objects")
    latency_ms: float | None = Field(default=None, description="Model inference latency in milliseconds")


class InferenceEngine:
    """
    High-throughput object detection engine.
    Supports single image, batch tensors, tracking persistence, and structured outputs.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_tracking: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_tracking = enable_tracking
        self.detector = VeyraxisDetector(model_path=model_path, device=device)
        self.detector.warmup()
        self.class_names = self.detector.names
        self._track_boxes: dict[int, tuple[int, int, int, int]] = {}
        self.smooth_factor: float = 0.70

    def predict_image(
        self,
        image: np.ndarray,
        image_id: str = "frame_00001",
        conf_override: float | None = None,
        classes: list[int] | None = None,
    ) -> DetectionResult:
        """Run inference on a single image numpy array (H, W, 3)."""
        conf = conf_override if conf_override is not None else self.conf_threshold
        t0 = time.perf_counter()

        if self.enable_tracking:
            results = self.detector.track(
                image,
                conf_threshold=conf,
                iou_threshold=self.iou_threshold,
                verbose=False,
            )
        else:
            results = self.detector.predict(
                image,
                conf_threshold=conf,
                iou_threshold=self.iou_threshold,
                classes=classes,
                verbose=False,
            )

        latency = (time.perf_counter() - t0) * 1000.0
        detections: list[SingleDetection] = []

        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                track_ids = (
                    res.boxes.id.int().cpu().numpy()
                    if (res.boxes.is_track and res.boxes.id is not None)
                    else [None] * len(boxes)
                )

                for box, score, cls_id, trk_id in zip(boxes, confs, clss, track_ids):
                    cid = int(cls_id)
                    cname = self.class_names.get(cid, f"class_{cid}")
                    raw_box = (
                        int(round(box[0])),
                        int(round(box[1])),
                        int(round(box[2])),
                        int(round(box[3])),
                    )

                    if trk_id is not None and self.smooth_factor > 0:
                        tid = int(trk_id)
                        if tid in self._track_boxes:
                            prev = self._track_boxes[tid]
                            smoothed = (
                                int(round(self.smooth_factor * raw_box[0] + (1.0 - self.smooth_factor) * prev[0])),
                                int(round(self.smooth_factor * raw_box[1] + (1.0 - self.smooth_factor) * prev[1])),
                                int(round(self.smooth_factor * raw_box[2] + (1.0 - self.smooth_factor) * prev[2])),
                                int(round(self.smooth_factor * raw_box[3] + (1.0 - self.smooth_factor) * prev[3])),
                            )
                            self._track_boxes[tid] = smoothed
                            box_final = smoothed
                        else:
                            self._track_boxes[tid] = raw_box
                            box_final = raw_box
                    else:
                        box_final = raw_box

                    bbox = BoundingBox(
                        x1=box_final[0],
                        y1=box_final[1],
                        x2=box_final[2],
                        y2=box_final[3],
                    )
                    detections.append(
                        SingleDetection(
                            class_id=cid,
                            class_name=cname,
                            confidence=round(float(score), 4),
                            bbox=bbox,
                            track_id=int(trk_id) if trk_id is not None else None,
                        )
                    )

        return DetectionResult(
            image_id=image_id,
            detections=detections,
            latency_ms=round(latency, 2),
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
        image_ids: list[str] | None = None,
        conf_override: float | None = None,
    ) -> list[DetectionResult]:
        """Run batch inference across multiple images."""
        if not images:
            return []

        ids = image_ids or [f"batch_img_{i:04d}" for i in range(len(images))]
        conf = conf_override if conf_override is not None else self.conf_threshold

        t0 = time.perf_counter()
        results = self.detector.predict(
            images,
            conf_threshold=conf,
            iou_threshold=self.iou_threshold,
            verbose=False,
        )
        total_latency = (time.perf_counter() - t0) * 1000.0
        per_img_lat = total_latency / len(images)

        batch_out: list[DetectionResult] = []
        for img_id, res in zip(ids, results):
            detections: list[SingleDetection] = []
            if hasattr(res, "boxes") and res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()

                for box, score, cls_id in zip(boxes, confs, clss):
                    cid = int(cls_id)
                    cname = self.class_names.get(cid, f"class_{cid}")
                    bbox = BoundingBox(
                        x1=int(round(box[0])),
                        y1=int(round(box[1])),
                        x2=int(round(box[2])),
                        y2=int(round(box[3])),
                    )
                    detections.append(
                        SingleDetection(
                            class_id=cid,
                            class_name=cname,
                            confidence=round(float(score), 4),
                            bbox=bbox,
                        )
                    )
            batch_out.append(
                DetectionResult(
                    image_id=img_id,
                    detections=detections,
                    latency_ms=round(per_img_lat, 2),
                )
            )

        return batch_out
