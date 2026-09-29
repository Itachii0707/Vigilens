"""Core VeyraxisDetector model wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from veyraxis.models.registry import resolve_model_weights
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.models.detector")


class VeyraxisDetector:
    """
    Production detector encapsulation around Ultralytics YOLO models.
    Supports CUDA automatic mixed precision, device fallback, and structured output formatting.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        device: str = "auto",
        half: bool = True,
    ):
        self.requested_name = str(model_path)
        self.resolved_weights = resolve_model_weights(self.requested_name)

        # Device determination
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.half = half and ("cuda" in self.device)

        # Check local search paths (weights/ and runs/checkpoints/) if file not in root
        target_path = self.resolved_weights
        if not Path(target_path).exists():
            for candidate_dir in [Path("weights"), Path("runs/checkpoints")]:
                candidate = candidate_dir / self.resolved_weights
                if candidate.exists():
                    target_path = str(candidate)
                    break

        logger.info(f"Loading VeyraxisDetector weights: {target_path} on {self.device} (half={self.half})")
        self.model = YOLO(target_path)

        # Query class names
        self.names: dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

    def warmup(self, imgsz: int = 640) -> None:
        """Execute a dummy inference pass to compile CUDA kernels and warm caches."""
        try:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            self.predict(dummy, conf_threshold=0.25, verbose=False)
            logger.info("Detector warmed up successfully.")
        except Exception as e:
            logger.warning(f"Detector warmup skipped or failed: {e}")

    def predict(
        self,
        source: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        imgsz: int = 640,
        verbose: bool = False,
    ) -> list[Any]:
        """
        Run forward inference.
        Returns raw Ultralytics Results list.
        """
        results = self.model.predict(
            source=source,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            imgsz=imgsz,
            device=self.device,
            half=self.half,
            verbose=verbose,
        )
        return results

    def track(
        self,
        source: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tracker_type: str = "bytetrack.yaml",
        persist: bool = True,
        verbose: bool = False,
    ) -> list[Any]:
        """
        Run tracking inference across sequential frames.
        """
        results = self.model.track(
            source=source,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker=tracker_type,
            persist=persist,
            device=self.device,
            half=self.half,
            verbose=verbose,
        )
        return results

    def export(self, format: str = "onnx", imgsz: int = 640, dynamic: bool = True, half: bool = False) -> str:
        """Export model to target format (onnx, engine, openvino, torchscript)."""
        logger.info(f"Exporting detector to {format} (imgsz={imgsz}, dynamic={dynamic}, half={half})...")
        export_path = self.model.export(
            format=format,
            imgsz=imgsz,
            dynamic=dynamic,
            half=half,
            device=self.device if format == "engine" else "cpu",
        )
        return str(export_path)
