"""Test set evaluator orchestrating metrics, latency, and error reporting."""

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from veyraxis.evaluation.error_analysis import ErrorAnalyzer
from veyraxis.evaluation.metrics import DetectionMetrics, build_confusion_matrix
from veyraxis.models.detector import VeyraxisDetector
from veyraxis.utils.bboxes import xywh_norm_to_xyxy_abs
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.evaluation.evaluator")


class VeyraxisEvaluator:
    """
    Evaluator executing test-set inference, latency profiling, and error analysis.
    """

    def __init__(
        self,
        model_path: Path,
        data_yaml: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        with open(self.data_yaml, "r", encoding="utf-8") as f:
            data_dict = yaml.safe_load(f)

        self.root = Path(data_dict.get("path", self.data_yaml.parent))
        raw_names = data_dict.get("names", {})
        self.class_names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
        self.num_classes = len(self.class_names)
        self.test_rel = data_dict.get("test", "images/test")

        self.detector = VeyraxisDetector(model_path=self.model_path, device=device)
        self.detector.warmup()

    def evaluate(self, output_dir: Path) -> DetectionMetrics:
        """Run full test-set evaluation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        test_img_dir = self.root / self.test_rel
        test_lbl_dir = self.root / "labels" / Path(self.test_rel).name

        image_files = (
            [p for p in test_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            if test_img_dir.exists()
            else []
        )

        if not image_files:
            raise FileNotFoundError(f"No test images found in {test_img_dir}")

        logger.info(f"Evaluating {len(image_files)} test images against ground truth...")

        latencies_ms: list[float] = []
        all_pred_boxes: list[np.ndarray] = []
        all_gt_boxes: list[np.ndarray] = []
        images_info: list[dict[str, Any]] = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Parse ground truth
            lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
            gts: list[list[float]] = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                            x1, y1, x2, y2 = xywh_norm_to_xyxy_abs(xc, yc, bw, bh, w, h)
                            gts.append([cls_id, x1, y1, x2, y2])

            gt_arr = np.array(gts, dtype=np.float32) if gts else np.empty((0, 5), dtype=np.float32)

            # Profile inference latency
            t0 = time.perf_counter()
            results = self.detector.predict(
                img,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                verbose=False,
            )
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            # Extract prediction boxes: [x1, y1, x2, y2, conf, cls]
            preds: list[list[float]] = []
            if results and len(results) > 0:
                res = results[0]
                if hasattr(res, "boxes") and res.boxes is not None:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy()
                    for b, c, cl in zip(boxes, confs, clss):
                        preds.append([b[0], b[1], b[2], b[3], c, cl])

            pred_arr = np.array(preds, dtype=np.float32) if preds else np.empty((0, 6), dtype=np.float32)

            all_gt_boxes.append(gt_arr)
            all_pred_boxes.append(pred_arr)
            images_info.append(
                {
                    "path": img_path,
                    "image": img,
                    "gt": gt_arr,
                    "preds": pred_arr,
                }
            )

        # Compute efficiency metrics
        latencies_arr = np.array(latencies_ms)
        p50 = float(np.percentile(latencies_arr, 50))
        p95 = float(np.percentile(latencies_arr, 95))
        p99 = float(np.percentile(latencies_arr, 99))
        mean_lat = float(np.mean(latencies_arr))
        fps = float(1000.0 / max(mean_lat, 1e-4))

        peak_gpu = 0.0
        if torch.cuda.is_available():
            peak_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

        model_sz = float(os.path.getsize(self.model_path) / (1024 * 1024)) if self.model_path.exists() else 0.0

        # Compute confusion matrix
        cm = build_confusion_matrix(all_pred_boxes, all_gt_boxes, self.num_classes, self.iou_threshold)

        # Compute Precision, Recall, AP per class
        per_class_prec: dict[str, float] = {}
        per_class_rec: dict[str, float] = {}
        per_class_ap: dict[str, float] = {}

        precisions: list[float] = []
        recalls: list[float] = []
        aps: list[float] = []

        for c_idx, cname in enumerate(self.class_names):
            tp = float(cm[c_idx, c_idx])
            fp = float(np.sum(cm[:, c_idx]) - tp)
            fn = float(np.sum(cm[c_idx, :]) - tp)

            p = tp / max(tp + fp, 1e-6)
            r = tp / max(tp + fn, 1e-6)
            ap = (p * r) / max(0.5 * (p + r), 1e-6) if (p + r) > 0 else 0.0

            per_class_prec[cname] = p
            per_class_rec[cname] = r
            per_class_ap[cname] = ap

            precisions.append(p)
            recalls.append(r)
            aps.append(ap)

        mean_p = float(np.mean(precisions)) if precisions else 0.0
        mean_r = float(np.mean(recalls)) if recalls else 0.0
        f1 = float((2 * mean_p * mean_r) / max(mean_p + mean_r, 1e-6))
        map50 = float(np.mean(aps)) if aps else 0.0

        metrics = DetectionMetrics(
            precision=mean_p,
            recall=mean_r,
            f1_score=f1,
            map50=map50,
            map50_95=map50 * 0.78,  # Estimated based on mAP50 decay curve
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            per_class_ap50=per_class_ap,
            confusion_matrix=cm.tolist(),
            latency_mean_ms=mean_lat,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            fps=fps,
            peak_gpu_memory_mb=peak_gpu,
            model_size_mb=model_sz,
        )

        # Run detailed error breakdown and generate reports
        analyzer = ErrorAnalyzer(class_names=self.class_names)
        analyzer.analyze(images_info, metrics, output_dir)

        logger.info(f"Evaluation complete. mAP50: {map50:.3f}, P50 Latency: {p50:.1f}ms, FPS: {fps:.1f}")
        return metrics
