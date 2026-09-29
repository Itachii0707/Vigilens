"""Detection metrics computation, latency profiling, and confusion matrix calculation."""

from dataclasses import asdict, dataclass, field

import numpy as np

from veyraxis.utils.bboxes import compute_batch_iou


@dataclass
class DetectionMetrics:
    """Consolidated detection performance and computational efficiency metrics."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_ap50: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    fps: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    model_size_mb: float = 0.0

    def to_dict(self) -> dict[str, any]:
        return asdict(self)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO standard)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))
    return ap


def build_confusion_matrix(
    pred_boxes: list[np.ndarray],  # list of (N, 6): [x1, y1, x2, y2, conf, cls]
    gt_boxes: list[np.ndarray],  # list of (M, 5): [cls, x1, y1, x2, y2]
    num_classes: int,
    iou_thresh: float = 0.5,
) -> np.ndarray:
    """
    Build (num_classes + 1, num_classes + 1) confusion matrix including background class.
    Index `num_classes` represents background/unmatched.
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for preds, gts in zip(pred_boxes, gt_boxes):
        if len(gts) == 0 and len(preds) == 0:
            continue

        if len(gts) == 0:
            # All predictions are false positives (background ground truth)
            for p in preds:
                p_cls = int(p[5])
                matrix[num_classes, p_cls] += 1
            continue

        if len(preds) == 0:
            # All ground truths are false negatives (background prediction)
            for g in gts:
                g_cls = int(g[0])
                matrix[g_cls, num_classes] += 1
            continue

        gt_coords = gts[:, 1:5]
        pred_coords = preds[:, :4]
        ious = compute_batch_iou(gt_coords, pred_coords)  # (G, P)

        matched_gt = set()
        matched_pred = set()

        # Sort predictions by confidence descending
        conf_order = np.argsort(-preds[:, 4])

        for p_idx in conf_order:
            best_gt_idx = -1
            best_iou = iou_thresh

            for g_idx in range(len(gts)):
                if g_idx in matched_gt:
                    continue
                if ious[g_idx, p_idx] > best_iou:
                    best_iou = ious[g_idx, p_idx]
                    best_gt_idx = g_idx

            p_cls = int(preds[p_idx, 5])
            if best_gt_idx >= 0:
                g_cls = int(gts[best_gt_idx, 0])
                matrix[g_cls, p_cls] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
            else:
                matrix[num_classes, p_cls] += 1  # False positive

        # Remaining unmatched ground truths are false negatives
        for g_idx in range(len(gts)):
            if g_idx not in matched_gt:
                g_cls = int(gts[g_idx, 0])
                matrix[g_cls, num_classes] += 1

    return matrix
