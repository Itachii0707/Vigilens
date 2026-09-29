"""Bounding box transformations, IoU calculations, and clipping utilities."""

import numpy as np


def xywh_norm_to_xyxy_abs(
    x_c: float, y_c: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """
    Convert normalized YOLO center format [x_center, y_center, width, height]
    to absolute pixel coordinates [x1, y1, x2, y2].
    """
    x1 = int(round((x_c - w / 2.0) * img_w))
    y1 = int(round((y_c - h / 2.0) * img_h))
    x2 = int(round((x_c + w / 2.0) * img_w))
    y2 = int(round((y_c + h / 2.0) * img_h))

    return clip_box_coords(x1, y1, x2, y2, img_w, img_h)


def xyxy_abs_to_xywh_norm(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """
    Convert absolute pixel coordinates [x1, y1, x2, y2]
    to normalized YOLO center format [x_center, y_center, width, height].
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img_w}x{img_h}")

    x_c = ((x1 + x2) / 2.0) / img_w
    y_c = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # Clamp normalized values between 0.0 and 1.0
    x_c = max(0.0, min(1.0, float(x_c)))
    y_c = max(0.0, min(1.0, float(y_c)))
    w = max(0.0, min(1.0, float(w)))
    h = max(0.0, min(1.0, float(h)))

    return round(x_c, 6), round(y_c, 6), round(w, 6), round(h, 6)


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    return x, y, x + w, y + h


def clip_box_coords(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Clip integer box coordinates to ensure they remain inside image boundaries."""
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1, min(img_w, x2))
    y2 = max(y1, min(img_h, y2))
    return x1, y1, x2, y2


def clip_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clip a batch of boxes [[x1, y1, x2, y2], ...] to image boundary."""
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, img_w)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, img_h)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, img_w)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, img_h)
    return clipped


def compute_iou(
    box1: list[float] | tuple[float, ...] | np.ndarray,
    box2: list[float] | tuple[float, ...] | np.ndarray,
) -> float:
    """
    Calculate Intersection-over-Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:4]

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    b2_area = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of pairwise IoU between two sets of boxes.
    boxes1: (N, 4) in [x1, y1, x2, y2]
    boxes2: (M, 4) in [x1, y1, x2, y2]
    Returns: (N, M) matrix of IoU values
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    union = np.maximum(union, 1e-8)

    return inter / union
