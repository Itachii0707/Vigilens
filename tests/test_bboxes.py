"""Unit tests for bounding box math, transformations, and IoU calculations."""

import numpy as np
import pytest

from veyraxis.utils.bboxes import (
    clip_box_coords,
    compute_batch_iou,
    compute_iou,
    xywh_norm_to_xyxy_abs,
    xyxy_abs_to_xywh_norm,
)


def test_xywh_norm_to_xyxy_abs():
    """Test conversion of normalized center [xc, yc, w, h] to absolute [x1, y1, x2, y2]."""
    # Centered box: xc=0.5, yc=0.5, w=0.2, h=0.4 in 1000x1000 image
    # x1 = 0.5 - 0.1 = 0.4 -> 400
    # x2 = 0.5 + 0.1 = 0.6 -> 600
    # y1 = 0.5 - 0.2 = 0.3 -> 300
    # y2 = 0.5 + 0.2 = 0.7 -> 700
    x1, y1, x2, y2 = xywh_norm_to_xyxy_abs(0.5, 0.5, 0.2, 0.4, 1000, 1000)
    assert x1 == 400
    assert y1 == 300
    assert x2 == 600
    assert y2 == 700


def test_xyxy_abs_to_xywh_norm_roundtrip():
    """Test round-trip conversion [x1, y1, x2, y2] -> norm [xc, yc, w, h] -> [x1, y1, x2, y2]."""
    img_w, img_h = 640, 480
    orig_x1, orig_y1, orig_x2, orig_y2 = 120, 80, 420, 360

    xc, yc, nw, nh = xyxy_abs_to_xywh_norm(orig_x1, orig_y1, orig_x2, orig_y2, img_w, img_h)
    rec_x1, rec_y1, rec_x2, rec_y2 = xywh_norm_to_xyxy_abs(xc, yc, nw, nh, img_w, img_h)

    assert abs(orig_x1 - rec_x1) <= 1
    assert abs(orig_y1 - rec_y1) <= 1
    assert abs(orig_x2 - rec_x2) <= 1
    assert abs(orig_y2 - rec_y2) <= 1


def test_compute_iou():
    """Test exact IoU values for identical, overlapping, and disjoint boxes."""
    box_a = [100, 100, 200, 200]
    box_b = [100, 100, 200, 200]
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    # Disjoint
    box_c = [300, 300, 400, 400]
    assert compute_iou(box_a, box_c) == pytest.approx(0.0)

    # Half overlap: 50% intersection
    box_d = [100, 100, 200, 150]
    assert compute_iou(box_a, box_d) == pytest.approx(0.5)


def test_compute_batch_iou():
    """Test vectorized batch IoU matrix computation."""
    boxes1 = np.array(
        [
            [100, 100, 200, 200],
            [300, 300, 400, 400],
        ],
        dtype=np.float32,
    )

    boxes2 = np.array(
        [
            [100, 100, 200, 200],
            [500, 500, 600, 600],
        ],
        dtype=np.float32,
    )

    iou_mat = compute_batch_iou(boxes1, boxes2)
    assert iou_mat.shape == (2, 2)
    assert iou_mat[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert iou_mat[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert iou_mat[1, 0] == pytest.approx(0.0, abs=1e-3)


def test_clip_boxes():
    """Test boundary clipping logic."""
    x1, y1, x2, y2 = clip_box_coords(-20, -10, 700, 800, 640, 480)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 640
    assert y2 == 480
