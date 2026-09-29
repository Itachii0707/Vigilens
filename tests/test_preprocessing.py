"""Unit tests for LetterboxPreprocessor and coordinate scaling."""

import numpy as np
import pytest

from veyraxis.data.preprocessor import LetterboxPreprocessor


def test_letterbox_resizing_and_padding():
    """Verify aspect-ratio preserving padding to target resolution."""
    # Input image: 400x800 (h=400, w=800), target: 640x640
    # Aspect ratio is 2:1. Width scales to 640 (r=0.8), height becomes 320.
    # Vertical padding: (640 - 320) / 2 = 160 top and bottom.
    preprocessor = LetterboxPreprocessor(target_size=(640, 640), auto=False)
    input_img = np.full((400, 800, 3), 100, dtype=np.uint8)

    padded, ratio, (pad_w, pad_h) = preprocessor(input_img)

    assert padded.shape == (640, 640, 3)
    assert ratio == pytest.approx(0.8, abs=1e-3)
    assert pad_w == pytest.approx(0.0, abs=1e-3)
    assert pad_h == pytest.approx(160.0, abs=1e-3)


def test_scale_coords_inverse_mapping():
    """Test mapping bounding box from padded letterbox space back to original resolution."""
    orig_shape = (400, 800)
    ratio = 0.8
    pad = (0.0, 160.0)

    # Let a box in padded space be centered:
    # x1=160, y1=240, x2=480, y2=400
    padded_box = np.array([[160.0, 240.0, 480.0, 400.0]], dtype=np.float32)

    rescaled = LetterboxPreprocessor.scale_coords(padded_box, ratio, pad, orig_shape)

    # Expected:
    # x: 160 / 0.8 = 200, 480 / 0.8 = 600
    # y: (240 - 160) / 0.8 = 100, (400 - 160) / 0.8 = 300
    assert rescaled[0, 0] == pytest.approx(200.0, abs=1.0)
    assert rescaled[0, 1] == pytest.approx(100.0, abs=1.0)
    assert rescaled[0, 2] == pytest.approx(600.0, abs=1.0)
    assert rescaled[0, 3] == pytest.approx(300.0, abs=1.0)
