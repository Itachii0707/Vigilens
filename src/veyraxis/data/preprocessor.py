"""Letterboxing and image preprocessing for Veyraxis Sentinel."""

import cv2
import numpy as np


class LetterboxPreprocessor:
    """
    Aspect-ratio preserving letterbox resizer.
    Pads the smaller dimension with constant fill (default (114, 114, 114)) to meet
    the target resolution while preserving aspect ratio and satisfying network stride constraints.
    """

    def __init__(
        self,
        target_size: int | tuple[int, int] = 640,
        stride: int = 32,
        color: tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scale_fill: bool = False,
        scale_up: bool = True,
    ):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.stride = stride
        self.color = color
        self.auto = auto
        self.scale_fill = scale_fill
        self.scale_up = scale_up

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Resize and pad image.
        Returns:
            padded_image: Resized and padded numpy array (H, W, C)
            ratio: Scale factor applied (gain)
            (pad_w, pad_h): Padding added to left/right and top/bottom
        """
        shape = image.shape[:2]  # [height, width]
        target_h, target_w = self.target_size

        # Scale ratio (new / old)
        r = min(target_h / shape[0], target_w / shape[1])
        if not self.scale_up:
            r = min(r, 1.0)

        # Compute unpadded new dimensions
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = target_w - new_unpad[0]
        dh = target_h - new_unpad[1]

        if self.auto:  # Minimum rectangle padding divisible by stride
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:  # Stretch directly without padding
            dw, dh = 0, 0
            new_unpad = (target_w, target_h)
            r = min(target_w / shape[1], target_h / shape[0])

        dw /= 2.0  # Divide padding into 2 sides
        dh /= 2.0

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return image, r, (dw, dh)

    @staticmethod
    def scale_coords(
        coords: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        orig_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Rescale [x1, y1, x2, y2] bounding boxes from padded letterbox space back to original image coordinates.
        """
        if len(coords) == 0:
            return coords

        dw, dh = pad
        rescaled = coords.copy().astype(np.float32)
        rescaled[:, [0, 2]] -= dw
        rescaled[:, [1, 3]] -= dh
        rescaled[:, :4] /= max(ratio, 1e-8)

        # Clip to original image bounds
        orig_h, orig_w = orig_shape[:2]
        rescaled[:, 0] = np.clip(rescaled[:, 0], 0, orig_w)
        rescaled[:, 1] = np.clip(rescaled[:, 1], 0, orig_h)
        rescaled[:, 2] = np.clip(rescaled[:, 2], 0, orig_w)
        rescaled[:, 3] = np.clip(rescaled[:, 3], 0, orig_h)
        return rescaled
