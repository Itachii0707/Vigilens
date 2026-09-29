"""Dataset generator and management utilities for Veyraxis Sentinel."""

import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from veyraxis.utils.bboxes import xyxy_abs_to_xywh_norm
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.data.dataset")

DEFAULT_CLASSES = ["person", "vehicle", "helmet", "fire", "damaged_component"]


def create_synthetic_sentinel_dataset(
    output_root: Path,
    classes: list[str] = DEFAULT_CLASSES,
    samples_per_split: dict[str, int] = {"train": 16, "val": 6, "test": 6},
    img_size: tuple[int, int] = (640, 640),
) -> Path:
    """
    Generate a deterministic synthetic object detection dataset for smoke testing,
    pipeline integration testing, and local benchmarking.
    """
    output_root = Path(output_root)
    w, h = img_size

    class_colors = [
        (60, 180, 75),  # person (green)
        (0, 130, 200),  # vehicle (blue)
        (245, 130, 48),  # helmet (orange)
        (230, 25, 75),  # fire (red)
        (145, 30, 180),  # damaged component (purple)
    ]

    for split, count in samples_per_split.items():
        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            # Create synthetic canvas with gradient/textured background
            bg_color = random.randint(30, 90)
            img = np.full((h, w, 3), bg_color, dtype=np.uint8)

            # Add background noise/texture
            noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            annotations: list[str] = []
            num_objects = random.randint(1, 3)

            for _ in range(num_objects):
                cls_id = random.randint(0, len(classes) - 1)
                obj_w = random.randint(50, 180)
                obj_h = random.randint(60, 220)
                x1 = random.randint(20, w - obj_w - 20)
                y1 = random.randint(20, h - obj_h - 20)
                x2 = x1 + obj_w
                y2 = y1 + obj_h

                color = class_colors[cls_id % len(class_colors)]

                # Draw distinguishable geometric shapes for the synthetic classes
                if cls_id == 0:  # person: head circle + torso rect
                    head_radius = obj_w // 4
                    cv2.circle(img, (x1 + obj_w // 2, y1 + head_radius), head_radius, color, -1)
                    cv2.rectangle(img, (x1 + obj_w // 4, y1 + head_radius * 2), (x2 - obj_w // 4, y2), color, -1)
                elif cls_id == 1:  # vehicle: wide rectangle + wheels
                    cv2.rectangle(img, (x1, y1 + obj_h // 3), (x2, y2), color, -1)
                    cv2.circle(img, (x1 + obj_w // 4, y2), 12, (20, 20, 20), -1)
                    cv2.circle(img, (x2 - obj_w // 4, y2), 12, (20, 20, 20), -1)
                elif cls_id == 2:  # helmet: dome ellipse
                    cv2.ellipse(img, (x1 + obj_w // 2, y1 + obj_h // 2), (obj_w // 2, obj_h // 3), 0, 0, 180, color, -1)
                elif cls_id == 3:  # fire: bright flame triangle
                    pts = np.array([[x1 + obj_w // 2, y1], [x1, y2], [x2, y2]], np.int32)
                    cv2.fillPoly(img, [pts], (0, 140, 255))
                else:  # damaged component: notched polygon
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

                # Convert to normalized YOLO format
                xc, yc, norm_w, norm_h = xyxy_abs_to_xywh_norm(x1, y1, x2, y2, w, h)
                annotations.append(f"{cls_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

            # Save image
            img_filename = f"sample_{split}_{i:04d}.jpg"
            img_path = img_dir / img_filename
            cv2.imwrite(str(img_path), img)

            # Save label
            lbl_path = lbl_dir / f"sample_{split}_{i:04d}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(annotations) + "\n")

    # Generate data.yaml
    data_yaml_path = output_root / "data.yaml"
    data_yaml_dict = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)

    logger.info(f"Synthetic dataset created successfully at: {output_root}")
    return data_yaml_path
