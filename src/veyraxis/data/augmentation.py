"""Data augmentation pipelines with Albumentations for Veyraxis Sentinel."""

import albumentations as A


def get_training_augmentation(
    imgsz: int = 640,
    enable_cutout: bool = True,
    cutout_prob: float = 0.25,
) -> A.Compose:
    """
    Build a realistic data augmentation pipeline for object detection.

    Augmentation Rationale and Risk Analysis:
    -----------------------------------------
    1. HorizontalFlip (p=0.5):
       - Purpose: Simulates symmetric object poses (e.g. person walking left vs right, vehicle moving opposite direction).
       - Risk: Low for generic objects; do not use if text or directional signs are semantic targets.

    2. ShiftScaleRotate (p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=10):
       - Purpose: Handles slight camera tilt, perspective drift, and varying camera-to-object distances.
       - Risk: Excessive rotation (>15 deg) can distort bounding box tight-fit or introduce black border artifacts.

    3. RandomCrop / CropAndPad (handled conservatively):
       - Purpose: Simulates partial sensor framing and close-up views.
       - Risk: Risk of cropping out entire objects; bounding box min_visibility=0.3 is enforced.

    4. ColorJitter (p=0.4, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
       - Purpose: Simulates changing environmental lighting (indoor vs outdoor, shadows, cloudy weather).
       - Risk: Over-saturation or severe hue shifts can destroy semantic color cues (e.g. fire vs orange clothing).

    5. MotionBlur / GaussianBlur (p=0.2):
       - Purpose: Simulates camera shutter blur from moving targets or fast-panning PTZ cameras.
       - Risk: Extreme blur degrades edge details necessary for small object localization.

    6. GaussNoise / MultiplicativeNoise (p=0.2):
       - Purpose: Simulates low-light sensor noise (ISO gain artifacts in night-vision / RTSP streams).
       - Risk: High noise can mask low-contrast object edges.

    7. CoarseDropout / Cutout (p=0.25):
       - Purpose: Forces the network to learn distributed feature representations and handle partial occlusions.
       - Risk: May obscure small objects entirely; hole dimensions must be restrained relative to object scale.
    """
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=0,
            p=0.5,
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.4,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            ],
            p=0.2,
        ),
        A.GaussNoise(p=0.2),
    ]

    if enable_cutout:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 48),
                hole_width_range=(16, 48),
                fill=0,
                p=cutout_prob,
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            check_each_transform=True,
        ),
    )


def get_validation_augmentation() -> A.Compose:
    """
    Deterministic validation pipeline with identity transform for ground truth verification.
    """
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.0,
            check_each_transform=False,
        ),
    )
