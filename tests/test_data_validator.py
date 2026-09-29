"""Unit tests for DatasetValidator edge cases, corruption, and leakage detection."""

import shutil
from pathlib import Path

from PIL import Image

from veyraxis.data.validator import DatasetValidator


def test_validator_on_clean_dataset(temp_dataset_dir: Path):
    """Clean dataset should pass validation with zero fatal errors."""
    classes = ["person", "vehicle", "helmet", "fire", "damaged_component"]
    validator = DatasetValidator(dataset_root=temp_dataset_dir, classes=classes)
    report = validator.validate()

    assert report.is_valid is True
    assert report.total_images == 8
    assert report.total_annotations > 0
    assert len(report.corrupt_images) == 0
    assert len(report.data_leakage) == 0


def test_validator_detects_corrupt_image(temp_dataset_dir: Path):
    """Validator must flag broken or zero-byte image files."""
    corrupt_path = temp_dataset_dir / "images" / "train" / "corrupt_sample.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

    classes = ["person", "vehicle", "helmet", "fire", "damaged_component"]
    validator = DatasetValidator(dataset_root=temp_dataset_dir, classes=classes)
    report = validator.validate()

    assert report.is_valid is False
    assert any("corrupt_sample.jpg" in p for p in report.corrupt_images)

    # Clean up
    corrupt_path.unlink()


def test_validator_detects_out_of_bounds_bbox(temp_dataset_dir: Path):
    """Validator must flag normalized coordinates exceeding 1.0 or below 0.0."""
    bad_lbl_path = temp_dataset_dir / "labels" / "train" / "bad_bbox_sample.txt"
    bad_img_path = temp_dataset_dir / "images" / "train" / "bad_bbox_sample.jpg"

    # Create dummy image
    img = Image.new("RGB", (640, 640), color="blue")
    img.save(bad_img_path)

    # Write out-of-bounds coordinates (x_center = 1.5)
    with open(bad_lbl_path, "w") as f:
        f.write("0 1.500000 0.500000 0.200000 0.200000\n")

    classes = ["person", "vehicle", "helmet", "fire", "damaged_component"]
    validator = DatasetValidator(dataset_root=temp_dataset_dir, classes=classes)
    report = validator.validate()

    has_oob = any(p["category"] == "out_of_bounds_bbox" for p in report.problematic_annotations)
    assert has_oob is True

    # Clean up
    bad_lbl_path.unlink()
    bad_img_path.unlink()


def test_validator_detects_data_leakage(temp_dataset_dir: Path):
    """Validator must detect identical images present across train and test splits."""
    train_img = temp_dataset_dir / "images" / "train" / "sample_train_0000.jpg"
    if train_img.exists():
        # Duplicate exactly into test split
        leaked_test_img = temp_dataset_dir / "images" / "test" / "leaked_from_train.jpg"
        shutil.copy2(train_img, leaked_test_img)

        classes = ["person", "vehicle", "helmet", "fire", "damaged_component"]
        validator = DatasetValidator(dataset_root=temp_dataset_dir, classes=classes)
        report = validator.validate()

        assert report.is_valid is False
        assert len(report.data_leakage) > 0

        # Clean up
        leaked_test_img.unlink()
