"""Dataset validation CLI script for Veyraxis Sentinel."""

import argparse
import sys
from pathlib import Path

import yaml

from veyraxis.data.validator import DatasetValidator
from veyraxis.utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.validate_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset integrity and produce distribution reports.")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to dataset YAML configuration")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dataset_reports",
        help="Directory to save generated charts and reports",
    )
    args = parser.parse_args()

    yaml_path = Path(args.data)
    if not yaml_path.exists():
        logger.error(f"Dataset YAML not found: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)

    root = Path(data_dict.get("path", yaml_path.parent))
    raw_names = data_dict.get("names", {})
    classes = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)

    logger.info(f"Validating dataset at: {root.resolve()} for classes: {classes}")
    validator = DatasetValidator(dataset_root=root, classes=classes)
    report = validator.validate(output_dir=Path(args.output_dir))

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Status:             {'PASSED' if report.is_valid else 'FAILED (Errors Detected)'}")
    print(f"Total Images:       {report.total_images}")
    print(f"Total Annotations:  {report.total_annotations}")
    print(f"Splits breakdown:   {report.splits_count}")
    print(f"Class distribution: {report.class_distribution}")
    print(f"Corrupt images:     {len(report.corrupt_images)}")
    print(f"Data leakage items: {len(report.data_leakage)}")
    print(f"Reports & Plots in: {Path(args.output_dir).resolve()}")
    print("=" * 60 + "\n")

    if not report.is_valid:
        logger.error("Dataset integrity check failed! Fix flagged errors before commencing model training.")
        sys.exit(1)

    logger.info("Dataset validated successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
