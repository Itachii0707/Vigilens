"""Multi-source dataset download, API ingestion, and split partition utility for Veyraxis Sentinel."""

import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml
from ultralytics.data.utils import check_det_dataset

from veyraxis.data.validator import DatasetValidator
from veyraxis.utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.download_dataset")


def ingest_coco128(
    target_root: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Download COCO128 (80 real-world object classes) and partition into train/val/test splits."""
    logger.info("Downloading and preparing COCO128 dataset (80 object categories)...")
    dataset_info = check_det_dataset("coco128.yaml")
    source_path = Path(dataset_info["path"])

    source_imgs = list((source_path / "images" / "train2017").glob("*.jpg"))
    source_lbls = source_path / "labels" / "train2017"

    if not source_imgs:
        raise FileNotFoundError(f"No images found in {source_path / 'images' / 'train2017'}")

    logger.info(f"Loaded {len(source_imgs)} annotated images from COCO128.")

    random.seed(seed)
    shuffled_imgs = list(source_imgs)
    random.shuffle(shuffled_imgs)

    n_total = len(shuffled_imgs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": shuffled_imgs[:n_train],
        "val": shuffled_imgs[n_train : n_train + n_val],
        "test": shuffled_imgs[n_train + n_val :],
    }

    target_root = Path(target_root)
    for split_name in ("train", "val", "test"):
        img_dir = target_root / "images" / split_name
        lbl_dir = target_root / "labels" / split_name
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    for split_name, img_list in splits.items():
        img_dest = target_root / "images" / split_name
        lbl_dest = target_root / "labels" / split_name
        for img_p in img_list:
            shutil.copy2(img_p, img_dest / img_p.name)
            src_lbl = source_lbls / f"{img_p.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_dest / f"{img_p.stem}.txt")

        logger.info(f"Split '{split_name}': {len(img_list)} images copied.")

    names_dict = dataset_info["names"]
    data_yaml_dict = {
        "path": str(target_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_dict,
    }

    yaml_dest = target_root / "data.yaml"
    with open(yaml_dest, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)

    with open(Path("data/dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)

    logger.info(f"Dataset configuration saved to: {yaml_dest} and data/dataset.yaml (Classes: {len(names_dict)})")
    return yaml_dest


def ingest_pascal_voc(target_root: Path) -> Path:
    """Download large Pascal VOC dataset (16,551 annotated real images, 20 object categories)."""
    logger.info("Downloading large Pascal VOC dataset (16,551 images, 20 classes)... This may take a few minutes.")
    dataset_info = check_det_dataset("VOC.yaml")
    source_path = Path(dataset_info["path"])

    logger.info(f"Pascal VOC downloaded to: {source_path}")
    yaml_dest = target_root / "data.yaml"
    names_dict = dataset_info["names"]

    data_yaml_dict = {
        "path": str(source_path.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": names_dict,
    }

    with open(yaml_dest, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)
    with open(Path("data/dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)

    return yaml_dest


def ingest_from_url(url: str, target_root: Path) -> Path:
    """Download any public YOLO dataset ZIP archive from a direct URL and unpack."""
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    temp_zip = target_root / "temp_download.zip"

    logger.info(f"Downloading dataset archive from: {url}...")
    urllib.request.urlretrieve(url, temp_zip)

    logger.info("Extracting dataset archive...")
    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(target_root)

    temp_zip.unlink(missing_ok=True)
    logger.info(f"Dataset extracted to {target_root.resolve()}")
    return target_root / "data.yaml"


def ingest_from_roboflow(api_key: str, workspace: str, project: str, version: int, target_root: Path) -> Path:
    """Download any open-source or private dataset via Roboflow API in YOLOv11 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.info("Installing roboflow SDK...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    logger.info(f"Fetching dataset {workspace}/{project}/{version} via Roboflow API...")
    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_dataset = rf_project.version(version).download("yolov11", location=str(target_root))

    logger.info(f"Roboflow dataset downloaded to: {rf_dataset.location}")
    return Path(rf_dataset.location) / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare multi-category real-world object detection datasets."
    )
    parser.add_argument(
        "--dataset", type=str, default="coco128", choices=["coco128", "voc", "url", "roboflow"], help="Dataset source"
    )
    parser.add_argument("--url", type=str, default=None, help="Direct ZIP URL if --dataset url is selected")
    parser.add_argument("--target-dir", type=str, default="dataset", help="Target dataset directory")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", type=str, default=None, help="Roboflow workspace slug")
    parser.add_argument("--roboflow-project", type=str, default=None, help="Roboflow project slug")
    parser.add_argument("--roboflow-version", type=int, default=1, help="Roboflow dataset version")
    args = parser.parse_args()

    try:
        if args.dataset == "coco128":
            yaml_path = ingest_coco128(target_root=Path(args.target_dir))
        elif args.dataset == "voc":
            yaml_path = ingest_pascal_voc(target_root=Path(args.target_dir))
        elif args.dataset == "url":
            if not args.url:
                raise ValueError("Must provide --url when --dataset url is chosen.")
            yaml_path = ingest_from_url(url=args.url, target_root=Path(args.target_dir))
        elif args.dataset == "roboflow":
            if not (args.roboflow_key and args.roboflow_workspace and args.roboflow_project):
                raise ValueError("Must provide --roboflow-key, --roboflow-workspace, and --roboflow-project.")
            yaml_path = ingest_from_roboflow(
                api_key=args.roboflow_key,
                workspace=args.roboflow_workspace,
                project=args.roboflow_project,
                version=args.roboflow_version,
                target_root=Path(args.target_dir),
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Run automated validation if local images exist
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            raw_names = d.get("names", {})
            classes = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)

            root = Path(d.get("path", Path(args.target_dir)))
            validator = DatasetValidator(dataset_root=root, classes=classes)
            report = validator.validate(output_dir=Path("outputs/dataset_reports"))

            print("\n" + "=" * 60)
            print("DATASET INGESTION & INTEGRITY SUMMARY")
            print("=" * 60)
            print(f"Status:             {'PASSED' if report.is_valid else 'WARNINGS DETECTED'}")
            print(f"Total Real Images:  {report.total_images}")
            print(f"Total Annotations:  {report.total_annotations}")
            print(f"Total Classes:      {len(classes)}")
            print(f"Splits:             {report.splits_count}")
            print(f"Config YAML:        {yaml_path.resolve()}")
            print("Reports & Charts:   outputs/dataset_reports")
            print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Dataset ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
