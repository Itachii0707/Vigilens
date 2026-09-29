"""Dataset integrity validator and statistical analysis for Veyraxis Sentinel."""

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from veyraxis.utils.bboxes import xywh_norm_to_xyxy_abs
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.data.validator")


@dataclass
class ProblemReport:
    """Record of individual dataset defects."""

    category: str
    filepath: str
    details: str
    line_number: int | None = None


@dataclass
class ValidationReport:
    """Aggregated dataset validation and statistical profile."""

    total_images: int = 0
    total_annotations: int = 0
    splits_count: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    background_images_count: int = 0
    corrupt_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    empty_label_files: list[str] = field(default_factory=list)
    duplicate_images: list[tuple[str, str]] = field(default_factory=list)
    data_leakage: list[tuple[str, str, str]] = field(default_factory=list)  # (hash, splitA/file, splitB/file)
    problematic_annotations: list[dict[str, Any]] = field(default_factory=list)
    resolutions: list[tuple[int, int]] = field(default_factory=list)
    bbox_aspect_ratios: list[float] = field(default_factory=list)
    bbox_normalized_areas: list[float] = field(default_factory=list)
    is_valid: bool = True


class DatasetValidator:
    """
    Comprehensive validator for YOLO-formatted object detection datasets.
    """

    def __init__(
        self,
        dataset_root: Path,
        classes: list[str],
        splits: list[str] = ["train", "val", "test"],
        min_bbox_area: float = 0.0001,
        max_bbox_area: float = 0.98,
    ):
        self.root = Path(dataset_root)
        self.classes = classes
        self.num_classes = len(classes)
        self.class_map = {i: name for i, name in enumerate(classes)}
        self.splits = splits
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area

        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def validate(self, output_dir: Path | None = None) -> ValidationReport:
        """Run full suite of validation checks and optionally generate reports and plots."""
        report = ValidationReport()
        problems: list[ProblemReport] = []

        image_hashes: dict[str, tuple[str, Path]] = {}  # sha256 -> (split, filepath)
        all_resolutions: list[tuple[int, int]] = []
        all_bbox_areas: list[float] = []
        all_aspect_ratios: list[float] = []
        class_counts: Counter = Counter()

        for split in self.splits:
            img_dir = self.root / "images" / split
            lbl_dir = self.root / "labels" / split

            if not img_dir.exists():
                logger.warning(f"Image directory does not exist: {img_dir}")
                continue

            images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in self.image_extensions]
            report.splits_count[split] = len(images)
            report.total_images += len(images)

            for img_path in images:
                # 1. Image integrity check
                is_corrupt, img_size, img_hash = self._check_image(img_path)
                if is_corrupt:
                    report.corrupt_images.append(str(img_path))
                    problems.append(ProblemReport("corrupt_image", str(img_path), "File header damaged or unreadable"))
                    continue

                all_resolutions.append(img_size)

                # 2. Duplicate detection & Data Leakage across splits
                if img_hash in image_hashes:
                    orig_split, orig_path = image_hashes[img_hash]
                    if orig_split != split:
                        report.data_leakage.append(
                            (img_hash, f"{orig_split}:{orig_path.name}", f"{split}:{img_path.name}")
                        )
                        problems.append(
                            ProblemReport(
                                "data_leakage", str(img_path), f"Exact match with {orig_split}:{orig_path.name}"
                            )
                        )
                    else:
                        report.duplicate_images.append((str(orig_path), str(img_path)))
                        problems.append(
                            ProblemReport("duplicate_image", str(img_path), f"Duplicate of {orig_path.name}")
                        )
                else:
                    image_hashes[img_hash] = (split, img_path)

                # 3. Associated label check
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    report.missing_labels.append(str(lbl_path))
                    report.background_images_count += 1
                    continue

                # 4. Label annotations content check
                annotations, label_problems = self._validate_label_file(lbl_path, img_size)
                problems.extend(label_problems)

                if not annotations:
                    report.empty_label_files.append(str(lbl_path))
                    report.background_images_count += 1
                else:
                    for ann in annotations:
                        cls_id, xc, yc, w, h = ann
                        class_counts[self.class_map.get(cls_id, f"unknown_{cls_id}")] += 1
                        area = w * h
                        all_bbox_areas.append(area)
                        ar = w / max(h, 1e-6)
                        all_aspect_ratios.append(ar)
                        report.total_annotations += 1

            # 5. Check orphan labels (label exists without image)
            if lbl_dir.exists():
                labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
                for lbl_path in labels:
                    matching_img = any((img_dir / f"{lbl_path.stem}{ext}").exists() for ext in self.image_extensions)
                    if not matching_img:
                        report.missing_images.append(str(lbl_path))
                        problems.append(
                            ProblemReport(
                                "missing_image", str(lbl_path), f"Label exists but no matching image in {img_dir}"
                            )
                        )

        report.class_distribution = dict(class_counts)
        report.resolutions = all_resolutions
        report.bbox_normalized_areas = all_bbox_areas
        report.bbox_aspect_ratios = all_aspect_ratios
        report.problematic_annotations = [asdict(p) for p in problems]

        # Determine overall validity
        if report.corrupt_images or report.data_leakage:
            report.is_valid = False

        if output_dir:
            self._generate_reports_and_plots(report, problems, output_dir)

        return report

    def _check_image(self, path: Path) -> tuple[bool, tuple[int, int], str]:
        """Check image readability, resolution, and sha256 hash."""
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                size = (img.width, img.height)
            # Compute quick content hash
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return False, size, hasher.hexdigest()
        except Exception:
            return True, (0, 0), ""

    def _validate_label_file(
        self, lbl_path: Path, img_size: tuple[int, int]
    ) -> tuple[list[tuple[int, float, float, float, float]], list[ProblemReport]]:
        """Validate all YOLO bounding box lines in a single label text file."""
        valid_anns: list[tuple[int, float, float, float, float]] = []
        problems: list[ProblemReport] = []
        seen_boxes: set[tuple[int, float, float, float, float]] = set()

        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            problems.append(ProblemReport("corrupt_label", str(lbl_path), f"Cannot read label: {e}"))
            return valid_anns, problems

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 5:
                problems.append(
                    ProblemReport("invalid_format", str(lbl_path), f"Expected 5 fields, got {len(parts)}", line_num)
                )
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                problems.append(
                    ProblemReport("invalid_types", str(lbl_path), f"Invalid non-numeric values: {parts}", line_num)
                )
                continue

            # Class ID validity
            if cls_id < 0 or cls_id >= self.num_classes:
                problems.append(
                    ProblemReport(
                        "invalid_class_id",
                        str(lbl_path),
                        f"Class {cls_id} out of bounds [0, {self.num_classes - 1}]",
                        line_num,
                    )
                )
                continue

            # Bounding box bounds check
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                problems.append(
                    ProblemReport(
                        "out_of_bounds_bbox",
                        str(lbl_path),
                        f"Normalized bbox outside [0, 1]: ({xc}, {yc}, {w}, {h})",
                        line_num,
                    )
                )
                continue

            # Duplicate annotation check
            ann_key = (cls_id, round(xc, 4), round(yc, 4), round(w, 4), round(h, 4))
            if ann_key in seen_boxes:
                problems.append(
                    ProblemReport(
                        "duplicate_annotation",
                        str(lbl_path),
                        f"Duplicate annotation found on line {line_num}",
                        line_num,
                    )
                )
                continue
            seen_boxes.add(ann_key)

            # Area abnormality check
            area = w * h
            if area < self.min_bbox_area:
                problems.append(
                    ProblemReport(
                        "extremely_small_bbox",
                        str(lbl_path),
                        f"Normalized area {area:.6f} below {self.min_bbox_area}",
                        line_num,
                    )
                )
            elif area > self.max_bbox_area:
                problems.append(
                    ProblemReport(
                        "extremely_large_bbox",
                        str(lbl_path),
                        f"Normalized area {area:.4f} above {self.max_bbox_area}",
                        line_num,
                    )
                )

            valid_anns.append((cls_id, xc, yc, w, h))

        return valid_anns, problems

    def _generate_reports_and_plots(
        self, report: ValidationReport, problems: list[ProblemReport], output_dir: Path
    ) -> None:
        """Render JSON/CSV summaries, markdown report, distribution charts, and annotated samples."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Summary JSON
        summary_dict = {
            "is_valid": report.is_valid,
            "total_images": report.total_images,
            "total_annotations": report.total_annotations,
            "splits": report.splits_count,
            "class_distribution": report.class_distribution,
            "background_images_count": report.background_images_count,
            "corrupt_images_count": len(report.corrupt_images),
            "missing_labels_count": len(report.missing_labels),
            "missing_images_count": len(report.missing_images),
            "duplicate_images_count": len(report.duplicate_images),
            "data_leakage_count": len(report.data_leakage),
            "total_problems_flagged": len(problems),
        }
        with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        # 2. Class Summary CSV
        with open(output_dir / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "count", "percentage"])
            total = max(report.total_annotations, 1)
            for idx, cname in enumerate(self.classes):
                cnt = report.class_distribution.get(cname, 0)
                writer.writerow([idx, cname, cnt, f"{(cnt / total) * 100:.2f}%"])

        # 3. Problematic annotation report (Markdown)
        with open(output_dir / "problematic_annotations.md", "w", encoding="utf-8") as f:
            f.write("# Dataset Validation & Problematic Annotation Report\n\n")
            f.write(f"- **Overall Status**: {'PASSED' if report.is_valid else 'FAILED (Issues Detected)'}\n")
            f.write(f"- **Total Images**: {report.total_images}\n")
            f.write(f"- **Total Annotations**: {report.total_annotations}\n")
            f.write(f"- **Data Leakage Issues**: {len(report.data_leakage)}\n")
            f.write(f"- **Corrupt Images**: {len(report.corrupt_images)}\n\n")

            if problems:
                f.write("## Identified Issues\n\n")
                f.write("| Category | File | Line | Details |\n")
                f.write("| --- | --- | --- | --- |\n")
                for p in problems[:200]:  # Limit top 200 in markdown table
                    f.write(f"| {p.category} | `{Path(p.filepath).name}` | {p.line_number or '-'} | {p.details} |\n")
            else:
                f.write("No validation anomalies detected. Dataset conforms to YOLO specifications.\n")

        # 4. Generate Visual Charts (Class distribution, BBox sizes, Resolutions)
        self._plot_charts(report, output_dir)

        # 5. Draw random annotated samples
        self._save_sample_annotated_images(output_dir)

    def _plot_charts(self, report: ValidationReport, out_dir: Path) -> None:
        """Generate high resolution diagnostic charts."""
        plt.style.use("default")

        # Chart 1: Class distribution
        if report.class_distribution:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            classes = list(report.class_distribution.keys())
            counts = list(report.class_distribution.values())
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
            bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.8)
            ax.set_title("Veyraxis Sentinel: Class Distribution", fontsize=12, fontweight="bold")
            ax.set_ylabel("Instance Count", fontsize=10)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=25, ha="right")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.tight_layout()
            plt.savefig(out_dir / "class_distribution.png", dpi=150)
            plt.close()

        # Chart 2: Bounding Box Size Histogram
        if report.bbox_normalized_areas:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(report.bbox_normalized_areas, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.85)
            ax.set_title("Bounding Box Area Distribution (Normalized)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Relative Area (W * H)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / "bbox_size_distribution.png", dpi=150)
            plt.close()

        # Chart 3: Image Resolution Scatter
        if report.resolutions:
            widths = [w for w, h in report.resolutions]
            heights = [h for w, h in report.resolutions]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(widths, heights, color="#d95f02", alpha=0.6, edgecolors="none")
            ax.set_title("Image Resolution Distribution", fontsize=11, fontweight="bold")
            ax.set_xlabel("Width (px)", fontsize=10)
            ax.set_ylabel("Height (px)", fontsize=10)
            plt.tight_layout()
            plt.savefig(out_dir / "image_resolutions.png", dpi=150)
            plt.close()

    def _save_sample_annotated_images(self, out_dir: Path, num_samples: int = 5) -> None:
        """Render and save annotated sample previews with bounding box overlays."""
        sample_dir = out_dir / "annotated_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        candidate_images = []
        for split in self.splits:
            img_dir = self.root / "images" / split
            if img_dir.exists():
                candidate_images.extend([p for p in img_dir.iterdir() if p.suffix.lower() in self.image_extensions])

        if not candidate_images:
            return

        sampled = random.sample(candidate_images, min(num_samples, len(candidate_images)))
        colors = [
            (255, 50, 50),  # Red
            (50, 200, 50),  # Green
            (50, 50, 255),  # Blue
            (255, 165, 0),  # Orange
            (180, 50, 220),  # Purple
        ]

        for idx, img_path in enumerate(sampled):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Look up label in corresponding split
            split_name = img_path.parent.name
            lbl_path = self.root / "labels" / split_name / f"{img_path.stem}.txt"
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                            x1, y1, x2, y2 = xywh_norm_to_xyxy_abs(xc, yc, bw, bh, w, h)
                            color = colors[cls_id % len(colors)]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            label_name = self.class_map.get(cls_id, f"ID:{cls_id}")
                            cv2.putText(
                                img,
                                label_name,
                                (x1, max(y1 - 6, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

            cv2.imwrite(str(sample_dir / f"sample_{idx + 1}_{img_path.stem}.jpg"), img)
