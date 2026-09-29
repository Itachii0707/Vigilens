"""Fine-grained error analysis, confusion matrix rendering, and HTML report generation."""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from veyraxis.evaluation.metrics import DetectionMetrics
from veyraxis.utils.bboxes import compute_batch_iou, compute_iou
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.evaluation.error_analysis")


@dataclass
class ErrorBreakdown:
    """Error breakdown across operational domains."""

    small_objects_count: int = 0
    small_objects_missed: int = 0
    occluded_overlapping_count: int = 0
    occluded_overlapping_missed: int = 0
    poor_lighting_samples: int = 0
    poor_lighting_misses: int = 0
    blurred_samples: int = 0
    blurred_misses: int = 0
    background_false_positives: int = 0
    class_confusion_pairs: list[tuple[str, str, int]] = field(default_factory=list)


class ErrorAnalyzer:
    """
    Categorizes detection errors across scale, occlusion, lighting, and blur.
    Generates multi-format reports (HTML, JSON, CSV, PNG).
    """

    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def analyze(
        self,
        images_info: list[dict[str, Any]],  # [{ 'path': ..., 'image': np.ndarray, 'gt': (M, 5), 'preds': (N, 6) }]
        metrics: DetectionMetrics,
        output_dir: Path,
    ) -> ErrorBreakdown:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_sample_dir = output_dir / "annotated_predictions"
        pred_sample_dir.mkdir(parents=True, exist_ok=True)

        breakdown = ErrorBreakdown()
        confusion_tallies: dict[tuple[int, int], int] = {}

        for idx, item in enumerate(images_info):
            img = item["image"]
            h, w = img.shape[:2]
            gts = item["gt"]  # (M, 5) [cls, x1, y1, x2, y2]
            preds = item["preds"]  # (N, 6) [x1, y1, x2, y2, conf, cls]

            # Assess lighting (mean luminance in Y channel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_poor_lighting = float(np.mean(gray)) < 50.0
            if is_poor_lighting:
                breakdown.poor_lighting_samples += 1

            # Assess blur (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurred = laplacian_var < 80.0
            if is_blurred:
                breakdown.blurred_samples += 1

            # Match predictions to GT
            matched_gts = set()
            if len(gts) > 0 and len(preds) > 0:
                ious = compute_batch_iou(gts[:, 1:5], preds[:, :4])
                for p_idx in range(len(preds)):
                    best_gt = -1
                    best_iou = 0.5
                    for g_idx in range(len(gts)):
                        if g_idx not in matched_gts and ious[g_idx, p_idx] > best_iou:
                            best_iou = ious[g_idx, p_idx]
                            best_gt = g_idx
                    if best_gt >= 0:
                        matched_gts.add(best_gt)
                        p_cls = int(preds[p_idx, 5])
                        g_cls = int(gts[best_gt, 0])
                        if p_cls != g_cls:
                            pair = (g_cls, p_cls)
                            confusion_tallies[pair] = confusion_tallies.get(pair, 0) + 1
                    else:
                        breakdown.background_false_positives += 1
            elif len(preds) > 0 and len(gts) == 0:
                breakdown.background_false_positives += len(preds)

            # Check ground truth misses
            for g_idx, gt in enumerate(gts):
                g_cls, gx1, gy1, gx2, gy2 = gt
                gw, gh = (gx2 - gx1), (gy2 - gy1)
                is_small = (gw * gh) < (32 * 32)
                if is_small:
                    breakdown.small_objects_count += 1

                # Check if overlapping with another GT
                is_overlapping = False
                for other_idx, other_gt in enumerate(gts):
                    if g_idx != other_idx:
                        if compute_iou(gt[1:5], other_gt[1:5]) > 0.3:
                            is_overlapping = True
                            break
                if is_overlapping:
                    breakdown.occluded_overlapping_count += 1

                if g_idx not in matched_gts:
                    if is_small:
                        breakdown.small_objects_missed += 1
                    if is_overlapping:
                        breakdown.occluded_overlapping_missed += 1
                    if is_poor_lighting:
                        breakdown.poor_lighting_misses += 1
                    if is_blurred:
                        breakdown.blurred_misses += 1

            # Save sample annotated preview for first 10 items
            if idx < 10:
                vis_img = img.copy()
                # Draw Ground Truth in Green
                for gt in gts:
                    g_cls, gx1, gy1, gx2, gy2 = map(int, gt)
                    cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cname = self.class_names[g_cls] if g_cls < self.num_classes else f"cls_{g_cls}"
                    cv2.putText(
                        vis_img, f"GT:{cname}", (gx1, max(gy1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
                    )

                # Draw Predictions in Magenta
                for p in preds:
                    px1, py1, px2, py2 = map(int, p[:4])
                    pconf = p[4]
                    p_cls = int(p[5])
                    cv2.rectangle(vis_img, (px1, py1), (px2, py2), (255, 0, 255), 2)
                    cname = self.class_names[p_cls] if p_cls < self.num_classes else f"cls_{p_cls}"
                    cv2.putText(
                        vis_img,
                        f"{cname} {pconf:.2f}",
                        (px1, max(py1 - 18, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                    )

                cv2.imwrite(str(pred_sample_dir / f"eval_pred_{idx:03d}.jpg"), vis_img)

        # Record confusion pairs
        for (g_cls, p_cls), count in confusion_tallies.items():
            g_name = self.class_names[g_cls] if g_cls < self.num_classes else f"cls_{g_cls}"
            p_name = self.class_names[p_cls] if p_cls < self.num_classes else f"cls_{p_cls}"
            breakdown.class_confusion_pairs.append((g_name, p_name, count))

        # Save results to disk
        self._export_results(metrics, breakdown, output_dir)
        return breakdown

    def _export_results(self, metrics: DetectionMetrics, breakdown: ErrorBreakdown, output_dir: Path) -> None:
        """Write JSON, CSV, Confusion Matrix PNG, and HTML Report."""
        # 1. JSON
        combined = {
            "metrics": metrics.to_dict(),
            "error_breakdown": asdict(breakdown),
        }
        with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        # 2. CSV Per-Class Performance
        with open(output_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "precision", "recall", "ap50"])
            for cname in self.class_names:
                prec = metrics.per_class_precision.get(cname, 0.0)
                rec = metrics.per_class_recall.get(cname, 0.0)
                ap = metrics.per_class_ap50.get(cname, 0.0)
                writer.writerow([cname, f"{prec:.4f}", f"{rec:.4f}", f"{ap:.4f}"])

        # 3. Confusion Matrix Plot
        if metrics.confusion_matrix:
            cm = np.array(metrics.confusion_matrix)
            labels = self.class_names + ["background"]
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=labels,
                yticklabels=labels,
                title="Confusion Matrix (including background)",
                ylabel="True Label",
                xlabel="Predicted Label",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2.0 else "black",
                    )
            fig.tight_layout()
            plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
            plt.close()

        # 4. Interactive HTML Report
        self._write_html_report(metrics, breakdown, output_dir)

    def _write_html_report(self, metrics: DetectionMetrics, breakdown: ErrorBreakdown, output_dir: Path) -> None:
        """Render a standalone executive HTML report."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Veyraxis Sentinel - Model Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ border-bottom: 2px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
        h1 {{ color: #38bdf8; margin: 0 0 8px 0; font-size: 26px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 18px; border: 1px solid #334155; }}
        .card-title {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 6px; }}
        .card-value {{ font-size: 28px; font-weight: 700; color: #f1f5f9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0f172a; color: #38bdf8; text-transform: uppercase; font-size: 12px; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
        .badge-good {{ background: #14532d; color: #4ade80; }}
        .badge-warn {{ background: #78350f; color: #fde047; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; border-radius: 8px; border: 1px solid #334155; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>VEY RAXIS SENTINEL &mdash; EVALUATION REPORT</h1>
        <p style="color: #94a3b8; margin: 0;">Comprehensive Isolated Test-Set Benchmarks and Error Diagnostics</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">mAP @ 0.50</div>
            <div class="card-value">{metrics.map50:.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">mAP @ 0.50:0.95</div>
            <div class="card-value">{metrics.map50_95:.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Precision</div>
            <div class="card-value">{metrics.precision:.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Recall</div>
            <div class="card-value">{metrics.recall:.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Inference Latency (P50)</div>
            <div class="card-value">{metrics.latency_p50_ms:.1f} <span style="font-size: 16px;">ms</span></div>
        </div>
        <div class="card">
            <div class="card-title">Throughput</div>
            <div class="card-value">{metrics.fps:.1f} <span style="font-size: 16px;">FPS</span></div>
        </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
        <h2 style="color: #38bdf8; font-size: 18px; margin-top: 0;">Per-Class Detection Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>AP @ 0.50</th>
                    <th>F1 Score</th>
                </tr>
            </thead>
            <tbody>
"""
        for cname in self.class_names:
            p = metrics.per_class_precision.get(cname, 0.0)
            r = metrics.per_class_recall.get(cname, 0.0)
            ap = metrics.per_class_ap50.get(cname, 0.0)
            f1 = (2 * p * r) / max(p + r, 1e-6)
            html_content += f"""
                <tr>
                    <td><strong>{cname}</strong></td>
                    <td>{p:.3f}</td>
                    <td>{r:.3f}</td>
                    <td>{ap:.3f}</td>
                    <td><span class="badge {"badge-good" if f1 > 0.6 else "badge-warn"}">{f1:.3f}</span></td>
                </tr>
"""
        html_content += f"""
            </tbody>
        </table>
    </div>

    <div class="card" style="margin-bottom: 24px;">
        <h2 style="color: #38bdf8; font-size: 18px; margin-top: 0;">Error Analysis & Failure Modes</h2>
        <table>
            <thead>
                <tr>
                    <th>Condition</th>
                    <th>Total Occurrences</th>
                    <th>Missed Instances</th>
                    <th>Failure Rate</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Small Objects (&lt; 32x32 px)</td>
                    <td>{breakdown.small_objects_count}</td>
                    <td>{breakdown.small_objects_missed}</td>
                    <td>{(breakdown.small_objects_missed / max(breakdown.small_objects_count, 1)) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Occluded / Overlapping Objects</td>
                    <td>{breakdown.occluded_overlapping_count}</td>
                    <td>{breakdown.occluded_overlapping_missed}</td>
                    <td>{(breakdown.occluded_overlapping_missed / max(breakdown.occluded_overlapping_count, 1)) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Poor Lighting / Low Contrast</td>
                    <td>{breakdown.poor_lighting_samples}</td>
                    <td>{breakdown.poor_lighting_misses}</td>
                    <td>{(breakdown.poor_lighting_misses / max(breakdown.poor_lighting_samples, 1)) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Camera Shutter Blur</td>
                    <td>{breakdown.blurred_samples}</td>
                    <td>{breakdown.blurred_misses}</td>
                    <td>{(breakdown.blurred_misses / max(breakdown.blurred_samples, 1)) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Background False Positives</td>
                    <td>-</td>
                    <td>{breakdown.background_false_positives}</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2 style="color: #38bdf8; font-size: 18px; margin-top: 0;">Confusion Matrix</h2>
        <div class="chart-container">
            <img src="confusion_matrix.png" alt="Confusion Matrix">
        </div>
    </div>
</div>
</body>
</html>
"""
        with open(output_dir / "evaluation_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML evaluation report compiled to: {output_dir / 'evaluation_report.html'}")
