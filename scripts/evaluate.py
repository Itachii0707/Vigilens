"""Model evaluation CLI script for Veyraxis Sentinel."""

import argparse
import sys
from pathlib import Path

from veyraxis.evaluation.evaluator import VeyraxisEvaluator
from veyraxis.utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Veyraxis Sentinel model on isolated test set.")
    parser.add_argument("--model", type=str, default="runs/checkpoints/best.pt", help="Path to model weights file")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to data.yaml")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evaluation", help="Directory to save evaluation reports and plots"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    model_path = Path(args.model)
    # Fallback to base weights if fine-tuned checkpoint does not exist yet
    if not model_path.exists():
        logger.warning(f"Checkpoint {model_path} not found. Using baseline weights 'yolo11n.pt' for evaluation.")
        model_path = Path("yolo11n.pt")

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset configuration not found: {data_path}")
        sys.exit(1)

    evaluator = VeyraxisEvaluator(
        model_path=model_path,
        data_yaml=data_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    try:
        metrics = evaluator.evaluate(output_dir=Path(args.output_dir))
        print("\n" + "=" * 60)
        print("EVALUATION BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"mAP @ 0.50:          {metrics.map50:.4f}")
        print(f"mAP @ 0.50:0.95:     {metrics.map50_95:.4f}")
        print(f"Overall Precision:   {metrics.precision:.4f}")
        print(f"Overall Recall:      {metrics.recall:.4f}")
        print(f"F1 Score:            {metrics.f1_score:.4f}")
        print(f"Inference Latency:   {metrics.latency_p50_ms:.2f} ms (P50) | {metrics.latency_p95_ms:.2f} ms (P95)")
        print(f"Throughput:          {metrics.fps:.1f} FPS")
        print(f"Peak GPU VRAM:       {metrics.peak_gpu_memory_mb:.1f} MB")
        print(f"Model File Size:     {metrics.model_size_mb:.2f} MB")
        print(f"Executive Report:    {Path(args.output_dir).resolve() / 'evaluation_report.html'}")
        print("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
