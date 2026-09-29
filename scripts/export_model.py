"""Model export and latency benchmarking CLI script for Veyraxis Sentinel."""

import argparse
import json
import sys
from pathlib import Path

from veyraxis.deployment.benchmark import ModelBenchmark
from veyraxis.deployment.exporter import ModelExporter
from veyraxis.utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.export_model")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Veyraxis Sentinel model to ONNX, TensorRT, or OpenVINO with parity verification."
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to PyTorch checkpoint to export")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "engine", "openvino", "torchscript"],
        help="Target format",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--half", action="store_true", help="Export in FP16 half precision")
    parser.add_argument(
        "--dynamic", action="store_true", default=True, help="Enable dynamic batch and spatial axes for ONNX"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate numerical and spatial parity against PyTorch baseline",
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=True, help="Run latency, cold-start, and throughput benchmark"
    )
    parser.add_argument("--output-dir", type=str, default="outputs/exports", help="Directory to save export reports")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.warning(f"Specified model path {model_path} not found. Attempting to resolve via base weights...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        exporter = ModelExporter(model_path=model_path)
        exported_path = exporter.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
        )

        val_result = None
        if args.validate and args.format == "onnx":
            logger.info("Executing mathematical and spatial parity verification against PyTorch baseline...")
            val_result = exporter.validate_parity(exported_path)

        bench_report = None
        if args.benchmark:
            logger.info("Benchmarking runtime performance...")
            bench = ModelBenchmark(imgsz=args.imgsz)
            if args.format == "onnx":
                bench_report = bench.benchmark_onnx(exported_path, device="cpu", iterations=30)
            else:
                bench_report = bench.benchmark_pytorch(model_path, iterations=30)

        # Print summary
        print("\n" + "=" * 60)
        print("MODEL EXPORT & BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Export Format:       {args.format.upper()}")
        print(f"Exported Path:       {exported_path}")
        print(f"Exported File Size:  {exported_path.stat().st_size / (1024 * 1024):.2f} MB")

        if val_result:
            print(f"Parity Check:        {val_result.validation_status}")
            print(f"Max Conf Difference: {val_result.max_score_difference:.4f}")
            print(f"Mean Box IoU:        {val_result.mean_bbox_iou:.4f}")

        if bench_report:
            print(f"Cold Start Time:     {bench_report.cold_start_ms:.2f} ms")
            print(f"P50 Latency:         {bench_report.latency_p50_ms:.2f} ms")
            print(f"P95 Latency:         {bench_report.latency_p95_ms:.2f} ms")
            print(f"Throughput:          {bench_report.throughput_fps:.1f} FPS")
            print(f"Active Provider:     {bench_report.device}")
        print("=" * 60 + "\n")

        # Save summary JSON
        export_summary = {
            "format": args.format,
            "exported_file": str(exported_path),
            "file_size_mb": round(exported_path.stat().st_size / (1024 * 1024), 2),
            "validation": val_result.__dict__ if val_result else None,
            "benchmark": bench_report.__dict__ if bench_report else None,
        }
        with open(output_dir / f"export_{args.format}_summary.json", "w", encoding="utf-8") as f:
            json.dump(export_summary, f, indent=2)

    except Exception as e:
        logger.error(f"Model export or benchmarking failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
