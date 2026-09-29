"""Comprehensive runtime benchmarking across model formats and hardware backends."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch

from veyraxis.deployment.onnx_runner import ONNXRunner
from veyraxis.models.detector import VeyraxisDetector
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.deployment.benchmark")


@dataclass
class BenchmarkReport:
    """Format and hardware execution profile."""

    format: str
    device: str
    model_size_mb: float
    cold_start_ms: float
    warmup_passes: int
    benchmark_iterations: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_fps: float
    ram_usage_mb: float
    peak_gpu_memory_mb: float


class ModelBenchmark:
    """
    Benchmarks cold-start time, latency percentiles, throughput (FPS),
    RAM footprint, and GPU memory across model formats.
    """

    def __init__(self, imgsz: int = 640):
        self.imgsz = imgsz
        self.dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    def benchmark_pytorch(
        self,
        model_path: Path,
        device: str = "auto",
        iterations: int = 50,
        warmup: int = 10,
    ) -> BenchmarkReport:
        """Benchmark native PyTorch detector."""
        model_size_mb = float(os.path.getsize(model_path) / (1024 * 1024)) if Path(model_path).exists() else 0.0

        # Measure Cold-Start Time (Initialization + First Inference)
        t_cold_0 = time.perf_counter()
        detector = VeyraxisDetector(model_path=model_path, device=device)
        detector.predict(self.dummy_input, verbose=False)
        cold_start_ms = (time.perf_counter() - t_cold_0) * 1000.0

        # Warmup
        for _ in range(warmup):
            detector.predict(self.dummy_input, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies_ms: list[float] = []
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)

        for _ in range(iterations):
            if torch.cuda.is_available() and detector.device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            detector.predict(self.dummy_input, verbose=False)
            if torch.cuda.is_available() and detector.device != "cpu":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        ram_after = process.memory_info().rss / (1024 * 1024)
        peak_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

        lat_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(lat_arr))
        fps = float(1000.0 / max(mean_lat, 1e-4))

        return BenchmarkReport(
            format="pytorch",
            device=detector.device,
            model_size_mb=round(model_size_mb, 2),
            cold_start_ms=round(cold_start_ms, 2),
            warmup_passes=warmup,
            benchmark_iterations=iterations,
            latency_mean_ms=round(mean_lat, 2),
            latency_p50_ms=round(float(np.percentile(lat_arr, 50)), 2),
            latency_p95_ms=round(float(np.percentile(lat_arr, 95)), 2),
            latency_p99_ms=round(float(np.percentile(lat_arr, 99)), 2),
            throughput_fps=round(fps, 1),
            ram_usage_mb=round(max(ram_after - ram_before, 0.0), 1),
            peak_gpu_memory_mb=round(peak_gpu, 1),
        )

    def benchmark_onnx(
        self,
        onnx_path: Path,
        device: str = "cpu",
        iterations: int = 50,
        warmup: int = 10,
    ) -> BenchmarkReport:
        """Benchmark ONNX Runtime runner."""
        model_size_mb = float(os.path.getsize(onnx_path) / (1024 * 1024))

        t_cold_0 = time.perf_counter()
        runner = ONNXRunner(onnx_path, device=device)
        runner.predict(self.dummy_input)
        cold_start_ms = (time.perf_counter() - t_cold_0) * 1000.0

        for _ in range(warmup):
            runner.predict(self.dummy_input)

        latencies_ms: list[float] = []
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)

        for _ in range(iterations):
            t0 = time.perf_counter()
            runner.predict(self.dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        ram_after = process.memory_info().rss / (1024 * 1024)

        lat_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(lat_arr))
        fps = float(1000.0 / max(mean_lat, 1e-4))

        return BenchmarkReport(
            format="onnx",
            device=runner.active_provider,
            model_size_mb=round(model_size_mb, 2),
            cold_start_ms=round(cold_start_ms, 2),
            warmup_passes=warmup,
            benchmark_iterations=iterations,
            latency_mean_ms=round(mean_lat, 2),
            latency_p50_ms=round(float(np.percentile(lat_arr, 50)), 2),
            latency_p95_ms=round(float(np.percentile(lat_arr, 95)), 2),
            latency_p99_ms=round(float(np.percentile(lat_arr, 99)), 2),
            throughput_fps=round(fps, 1),
            ram_usage_mb=round(max(ram_after - ram_before, 0.0), 1),
            peak_gpu_memory_mb=0.0,
        )
