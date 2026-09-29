"""Application dependencies, singleton model loader, and telemetry tracker."""

import asyncio
import time
from collections import deque
from pathlib import Path

from veyraxis.inference.engine import InferenceEngine
from veyraxis.inference.visualizer import Visualizer
from veyraxis.utils.config import APISettings
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.api.dependencies")


class ModelContainer:
    """Singleton container managing model lifecycle, thread safety, and telemetry."""

    def __init__(self):
        self.engine: InferenceEngine | None = None
        self.engines: dict[str, InferenceEngine] = {}
        self.visualizer: Visualizer | None = None
        self.settings: APISettings = APISettings()
        self.start_time: float = time.time()

        # Telemetry counters
        self.total_requests: int = 0
        self.total_detections: int = 0
        self.failed_requests: int = 0
        self.active_requests: int = 0
        self.latencies_history: deque[float] = deque(maxlen=500)

        # Concurrency safety lock
        self.lock = asyncio.Lock()

    def initialize(self) -> None:
        """Load model weights once at application startup."""
        logger.info(f"Initializing Veyraxis InferenceEngine with weights: {self.settings.model_path}...")
        model_file = Path(self.settings.model_path)
        # Fall back to flagship weights if checkpoint path is not present
        if model_file.exists():
            target_path = str(model_file)
        elif Path("weights/yolo11x.pt").exists():
            target_path = "weights/yolo11x.pt"
        else:
            target_path = "yolo11n.pt"

        self.engine = InferenceEngine(
            model_path=target_path,
            device=self.settings.device,
            conf_threshold=self.settings.conf_threshold,
        )
        self.engines[target_path] = self.engine
        self.visualizer = Visualizer()
        logger.info(f"ModelContainer initialization complete. Device: {self.engine.detector.device}")

    def get_engine_for_model(self, model_name: str | None = None) -> InferenceEngine:
        """Retrieve default engine or load and cache a specific requested model weights file."""
        if not model_name or model_name in ("default", self.settings.model_path):
            return self.engine or self.initialize() or self.engine

        if model_name in self.engines:
            return self.engines[model_name]

        logger.info(f"Loading and caching model engine for: {model_name}...")
        new_engine = InferenceEngine(
            model_path=model_name,
            device=self.settings.device,
            conf_threshold=self.settings.conf_threshold,
        )
        self.engines[model_name] = new_engine
        return new_engine

    def record_inference(self, latency_ms: float, detections_count: int) -> None:
        """Update telemetry registers."""
        self.total_requests += 1
        self.total_detections += detections_count
        self.latencies_history.append(latency_ms)

    def record_failure(self) -> None:
        """Record a rejected or erroneous request."""
        self.failed_requests += 1


# Global container instance
container = ModelContainer()


def get_model_container() -> ModelContainer:
    """Dependency provider returning singleton ModelContainer."""
    return container
