"""Experiment tracking manager for MLflow and TensorBoard."""

from pathlib import Path
from typing import Any

import mlflow

from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.tracking.experiment")


class ExperimentTracker:
    """
    MLflow experiment tracking manager.
    Encapsulates run lifecycle, hyperparameters, metrics, and artifact uploads.
    """

    def __init__(
        self,
        experiment_name: str = "veyraxis-sentinel",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        self.active_run = None

        if self.enabled:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"MLflow initialized: experiment='{self.experiment_name}', uri='{self.tracking_uri}'")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}. Tracking disabled.")
                self.enabled = False

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> Any | None:
        """Start a new tracked MLflow run."""
        if not self.enabled:
            return None
        try:
            self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
            logger.info(f"Started MLflow run: ID={self.active_run.info.run_id}")
            return self.active_run
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {e}")
            return None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameter dictionary."""
        if not self.enabled:
            return
        try:
            # Flatten or stringify complex objects
            clean_params = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()}
            mlflow.log_params(clean_params)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics."""
        if not self.enabled:
            return
        try:
            import re

            clean_metrics = {}
            for k, v in metrics.items():
                clean_k = re.sub(r"[^\w\.\-\s/]", "_", k)
                clean_metrics[clean_k] = float(v)
            mlflow.log_metrics(clean_metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, local_path: Path, artifact_path: str | None = None) -> None:
        """Upload a file or directory artifact."""
        if not self.enabled or not Path(local_path).exists():
            return
        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path} to MLflow: {e}")

    def end_run(self) -> None:
        """End the currently active run."""
        if self.enabled and self.active_run:
            try:
                mlflow.end_run()
                self.active_run = None
                logger.info("MLflow run completed.")
            except Exception as e:
                logger.warning(f"Error ending MLflow run: {e}")
