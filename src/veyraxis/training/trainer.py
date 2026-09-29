"""Reproducible model training pipeline for Veyraxis Sentinel."""

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from veyraxis.data.validator import DatasetValidator
from veyraxis.models.registry import resolve_model_weights
from veyraxis.tracking.experiment import ExperimentTracker
from veyraxis.training.callbacks import setup_ultralytics_callbacks
from veyraxis.utils.config import SentinelConfig
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.training.trainer")


class VeyraxisTrainer:
    """
    Production-grade training orchestrator.
    Handles data validation, runs hierarchy creation, transfer learning,
    mixed precision, and experiment tracking.
    """

    def __init__(
        self,
        config: SentinelConfig,
        data_yaml: Path,
        project_dir: Path | None = None,
        experiment_name: str | None = None,
    ):
        self.config = config
        self.data_yaml = Path(data_yaml)
        self.project_dir = Path(project_dir or Path.cwd() / "runs")
        self.experiment_name = experiment_name or f"exp_{config.model.name.replace('.pt', '')}"

        # Initialize directory layout
        self.exp_dir = self.project_dir / "experiments" / self.experiment_name
        self.checkpoints_dir = self.project_dir / "checkpoints"
        self.logs_dir = self.project_dir / "logs"
        self.plots_dir = self.project_dir / "plots"
        self.predictions_dir = self.project_dir / "predictions"
        self.metrics_dir = self.project_dir / "metrics"

        for d in (
            self.exp_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.plots_dir,
            self.predictions_dir,
            self.metrics_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        try:
            from ultralytics import settings as yolo_settings

            yolo_settings.update({"mlflow": False})
        except Exception:
            pass

        self.tracker = ExperimentTracker(
            experiment_name=self.config.tracking.experiment_name,
            tracking_uri=self.config.tracking.tracking_uri,
            enabled=self.config.tracking.enabled,
        )

    def validate_data_before_training(self) -> bool:
        """Ensure dataset integrity before consuming compute resources."""
        logger.info(f"Validating dataset from: {self.data_yaml}")
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            data_dict = yaml.safe_load(f)

        root = Path(data_dict.get("path", self.data_yaml.parent))
        classes = (
            list(data_dict.get("names", {}).values())
            if isinstance(data_dict.get("names"), dict)
            else data_dict.get("names", [])
        )

        validator = DatasetValidator(dataset_root=root, classes=classes)
        report = validator.validate(output_dir=self.logs_dir / "pre_training_validation")

        if not report.is_valid:
            logger.error("Dataset validation failed prior to training! Review logs/pre_training_validation")
            return False

        logger.info(
            f"Dataset validated successfully: {report.total_images} images, {report.total_annotations} annotations."
        )
        return True

    def train(self, resume: bool = False) -> dict[str, Any]:
        """Execute full training lifecycle."""
        # 1. Pre-flight dataset verification
        if not self.validate_data_before_training():
            raise ValueError("Aborting training due to fatal dataset integrity issues.")

        # 2. Resolve model architecture weights
        resolved_weights = resolve_model_weights(self.config.model.name)
        logger.info(f"Initiating training with base model: {resolved_weights}")

        device = self.config.model.device
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"

        # 3. Start MLflow tracking run
        self.tracker.start_run(
            run_name=self.experiment_name,
            tags={"base_model": resolved_weights, "device": str(device)},
        )
        self.tracker.log_params(self.config.model.model_dump())
        self.tracker.log_params(self.config.training.model_dump())

        # 4. Load model
        model = YOLO(resolved_weights)
        setup_ultralytics_callbacks(model, self.tracker)

        # 5. Run Ultralytics train
        try:
            model.train(
                data=str(self.data_yaml.resolve()),
                epochs=self.config.training.epochs,
                batch=self.config.training.batch_size,
                imgsz=self.config.model.imgsz,
                device=device,
                workers=self.config.training.workers,
                optimizer=self.config.training.optimizer,
                lr0=self.config.training.lr0,
                lrf=self.config.training.lrf,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
                warmup_epochs=self.config.training.warmup_epochs,
                patience=self.config.training.patience,
                save_period=self.config.training.save_period,
                seed=self.config.training.seed,
                project=str(self.project_dir / "experiments"),
                name=self.experiment_name,
                exist_ok=True,
                pretrained=self.config.training.pretrained,
                amp=self.config.model.half and (device != "cpu"),
                resume=resume,
                verbose=True,
            )

            # 6. Post-training checkpoint and artifact organization
            ultralytics_save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else self.exp_dir
            best_weight = ultralytics_save_dir / "weights" / "best.pt"
            last_weight = ultralytics_save_dir / "weights" / "last.pt"

            if best_weight.exists():
                shutil.copy2(best_weight, self.checkpoints_dir / "best.pt")
                logger.info(f"Best checkpoint synchronized to: {self.checkpoints_dir / 'best.pt'}")
            if last_weight.exists():
                shutil.copy2(last_weight, self.checkpoints_dir / "last.pt")

            # Copy generated plots to central plots dir
            for plot_file in ultralytics_save_dir.glob("*.png"):
                shutil.copy2(plot_file, self.plots_dir / plot_file.name)

            # Save summary metrics
            metrics_summary = {}
            if hasattr(model.trainer, "metrics"):
                metrics_summary = {k: float(v) for k, v in model.trainer.metrics.items() if isinstance(v, (int, float))}
            with open(self.metrics_dir / "training_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=2)

            self.tracker.log_metrics(metrics_summary)
            if (self.checkpoints_dir / "best.pt").exists():
                self.tracker.log_artifact(self.checkpoints_dir / "best.pt")

            logger.info("Training cycle completed successfully.")
            return metrics_summary

        finally:
            self.tracker.end_run()
