"""Training callbacks bridging Ultralytics events to MLflow and custom loggers."""

from typing import Any

from veyraxis.tracking.experiment import ExperimentTracker
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.training.callbacks")


def setup_ultralytics_callbacks(yolo_model: Any, tracker: ExperimentTracker) -> None:
    """Register custom hook callbacks onto Ultralytics YOLO trainer."""

    def on_train_epoch_end(trainer: Any) -> None:
        """Log training loss and epoch progress."""
        try:
            epoch = getattr(trainer, "epoch", 0) + 1
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None:
                metrics = {}
                if isinstance(loss_items, dict):
                    for k, v in loss_items.items():
                        try:
                            metrics[f"train/{k}"] = float(v)
                        except (ValueError, TypeError):
                            pass
                elif hasattr(loss_items, "__iter__"):
                    loss_names = getattr(trainer, "loss_names", ["box_loss", "cls_loss", "dfl_loss"])
                    for idx, v in enumerate(loss_items):
                        name = loss_names[idx] if idx < len(loss_names) else f"loss_{idx}"
                        try:
                            metrics[f"train/{name}"] = float(v)
                        except (ValueError, TypeError):
                            pass
                if metrics:
                    tracker.log_metrics(metrics, step=epoch)
        except Exception as e:
            logger.debug(f"Error in on_train_epoch_end callback: {e}")

    def on_val_end(validator: Any) -> None:
        """Log validation mAP metrics."""
        epoch = getattr(validator, "epoch", 0) + 1
        metrics_dict = getattr(validator, "metrics", None)
        if metrics_dict and hasattr(metrics_dict, "results_dict"):
            res = metrics_dict.results_dict
            clean_metrics = {f"val/{k}": float(v) for k, v in res.items() if isinstance(v, (int, float))}
            tracker.log_metrics(clean_metrics, step=epoch)

    try:
        yolo_model.add_callback("on_train_epoch_end", on_train_epoch_end)
        yolo_model.add_callback("on_val_end", on_val_end)
        logger.info("Custom MLflow tracking callbacks registered with Ultralytics trainer.")
    except Exception as e:
        logger.warning(f"Could not attach custom callbacks to YOLO model: {e}")
