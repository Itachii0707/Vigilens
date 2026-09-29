"""Training pipelines and callbacks for Veyraxis Sentinel."""

from veyraxis.training.callbacks import setup_ultralytics_callbacks
from veyraxis.training.trainer import VeyraxisTrainer

__all__ = ["VeyraxisTrainer", "setup_ultralytics_callbacks"]
