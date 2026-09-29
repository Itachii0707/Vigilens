"""Model registry and detector wrappers for Veyraxis Sentinel."""

from veyraxis.models.detector import VeyraxisDetector
from veyraxis.models.registry import ModelRegistry, resolve_model_weights

__all__ = ["ModelRegistry", "VeyraxisDetector", "resolve_model_weights"]
