"""Real-time inference, video/RTSP streams, tracking, and visualization for Veyraxis Sentinel."""

from veyraxis.inference.engine import BoundingBox, DetectionResult, InferenceEngine, SingleDetection
from veyraxis.inference.geofencing import GeofenceManager, RestrictedZone, VirtualTripwire
from veyraxis.inference.stream import InputSource, StreamProcessor
from veyraxis.inference.tracker import ObjectTracker
from veyraxis.inference.visualizer import Visualizer

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "GeofenceManager",
    "InferenceEngine",
    "InputSource",
    "ObjectTracker",
    "RestrictedZone",
    "SingleDetection",
    "StreamProcessor",
    "VirtualTripwire",
    "Visualizer",
]
