"""Unit tests for StreamProcessor source parsing and ObjectTracker ID persistence."""

import numpy as np

from veyraxis.inference.stream import InputSource, determine_source_type
from veyraxis.inference.tracker import ObjectTracker
from veyraxis.inference.visualizer import Visualizer


def test_determine_source_type():
    """Verify proper classification of input source strings."""
    assert determine_source_type("rtsp://192.168.1.100:554/live") == InputSource.RTSP
    assert determine_source_type("0") == InputSource.WEBCAM
    assert determine_source_type("video.mp4") == InputSource.VIDEO
    assert determine_source_type("video.avi") == InputSource.VIDEO
    assert determine_source_type("photo.jpg") == InputSource.IMAGE


def test_object_tracker_id_persistence():
    """Verify tracker associates sequential detections and maintains track_id."""
    tracker = ObjectTracker(iou_threshold=0.3, min_hits=1)

    # Frame 1: Detection at (100, 100, 200, 200)
    dets_f1 = [(0, 0.9, (100, 100, 200, 200))]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id

    # Frame 2: Slightly moved object at (105, 102, 205, 202) -> Should match same track_id
    dets_f2 = [(0, 0.92, (105, 102, 205, 202))]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t1_id


def test_visualizer_rendering(sample_image: np.ndarray):
    """Verify visualizer renders bounding boxes and text onto frame array."""
    vis = Visualizer(alert_classes=["fire"])
    from veyraxis.inference.engine import BoundingBox, DetectionResult, SingleDetection

    res = DetectionResult(
        image_id="test_frame",
        detections=[
            SingleDetection(
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x1=50, y1=50, x2=200, y2=300),
                track_id=1,
            ),
            SingleDetection(
                class_id=3,
                class_name="fire",
                confidence=0.88,
                bbox=BoundingBox(x1=300, y1=300, x2=450, y2=450),
            ),
        ],
    )

    drawn = vis.draw(sample_image, res, fps=30.0, counts={"person": 1, "fire": 1})
    assert drawn.shape == sample_image.shape
    # Ensure drawing modified pixels
    assert not np.array_equal(drawn, sample_image)


def test_object_tracker_count_summary():
    """Verify ObjectTracker accurately tallies unique and active object counts."""
    tracker = ObjectTracker(iou_threshold=0.3, min_hits=1, class_names={0: "person", 1: "bicycle"})

    # Frame 1: Person 1 and Bicycle 1
    tracker.update([(0, 0.9, (10, 10, 50, 50)), (1, 0.85, (100, 100, 150, 150))])
    summary_f1 = tracker.get_count_summary()
    assert summary_f1["total_unique"] == 2
    assert summary_f1["active_total"] == 2
    assert summary_f1["unique_by_class"]["person"] == 1
    assert summary_f1["unique_by_class"]["bicycle"] == 1

    # Frame 2: Person 1 moves slightly, Bicycle 1 leaves, Person 2 appears
    tracker.update([(0, 0.92, (15, 12, 55, 52)), (0, 0.88, (200, 200, 250, 250))])
    summary_f2 = tracker.get_count_summary()
    # Unique total: Person 1, Bicycle 1, Person 2 = 3 unique objects
    assert summary_f2["total_unique"] == 3
    assert summary_f2["active_total"] == 2
    assert summary_f2["unique_by_class"]["person"] == 2
    assert summary_f2["unique_by_class"]["bicycle"] == 1

    # Reset
    tracker.reset()
    assert tracker.get_count_summary()["total_unique"] == 0
