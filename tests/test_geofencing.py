"""Unit tests for Geofencing, Virtual Tripwire, and Ray-Casting spatial algorithms."""

import numpy as np

from veyraxis.inference.engine import BoundingBox, SingleDetection
from veyraxis.inference.geofencing import (
    GeofenceManager,
    RestrictedZone,
    VirtualTripwire,
    point_in_polygon,
    segments_intersect,
)


def test_ray_casting_point_in_polygon():
    """Verify Ray-Casting algorithm accurately classifies points inside and outside polygon."""
    # Simple 100x100 square
    square = [(10, 10), (110, 10), (110, 110), (10, 110)]

    # Point clearly inside
    assert point_in_polygon((50, 50), square) is True

    # Point clearly outside
    assert point_in_polygon((5, 50), square) is False
    assert point_in_polygon((150, 50), square) is False
    assert point_in_polygon((50, 150), square) is False


def test_segments_intersect():
    """Verify line segment intersection detection."""
    # Crossing segments: (0, 5) -> (10, 5) intersects (5, 0) -> (5, 10)
    assert segments_intersect((0, 5), (10, 5), (5, 0), (5, 10)) is True

    # Parallel non-intersecting segments
    assert segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False


def test_restricted_zone_detection():
    """Verify RestrictedZone triggers violation on bottom-center contact point."""
    zone = RestrictedZone(
        zone_id="TEST_HAZARD",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        target_classes=["person"],
    )

    # Detection 1: Person with feet at (200, 250) -> Inside zone
    det_inside = SingleDetection(
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=BoundingBox(x1=180, y1=150, x2=220, y2=250),
        track_id=1,
    )
    assert zone.check_detection(det_inside) is True

    # Detection 2: Person with feet at (50, 80) -> Outside zone
    det_outside = SingleDetection(
        class_id=0,
        class_name="person",
        confidence=0.88,
        bbox=BoundingBox(x1=30, y1=20, x2=70, y2=80),
        track_id=2,
    )
    assert zone.check_detection(det_outside) is False


def test_virtual_tripwire_crossing():
    """Verify VirtualTripwire counts directional crossings."""
    wire = VirtualTripwire(
        wire_id="GATE_1",
        pt1=(0, 100),
        pt2=(200, 100),
        target_classes=["person"],
    )

    # Movement from (50, 80) to (50, 120) crosses downward (INBOUND)
    direction = wire.compute_crossing((50, 80), (50, 120), track_id=1, class_name="person")
    assert direction in ("INBOUND", "OUTBOUND")
    assert wire.inbound_count + wire.outbound_count == 1


def test_geofence_manager_workflow(sample_image: np.ndarray):
    """Verify GeofenceManager coordinates zones, detects intrusions, and renders visual overlay."""
    zone = RestrictedZone(
        zone_id="ZONE_A",
        polygon=[(50, 50), (250, 50), (250, 250), (50, 250)],
        target_classes=["person"],
    )
    wire = VirtualTripwire(wire_id="WIRE_1", pt1=(10, 150), pt2=(300, 150))
    mgr = GeofenceManager(zones=[zone], tripwires=[wire])

    # Det inside zone
    det = SingleDetection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=200),
        track_id=1,
    )

    events = mgr.update([det])
    assert len(events["violations"]) == 1
    assert zone.is_violated is True
    assert mgr.active_alert_message is not None

    drawn = mgr.draw(sample_image)
    assert drawn.shape == sample_image.shape
    assert not np.array_equal(drawn, sample_image)
