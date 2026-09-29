"""
Computational geometry and spatial analytics for VigiLens.
Implements Virtual Tripwires and Restricted Zone Geofencing for perimeter security,
safety intrusion monitoring, and directional crossing counting.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from veyraxis.inference.engine import SingleDetection


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """
    Ray-Casting algorithm (even-odd rule) to determine if a 2D point lies within a polygon.
    Returns True if inside, False otherwise.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def ccw(A: tuple[int, int], B: tuple[int, int], C: tuple[int, int]) -> bool:
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    p1: tuple[int, int], p2: tuple[int, int], q1: tuple[int, int], q2: tuple[int, int]
) -> bool:
    """Return True if line segment (p1, p2) intersects segment (q1, q2)."""
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


@dataclass
class RestrictedZone:
    """
    Polygon geofence monitoring unauthorized entry into hazardous or restricted areas.
    """

    zone_id: str
    polygon: list[tuple[int, int]]  # List of (x, y) vertices
    target_classes: list[str] = field(default_factory=lambda: ["person", "vehicle", "forklift"])
    color_normal: tuple[int, int, int] = (0, 165, 255)  # Amber
    color_violation: tuple[int, int, int] = (0, 0, 240)  # Flashing Red
    current_violators: list[int] = field(default_factory=list)  # Active track IDs inside
    total_violation_events: int = 0
    is_violated: bool = False

    def check_detection(self, det: SingleDetection) -> bool:
        """
        Check if detection's bottom-center contact point falls inside the polygon.
        Ground contact point is the standard in surveillance (feet of person / base of vehicle).
        """
        if self.target_classes and det.class_name.lower() not in self.target_classes:
            return False

        # Compute bottom-center contact point
        contact_x = (det.bbox.x1 + det.bbox.x2) // 2
        contact_y = det.bbox.y2
        return point_in_polygon((contact_x, contact_y), self.polygon)


@dataclass
class VirtualTripwire:
    """
    Directional boundary line detecting and counting crossings (Inbound vs Outbound).
    """

    wire_id: str
    pt1: tuple[int, int]
    pt2: tuple[int, int]
    target_classes: list[str] = field(default_factory=lambda: ["person", "vehicle", "car", "motorcycle"])
    inbound_count: int = 0
    outbound_count: int = 0
    recent_crossings: list[dict[str, Any]] = field(default_factory=list)

    def compute_crossing(
        self, prev_pt: tuple[int, int], curr_pt: tuple[int, int], track_id: int, class_name: str
    ) -> str | None:
        """
        Determine if trajectory (prev_pt -> curr_pt) intersected the tripwire.
        Returns 'INBOUND', 'OUTBOUND', or None.
        """
        if not segments_intersect(prev_pt, curr_pt, self.pt1, self.pt2):
            return None

        # Cross product of tripwire vector and motion vector determines crossing direction
        wx = self.pt2[0] - self.pt1[0]
        wy = self.pt2[1] - self.pt1[1]
        mx = curr_pt[0] - prev_pt[0]
        my = curr_pt[1] - prev_pt[1]

        cross = wx * my - wy * mx
        direction = "INBOUND" if cross > 0 else "OUTBOUND"

        if direction == "INBOUND":
            self.inbound_count += 1
        else:
            self.outbound_count += 1

        self.recent_crossings.append(
            {
                "track_id": track_id,
                "class_name": class_name,
                "direction": direction,
            }
        )
        return direction


class GeofenceManager:
    """
    Coordinates active zones and tripwires, evaluates spatial violations,
    and draws high-contrast visual overlays.
    """

    def __init__(
        self,
        zones: list[RestrictedZone] | None = None,
        tripwires: list[VirtualTripwire] | None = None,
    ):
        self.zones: list[RestrictedZone] = zones or []
        self.tripwires: list[VirtualTripwire] = tripwires or []
        self.previous_positions: dict[int, tuple[int, int]] = {}
        self.active_alert_message: str | None = None

    def add_zone(self, zone: RestrictedZone) -> None:
        self.zones.append(zone)

    def add_tripwire(self, tripwire: VirtualTripwire) -> None:
        self.tripwires.append(tripwire)

    def update(self, detections: list[SingleDetection]) -> dict[str, Any]:
        """
        Evaluate frame detections against all zones and tripwires.
        Returns event summary dict.
        """
        violations: list[dict[str, Any]] = []
        crossings: list[dict[str, Any]] = []

        # 1. Update Restricted Zones
        for zone in self.zones:
            zone.current_violators.clear()
            for det in detections:
                if zone.check_detection(det):
                    tid = det.track_id if det.track_id is not None else 0
                    zone.current_violators.append(tid)
                    violations.append(
                        {
                            "zone_id": zone.zone_id,
                            "track_id": tid,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                        }
                    )

            if zone.current_violators:
                zone.is_violated = True
                zone.total_violation_events += len(zone.current_violators)
            else:
                zone.is_violated = False

        # 2. Update Virtual Tripwires
        for det in detections:
            if det.track_id is not None:
                tid = det.track_id
                # Centroid of object
                curr_pt = ((det.bbox.x1 + det.bbox.x2) // 2, (det.bbox.y1 + det.bbox.y2) // 2)
                prev_pt = self.previous_positions.get(tid)

                if prev_pt is not None and prev_pt != curr_pt:
                    for wire in self.tripwires:
                        if wire.target_classes and det.class_name.lower() not in wire.target_classes:
                            continue
                        direction = wire.compute_crossing(prev_pt, curr_pt, tid, det.class_name)
                        if direction:
                            crossings.append(
                                {
                                    "wire_id": wire.wire_id,
                                    "track_id": tid,
                                    "class_name": det.class_name,
                                    "direction": direction,
                                }
                            )

                self.previous_positions[tid] = curr_pt

        # Build alert message
        if violations:
            v_zones = ", ".join(sorted({v["zone_id"] for v in violations}))
            self.active_alert_message = f"INTRUSION DETECTED IN: {v_zones}!"
        else:
            self.active_alert_message = None

        return {
            "violations": violations,
            "crossings": crossings,
            "active_alert": self.active_alert_message,
        }

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render geofence polygon zones, tripwires, and crossing HUD onto frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw Restricted Zones with translucent overlay
        overlay = vis.copy()
        for zone in self.zones:
            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            fill_color = zone.color_violation if zone.is_violated else zone.color_normal

            # Fill polygon semi-transparently
            cv2.fillPoly(overlay, [pts], fill_color)

            # Draw polygon perimeter line
            border_thickness = 3 if zone.is_violated else 2
            cv2.polylines(vis, [pts], isClosed=True, color=fill_color, thickness=border_thickness)

            # Zone Label Tag
            label_pt = zone.polygon[0]
            status_txt = f"{zone.zone_id}: VIOLATION!" if zone.is_violated else f"{zone.zone_id} [CLEAR]"
            cv2.putText(
                vis,
                status_txt,
                (label_pt[0] + 5, label_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if zone.is_violated else (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Blend semi-transparent overlay (alpha=0.25)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # Draw Virtual Tripwires
        for wire in self.tripwires:
            wire_color = (255, 200, 0)  # Cyan/Yellow
            cv2.line(vis, wire.pt1, wire.pt2, wire_color, 2, cv2.LINE_AA)

            # Draw line endpoints
            cv2.circle(vis, wire.pt1, 5, (0, 255, 255), -1)
            cv2.circle(vis, wire.pt2, 5, (0, 255, 255), -1)

            # Wire stats label
            mid_x = (wire.pt1[0] + wire.pt2[0]) // 2
            mid_y = (wire.pt1[1] + wire.pt2[1]) // 2
            stats_txt = f"{wire.wire_id} | IN: {wire.inbound_count} | OUT: {wire.outbound_count}"
            (tw, th), _ = cv2.getTextSize(stats_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (mid_x - 5, mid_y - th - 5), (mid_x + tw + 5, mid_y + 5), (20, 20, 20), -1)
            cv2.putText(vis, stats_txt, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw flashing Intrusion Alert Banner if active
        if self.active_alert_message:
            cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 220), -1)
            cv2.putText(
                vis,
                f"🚨 {self.active_alert_message}",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return vis
