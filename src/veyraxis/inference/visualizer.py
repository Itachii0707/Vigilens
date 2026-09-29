"""High-performance frame visualizer and detection overlay renderer."""

import cv2
import numpy as np

from veyraxis.inference.engine import DetectionResult, SingleDetection


class Visualizer:
    """
    Renders bounding boxes, confidence tags, tracking IDs, FPS overlays, and alert banners.
    """

    COLOR_PALETTE: list[tuple[int, int, int]] = [
        (46, 204, 113),  # Person: Emerald Green
        (52, 152, 219),  # Vehicle: Dodger Blue
        (241, 196, 15),  # Helmet: Sunflower Yellow
        (231, 76, 60),  # Fire: Alizarin Red
        (155, 89, 182),  # Damaged Component: Amethyst Purple
        (230, 126, 34),  # Orange
        (26, 188, 156),  # Turquoise
        (52, 73, 94),  # Midnight Blue
    ]

    def __init__(self, alert_classes: list[str] | None = None):
        self.alert_classes = set(alert_classes or ["fire", "damaged_component"])

    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult | list[SingleDetection],
        fps: float | None = None,
        draw_alerts: bool = True,
        counts: dict[str, object] | None = None,
        geofence_manager: object | None = None,
    ) -> np.ndarray:
        """Draw detections onto a copy of the frame."""
        vis = image.copy()
        detections = result.detections if isinstance(result, DetectionResult) else result

        # First draw Geofencing zones and tripwires under bounding boxes
        if geofence_manager is not None and hasattr(geofence_manager, "draw"):
            vis = geofence_manager.draw(vis)

        has_alert = False
        alert_messages: list[str] = []

        for det in detections:
            bbox = det.bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            color = self.COLOR_PALETTE[det.class_id % len(self.COLOR_PALETTE)]

            # Check if this class is a designated security/safety alert
            is_alert = det.class_name.lower() in self.alert_classes
            if is_alert:
                has_alert = True
                alert_messages.append(f"{det.class_name.upper()} DETECTED ({det.confidence:.2f})")
                color = (0, 0, 255)  # Flash red for alerts

            # Draw bounding box
            thickness = 3 if is_alert else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Build label text with crisp confidence percentage (e.g. PERSON 94%)
            conf_pct = int(round(det.confidence * 100))
            tag = f"{det.class_name.upper()} {conf_pct}%"
            if det.track_id is not None:
                tag = f"#{det.track_id} {tag}"

            # Calculate label background pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(tag, font, font_scale, font_thickness)

            tag_y1 = max(y1 - text_h - 8, 0)
            tag_y2 = tag_y1 + text_h + 8
            tag_x2 = min(x1 + text_w + 10, vis.shape[1])

            cv2.rectangle(vis, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                vis,
                tag,
                (x1 + 4, tag_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        # Draw FPS overlay in top-left
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.rectangle(vis, (10, 10), (130, 42), (20, 20, 20), -1)
            cv2.putText(
                vis,
                fps_text,
                (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 128),
                2,
                cv2.LINE_AA,
            )

        # Draw Live Object Count HUD
        if counts:
            hud_lines: list[str] = []
            if "total_unique" in counts and "unique_by_class" in counts:
                tot_u = counts.get("total_unique", 0)
                act_tot = counts.get("active_total", 0)
                hud_lines.append(f"TRACK: {act_tot} active ({tot_u} unique total)")
                by_cls = counts.get("unique_by_class", {})
                act_by_cls = counts.get("active_by_class", {})
                if isinstance(by_cls, dict):
                    details = [
                        f"{c}: {act_by_cls.get(c, 0)} ({u} tot)"
                        for c, u in list(by_cls.items())[:3]
                    ]
                    if details:
                        hud_lines.append(" | ".join(details))
            else:
                items = [f"{k}: {v}" for k, v in list(counts.items())[:4]]
                if items:
                    hud_lines.append("COUNT: " + " | ".join(items))

            if hud_lines:
                y_offset = 55 if fps is not None else 20
                for line_idx, line_txt in enumerate(hud_lines):
                    y_pos = y_offset + (line_idx * 24)
                    (tw, th), _ = cv2.getTextSize(line_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                    cv2.rectangle(vis, (10, y_pos - th - 4), (10 + tw + 12, y_pos + 6), (25, 25, 25), -1)
                    cv2.putText(
                        vis,
                        line_txt,
                        (16, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # Draw Safety Alert Banner across top of frame if critical hazards detected
        if draw_alerts and has_alert:
            banner_text = " | ".join(alert_messages[:3])
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 200), -1)
            cv2.putText(
                vis,
                f"ALERT: {banner_text}",
                (140, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return vis
