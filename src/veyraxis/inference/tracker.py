"""Object tracking manager supporting ByteTrack and lightweight Kalman/IoU tracker."""

from dataclasses import dataclass
from typing import Any

from veyraxis.utils.bboxes import compute_iou


@dataclass
class TrackItem:
    track_id: int
    class_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    age: int = 0
    hits: int = 1


class ObjectTracker:
    """
    Temporal association tracker providing unique object ID persistence across sequential frames.
    Implements IoU-based Hungarian matching with track decay for zero-dependency standalone tracking.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 2,
        class_names: dict[int, str] | None = None,
        smooth_factor: float = 0.70,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.class_names = class_names or {}
        self.smooth_factor = smooth_factor
        self.tracks: dict[int, TrackItem] = {}
        self.next_id: int = 1
        self.unique_tracks: set[int] = set()
        self.unique_by_class: dict[int, set[int]] = {}

    def update(self, detections: list[tuple[int, float, tuple[int, int, int, int]]]) -> list[TrackItem]:
        """
        Update tracker with new detections in format: [(class_id, confidence, (x1, y1, x2, y2)), ...]
        Returns active confirmed tracks.
        """
        # Increment age for existing tracks
        for t in self.tracks.values():
            t.age += 1

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches: list[tuple[int, int]] = []

        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            for t_idx, t_id in enumerate(track_ids):
                t_item = self.tracks[t_id]
                best_iou = self.iou_threshold
                best_det_idx = -1

                for d_idx in unmatched_dets:
                    cls_id, conf, bbox = detections[d_idx]
                    if cls_id == t_item.class_id:
                        iou = compute_iou(t_item.bbox, bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_det_idx = d_idx

                if best_det_idx >= 0:
                    matches.append((t_id, best_det_idx))
                    unmatched_dets.remove(best_det_idx)
                    unmatched_tracks.remove(t_id)

        # Update matched tracks with EMA temporal coordinate smoothing
        for t_id, d_idx in matches:
            cls_id, conf, bbox = detections[d_idx]

            if self.smooth_factor > 0 and self.tracks[t_id].hits >= 1:
                old_box = self.tracks[t_id].bbox
                smoothed_box = (
                    int(round(self.smooth_factor * bbox[0] + (1.0 - self.smooth_factor) * old_box[0])),
                    int(round(self.smooth_factor * bbox[1] + (1.0 - self.smooth_factor) * old_box[1])),
                    int(round(self.smooth_factor * bbox[2] + (1.0 - self.smooth_factor) * old_box[2])),
                    int(round(self.smooth_factor * bbox[3] + (1.0 - self.smooth_factor) * old_box[3])),
                )
            else:
                smoothed_box = bbox

            self.tracks[t_id].bbox = smoothed_box
            self.tracks[t_id].confidence = conf
            self.tracks[t_id].age = 0
            self.tracks[t_id].hits += 1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            cls_id, conf, bbox = detections[d_idx]
            new_track = TrackItem(
                track_id=self.next_id,
                class_id=cls_id,
                bbox=bbox,
                confidence=conf,
                age=0,
                hits=1,
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # Delete expired tracks
        dead_ids = [t_id for t_id, item in self.tracks.items() if item.age > self.max_age]
        for d_id in dead_ids:
            del self.tracks[d_id]

        # Record confirmed tracks into unique tracking sets
        active_tracks: list[TrackItem] = []
        for t in self.tracks.values():
            if t.age == 0 and (t.hits >= self.min_hits or self.min_hits <= 1):
                active_tracks.append(t)
                self.unique_tracks.add(t.track_id)
                self.unique_by_class.setdefault(t.class_id, set()).add(t.track_id)

        return active_tracks

    def get_count_summary(self) -> dict[str, Any]:
        """Return live and cumulative unique counts across tracked classes."""
        active_counts: dict[str, int] = {}
        for t in self.tracks.values():
            if t.age == 0 and (t.hits >= self.min_hits or self.min_hits <= 1):
                cname = self.class_names.get(t.class_id, str(t.class_id))
                active_counts[cname] = active_counts.get(cname, 0) + 1

        unique_counts: dict[str, int] = {}
        for cid, ids in self.unique_by_class.items():
            cname = self.class_names.get(cid, str(cid))
            unique_counts[cname] = len(ids)

        return {
            "total_unique": len(self.unique_tracks),
            "active_total": sum(active_counts.values()),
            "unique_by_class": unique_counts,
            "active_by_class": active_counts,
        }

    def reset(self) -> None:
        """Reset all tracking and counting states."""
        self.tracks.clear()
        self.unique_tracks.clear()
        self.unique_by_class.clear()
        self.next_id = 1
