"""Multi-source stream processor for Images, Directories, Video, Webcam, and RTSP streams."""

import json
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from veyraxis.inference.engine import DetectionResult, InferenceEngine
from veyraxis.inference.geofencing import GeofenceManager
from veyraxis.inference.visualizer import Visualizer
from veyraxis.utils.logger import get_logger

logger = get_logger("veyraxis.inference.stream")


class ThreadedCamera:
    """Decoupled background camera capture thread to eliminate USB I/O blocking latency and maximize FPS."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                with self.lock:
                    self.ret = False
                break
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()

    def get(self, prop_id: int) -> float:
        return self.cap.get(prop_id)

    def isOpened(self) -> bool:
        return self.cap.isOpened()


class InputSource(str, Enum):
    IMAGE = "image"
    DIRECTORY = "directory"
    VIDEO = "video"
    WEBCAM = "webcam"
    RTSP = "rtsp"


def determine_source_type(source_str: str) -> InputSource:
    """Classify input string into an InputSource enum."""
    s_lower = source_str.lower().strip()
    if s_lower.startswith("rtsp://") or s_lower.startswith("rtsps://") or s_lower.startswith("http://"):
        return InputSource.RTSP
    if s_lower.isdigit():
        return InputSource.WEBCAM

    p = Path(source_str)
    if p.is_dir():
        return InputSource.DIRECTORY
    if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".flv"}:
        return InputSource.VIDEO
    return InputSource.IMAGE


class StreamProcessor:
    """
    Robust stream ingestion and inference processor.
    Features automated RTSP reconnection with exponential backoff, corrupt frame handling,
    and video / JSON rendering.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        visualizer: Visualizer | None = None,
        geofence_manager: GeofenceManager | None = None,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_sec: float = 2.0,
    ):
        self.engine = engine
        self.visualizer = visualizer or Visualizer()
        self.geofence_manager = geofence_manager
        self.max_reconnect = max_reconnect_attempts
        self.reconnect_backoff = reconnect_backoff_sec
        self.last_count_summary: dict[str, object] = {}
        self.last_geofence_events: list[dict[str, object]] = []

    def process(
        self,
        source: str | int | Path,
        output_dir: Path | None = None,
        save_video: bool = False,
        display: bool = False,
        callback: Callable[[DetectionResult, np.ndarray], None] | None = None,
    ) -> list[DetectionResult]:
        """Process any supported source to completion or termination."""
        source_str = str(source)
        src_type = determine_source_type(source_str)
        logger.info(f"Initiating stream processing: source='{source_str}' (type={src_type})")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if src_type == InputSource.IMAGE:
            return self._process_single_image(Path(source_str), output_dir)
        elif src_type == InputSource.DIRECTORY:
            return self._process_directory(Path(source_str), output_dir)
        else:
            return self._process_video_stream(source_str, src_type, output_dir, save_video, display, callback)

    def _process_single_image(self, img_path: Path, output_dir: Path | None) -> list[DetectionResult]:
        """Inference on a single image file."""
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Cannot read image: {img_path}")
            return []

        result = self.engine.predict_image(img, image_id=img_path.name)
        if output_dir:
            vis = self.visualizer.draw(img, result)
            out_img_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_img_path), vis)
            with open(output_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        return [result]

    def _process_directory(self, dir_path: Path, output_dir: Path | None) -> list[DetectionResult]:
        """Inference on all images in a directory."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in dir_path.iterdir() if p.suffix.lower() in extensions]
        logger.info(f"Found {len(files)} candidate images in {dir_path}")

        results: list[DetectionResult] = []
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning(f"Corrupt or unreadable image skipped: {p}")
                continue
            res = self.engine.predict_image(img, image_id=p.name)
            results.append(res)
            if output_dir:
                vis = self.visualizer.draw(img, res)
                cv2.imwrite(str(output_dir / f"pred_{p.name}"), vis)

        if output_dir:
            summary = [r.model_dump() for r in results]
            with open(output_dir / "directory_detections.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return results

    def _process_video_stream(
        self,
        source_str: str,
        src_type: InputSource,
        output_dir: Path | None,
        save_video: bool,
        display: bool,
        callback: Callable[[DetectionResult, np.ndarray], None] | None,
    ) -> list[DetectionResult]:
        raw_cap = self._open_capture(source_str)
        if not raw_cap or not raw_cap.isOpened():
            raise RuntimeError(f"Unable to open video capture source: {source_str}")

        fps_in = raw_cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_h = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # Wrap live webcam in ThreadedCamera to eliminate USB I/O wait and maximize FPS
        if src_type == InputSource.WEBCAM:
            cap: cv2.VideoCapture | ThreadedCamera = ThreadedCamera(raw_cap)
        else:
            cap = raw_cap

        writer: cv2.VideoWriter | None = None
        if save_video and output_dir:
            out_vid_path = output_dir / "annotated_stream.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps_in, (frame_w, frame_h))
            logger.info(f"Recording output video to: {out_vid_path}")

        results: list[DetectionResult] = []
        frame_idx = 0
        reconnect_attempts = 0
        fps_tracker = 0.0
        unique_ids_by_class: dict[str, set[int]] = {}
        all_unique_ids: set[int] = set()
        t_stream_start = time.perf_counter()

        try:
            while True:
                t_frame_start = time.perf_counter()
                ret, frame = cap.read()

                if not ret or frame is None:
                    if src_type in (InputSource.RTSP, InputSource.WEBCAM):
                        reconnect_attempts += 1
                        logger.warning(
                            f"Stream disconnected. Attempting reconnect {reconnect_attempts}/{self.max_reconnect}..."
                        )
                        cap.release()
                        time.sleep(self.reconnect_backoff * reconnect_attempts)
                        cap = self._open_capture(source_str)
                        if cap and cap.isOpened():
                            logger.info("Stream successfully re-established.")
                            reconnect_attempts = 0
                            continue
                        if reconnect_attempts >= self.max_reconnect:
                            logger.error("Maximum RTSP stream reconnection attempts reached. Terminating.")
                            break
                        continue
                    else:
                        # End of file for video file
                        break

                frame_idx += 1
                frame_id = f"frame_{frame_idx:06d}"

                # Run inference with tracking
                res = self.engine.predict_image(frame, image_id=frame_id)
                results.append(res)

                # Compute active and unique counts
                active_counts: dict[str, int] = {}
                for det in res.detections:
                    cname = det.class_name
                    active_counts[cname] = active_counts.get(cname, 0) + 1
                    if det.track_id is not None:
                        all_unique_ids.add(det.track_id)
                        unique_ids_by_class.setdefault(cname, set()).add(det.track_id)

                if all_unique_ids:
                    live_counts: dict[str, object] = {
                        "total_unique": len(all_unique_ids),
                        "active_total": len(res.detections),
                        "unique_by_class": {c: len(ids) for c, ids in unique_ids_by_class.items()},
                        "active_by_class": active_counts,
                    }
                else:
                    live_counts = active_counts

                # Update Geofence and Virtual Tripwire if enabled
                if self.geofence_manager:
                    geo_res = self.geofence_manager.update(res.detections)
                    if geo_res.get("violations") or geo_res.get("crossings"):
                        self.last_geofence_events.append({"frame_idx": frame_idx, **geo_res})

                # Overlay visualization with live counts and geofence overlays
                vis_frame = self.visualizer.draw(
                    frame,
                    res,
                    fps=fps_tracker,
                    counts=live_counts,
                    geofence_manager=self.geofence_manager,
                )

                if writer:
                    writer.write(vis_frame)

                if callback:
                    callback(res, vis_frame)

                if display:
                    cv2.imshow("VigiLens AI Live Stream", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Stream stopped by operator ('q' pressed).")
                        break

                dt = time.perf_counter() - t_frame_start
                inst_fps = 1.0 / max(dt, 1e-4)
                fps_tracker = 0.85 * fps_tracker + 0.15 * inst_fps if fps_tracker > 0 else inst_fps

        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        elapsed_sec = max(time.perf_counter() - t_stream_start, 1e-4)
        avg_fps = round(frame_idx / elapsed_sec, 1)
        final_summary = {
            "total_frames_processed": frame_idx,
            "average_fps": avg_fps,
            "total_unique_tracks": len(all_unique_ids),
            "unique_counts_by_class": {c: len(ids) for c, ids in unique_ids_by_class.items()},
        }
        self.last_count_summary = final_summary

        if output_dir:
            json_path = output_dir / "stream_detections.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in results], f, indent=2)

            summary_path = output_dir / "count_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(final_summary, f, indent=2)

            if self.last_geofence_events:
                geo_path = output_dir / "geofence_events.json"
                with open(geo_path, "w", encoding="utf-8") as f:
                    json.dump(self.last_geofence_events, f, indent=2, default=str)

        logger.info(
            f"Processed {frame_idx} frames successfully ({avg_fps} FPS). "
            f"Unique objects tracked: {len(all_unique_ids)} ({final_summary['unique_counts_by_class']})."
        )
        return results

    def _open_capture(self, source_str: str) -> cv2.VideoCapture | None:
        """Open VideoCapture with backend optimizations."""
        try:
            if source_str.isdigit():
                # DirectShow backend on Windows is faster and avoids lag
                cap = cv2.VideoCapture(int(source_str), cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            elif source_str.startswith("rtsp://") or source_str.startswith("rtsps://"):
                # FFMPEG with TCP transport to avoid UDP packet loss
                cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            else:
                return cv2.VideoCapture(source_str)
        except Exception as e:
            logger.error(f"Error opening capture for {source_str}: {e}")
            return None
