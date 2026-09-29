"""Inference CLI script for images, directories, video files, webcam, and RTSP streams."""

import argparse
import sys
from pathlib import Path

from veyraxis.inference.engine import InferenceEngine
from veyraxis.inference.stream import StreamProcessor
from veyraxis.inference.visualizer import Visualizer
from veyraxis.utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Veyraxis Sentinel object detection on various input sources.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image path, directory, video file, webcam index ('0'), or RTSP URL",
    )
    default_model = "weights/yolo11x.pt" if Path("weights/yolo11x.pt").exists() else "yolo11n.pt"
    parser.add_argument("--model", type=str, default=default_model, help="Path to model weights or architecture name")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--track", action="store_true", help="Enable temporal object tracking with ID persistence")
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Enable restricted zone: 'preset' (center hazard area) or 'ID:x1,y1,x2,y2,x3,y3,x4,y4'",
    )
    parser.add_argument(
        "--tripwire",
        type=str,
        default=None,
        help="Enable virtual tripwire: 'preset' (center divider) or 'ID:x1,y1,x2,y2'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save visual overlays and JSON detections",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Record annotated video output for video and RTSP streams"
    )
    parser.add_argument("--display", action="store_true", help="Display live rendering window")
    args = parser.parse_args()

    geofence_mgr = None
    if args.zone or args.tripwire:
        from veyraxis.inference.geofencing import GeofenceManager, RestrictedZone, VirtualTripwire

        geofence_mgr = GeofenceManager()
        if args.zone:
            if args.zone in ("preset", "center", "hazard"):
                geofence_mgr.add_zone(
                    RestrictedZone(
                        zone_id="HAZARD_ZONE_A",
                        polygon=[(150, 150), (490, 150), (490, 420), (150, 420)],
                        target_classes=["person", "vehicle", "forklift", "motorcycle", "car"],
                    )
                )
            else:
                parts = args.zone.split(":")
                zid = parts[0] if len(parts) > 1 else "RESTRICTED_ZONE"
                coords = [int(c) for c in (parts[1] if len(parts) > 1 else parts[0]).split(",")]
                poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                geofence_mgr.add_zone(RestrictedZone(zone_id=zid, polygon=poly))

        if args.tripwire:
            if args.tripwire in ("preset", "center", "gate"):
                geofence_mgr.add_tripwire(
                    VirtualTripwire(
                        wire_id="ENTRY_GATE",
                        pt1=(50, 240),
                        pt2=(590, 240),
                    )
                )
            else:
                parts = args.tripwire.split(":")
                wid = parts[0] if len(parts) > 1 else "TRIPWIRE_1"
                coords = [int(c) for c in (parts[1] if len(parts) > 1 else parts[0]).split(",")]
                geofence_mgr.add_tripwire(
                    VirtualTripwire(
                        wire_id=wid,
                        pt1=(coords[0], coords[1]),
                        pt2=(coords[2], coords[3]),
                    )
                )

        args.track = True

    engine = InferenceEngine(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        enable_tracking=args.track,
    )
    visualizer = Visualizer()
    processor = StreamProcessor(engine=engine, visualizer=visualizer, geofence_manager=geofence_mgr)

    logger.info(f"Running inference with model '{args.model}' on source '{args.source}' (track={args.track})...")
    try:
        results = processor.process(
            source=args.source,
            output_dir=Path(args.output_dir),
            save_video=args.save_video,
            display=args.display,
        )

        total_dets = sum(len(r.detections) for r in results)
        logger.info(f"Inference complete: {len(results)} frame(s) processed, {total_dets} detections found.")

        # Print Object Count & Tracking Summary
        if processor.last_count_summary:
            summary = processor.last_count_summary
            print("\n" + "=" * 50)
            print("🎯 VIGILENS TRACKING & COUNT SUMMARY:")
            print(f" • Frames Processed : {summary.get('total_frames_processed', 0)}")
            print(f" • Average Speed    : {summary.get('average_fps', 0.0)} FPS")
            print(f" • Total Unique IDs : {summary.get('total_unique_tracks', 0)}")
            by_cls = summary.get("unique_counts_by_class", {})
            if isinstance(by_cls, dict) and by_cls:
                print(" • Unique Counts by Category:")
                for cname, count in sorted(by_cls.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {cname:18s}: {count}")
            print("=" * 50)
        elif results:
            from collections import Counter
            all_classes = [d.class_name for r in results for d in r.detections]
            counts = Counter(all_classes)
            print("\n" + "=" * 50)
            print("🎯 VIGILENS DETECTION SUMMARY:")
            print(f" • Total Detections : {len(all_classes)}")
            for cname, count in counts.most_common():
                print(f"    - {cname:18s}: {count}")
            print("=" * 50)

        # Print Geofence Violations & Tripwire Crossings
        if processor.last_geofence_events:
            print("\n" + "=" * 50)
            print("🚨 VIGILENS GEOFENCE VIOLATION & CROSSING LOG:")
            for ev in processor.last_geofence_events[:6]:
                fidx = ev.get("frame_idx")
                for viol in ev.get("violations", []):
                    conf_pct = int(viol["confidence"] * 100)
                    print(
                        f" • [Frame {fidx}] ⚠️ Zone {viol['zone_id']}: Track #{viol['track_id']} ({viol['class_name']} {conf_pct}%) INTRUSION!"
                    )
                for cross in ev.get("crossings", []):
                    print(
                        f" • [Frame {fidx}] ➡️ Wire {cross['wire_id']}: Track #{cross['track_id']} ({cross['class_name']}) crossed {cross['direction']}"
                    )
            print("=" * 50)

        # Print structured JSON of first detection frame
        if results:
            sample_json = results[0].model_dump_json(indent=2)
            print("\nSample Frame Detection JSON:")
            print(sample_json)

    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
