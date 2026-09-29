"""
VEY RAXIS SENTINEL — Demo Video & GIF Generator
Generates a polished GitHub showcase video and animated GIF from browser recordings.
Outputs:
  - assets/sentinel_demo.mp4  (H.264 / MP4 720p 24fps)
  - assets/sentinel_demo.gif  (Optimized animated GIF for GitHub README)
  - assets/sentinel_demo.webp (Full-color animated WebP)
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

BRAIN_DIR = Path(r"C:\Users\ayush\.gemini\antigravity-ide\brain\9c6b4c18-7e3b-4ab0-a93b-89082e3224a8")
ASSETS_DIR = Path(r"c:\Projects\veyraxis-sentinel\assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_W = 1280
TARGET_H = 720
FPS = 24


def create_banner_overlay(frame_img: Image.Image, tag: str, message: str) -> Image.Image:
    """Add a sleek Memphis-style bottom callout banner to the frame."""
    img = frame_img.copy()
    draw = ImageDraw.Draw(img)

    banner_h = 62
    banner_y = TARGET_H - banner_h - 16
    banner_x = 24
    banner_w = TARGET_W - 48

    # Banner background (solid dark with hard border)
    draw.rectangle([banner_x, banner_y, banner_x + banner_w, banner_y + banner_h], fill=(24, 24, 41), outline=(255, 113, 206), width=3)
    # Hard shadow
    draw.rectangle([banner_x + 4, banner_y + banner_h, banner_x + banner_w + 4, banner_y + banner_h + 4], fill=(255, 206, 92))

    # Badge tag pill (Neon Pink or Yellow)
    pill_w = len(tag) * 9 + 20
    draw.rectangle([banner_x + 14, banner_y + 14, banner_x + 14 + pill_w, banner_y + 46], fill=(255, 113, 206), outline=(24, 24, 41), width=2)
    draw.text((banner_x + 24, banner_y + 22), tag, fill=(24, 24, 41))

    # Message text
    draw.text((banner_x + 30 + pill_w, banner_y + 22), message, fill=(255, 255, 255))

    return img


def create_title_card(title: str, subtitle: str, badge: str, tags: list[str]) -> Image.Image:
    """Generate an intro or outro card in 1980s Memphis Design style."""
    im = Image.new("RGB", (TARGET_W, TARGET_H), (248, 247, 253))
    draw = ImageDraw.Draw(im)

    # Repeating diagonal accent lines top & bottom
    for i in range(0, TARGET_W + TARGET_H, 28):
        draw.line([(i, 0), (i - 200, 200)], fill=(230, 225, 245), width=2)

    # Memphis floating shapes
    draw.polygon([(80, 80), (140, 40), (160, 120)], fill=(255, 113, 206), outline=(24, 24, 41), width=3)
    draw.rectangle([TARGET_W - 140, 80, TARGET_W - 70, 150], fill=(78, 205, 196), outline=(24, 24, 41), width=3)
    draw.ellipse([80, TARGET_H - 150, 150, TARGET_H - 80], fill=(255, 206, 92), outline=(24, 24, 41), width=3)
    draw.polygon([(TARGET_W - 150, TARGET_H - 140), (TARGET_W - 80, TARGET_H - 70), (TARGET_W - 190, TARGET_H - 60)], fill=(255, 107, 107), outline=(24, 24, 41), width=3)

    # Center card box with heavy shadow
    box_w = 900
    box_h = 380
    bx = (TARGET_W - box_w) // 2
    by = (TARGET_H - box_h) // 2

    # Hard shadow
    draw.rectangle([bx + 10, by + 10, bx + box_w + 10, by + box_h + 10], fill=(24, 24, 41))
    # Card surface
    draw.rectangle([bx, by, bx + box_w, by + box_h], fill=(255, 255, 255), outline=(24, 24, 41), width=4)

    # Top strip inside card
    draw.rectangle([bx, by, bx + box_w, by + 14], fill=(255, 113, 206))

    # Badge
    draw.rectangle([bx + 40, by + 36, bx + 40 + len(badge) * 9 + 24, by + 68], fill=(255, 206, 92), outline=(24, 24, 41), width=2)
    draw.text((bx + 52, by + 44), badge, fill=(24, 24, 41))

    # Title
    draw.text((bx + 40, by + 90), title, fill=(24, 24, 41))
    # Subtitle
    draw.text((bx + 40, by + 145), subtitle, fill=(106, 123, 180))

    # Feature tags
    tx = bx + 40
    ty = by + 220
    colors = [(255, 113, 206), (78, 205, 196), (255, 206, 92), (255, 107, 107)]
    for idx, t in enumerate(tags):
        col = colors[idx % len(colors)]
        t_w = len(t) * 8 + 24
        draw.rectangle([tx, ty, tx + t_w, ty + 36], fill=col, outline=(24, 24, 41), width=2)
        draw.text((tx + 12, ty + 10), t, fill=(24, 24, 41))
        tx += t_w + 12

    return im


def extract_frames_from_webp(webp_path: Path, max_frames: int = 150, step: int = 1) -> list[Image.Image]:
    """Extract and resize frames from an animated WebP file."""
    frames = []
    if not webp_path.exists():
        return frames

    im = Image.open(webp_path)
    count = 0
    for idx, f in enumerate(ImageSequence.Iterator(im)):
        if idx % step != 0:
            continue
        rgb = f.convert("RGB").resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        frames.append(rgb)
        count += 1
        if count >= max_frames:
            break
    return frames


def build_showcase():
    print("Building Veyraxis Sentinel GitHub Demo Video & Animation...")
    all_frames: list[Image.Image] = []

    # 1. Intro Title Card (48 frames = 2.0s)
    intro_card = create_title_card(
        title="VEY RAXIS SENTINEL",
        subtitle="Autonomous Real-Time AI Vision Platform & Detection Studio",
        badge="PRODUCTION RELEASE v2.0",
        tags=["YOLO11 Backbone", "Sub-35ms Latency", "80 Categories", "Interactive Studio"],
    )
    for _ in range(48):
        all_frames.append(intro_card)

    # 2. Scene 1: Real-time Multi-Object Detection from test_detection_fixes
    fixes_webp = BRAIN_DIR / "test_detection_fixes_1790574440325.webp"
    fixes_frames = extract_frames_from_webp(fixes_webp, max_frames=60, step=2)

    for i, f in enumerate(fixes_frames):
        if i < 25:
            annotated = create_banner_overlay(
                f, "STEP 1: LIVE DETECTION", "YOLO11 AI Model Running • 4 Pedestrians Detected • 31.8 ms Latency (31 FPS)"
            )
        elif i < 45:
            annotated = create_banner_overlay(
                f, "STEP 2: PRESET MODES", "Switching to 'Smart Traffic' Preset • Active Class Domain Filter"
            )
        else:
            annotated = create_banner_overlay(
                f, "STEP 3: SMART RECOVERY", "Smart Alert Banner Appears • 1-Click 'Show All Detected Objects'"
            )
        all_frames.append(annotated)

    # 3. Scene 2: Full Taxonomy Restored & Re-inference
    restored_png = BRAIN_DIR / "active_canvas_bounding_boxes_1790575033376.png"
    if restored_png.exists():
        im_restored = Image.open(restored_png).convert("RGB").resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        annotated_restored = create_banner_overlay(
            im_restored, "STEP 4: FULL TAXONOMY", "All 80 COCO Classes Restored • Detection Hierarchy Table & Telemetry Active"
        )
        for _ in range(40):
            all_frames.append(annotated_restored)

    # 4. Scene 3: Urban Traffic & Motorcycles Detection
    traffic_webp = BRAIN_DIR / "inspect_detection_1790570614381.webp"
    traffic_frames = extract_frames_from_webp(traffic_webp, max_frames=50, step=3)
    for i, f in enumerate(traffic_frames):
        annotated = create_banner_overlay(
            f, "STEP 5: DIVERSE SCENES", "Urban Traffic & Motorcycles Scene • Rapid Inference & Vehicle Tracking"
        )
        all_frames.append(annotated)

    # 5. Scene 4: Wildlife Scene Detections
    wildlife_png = BRAIN_DIR / "zebra_scene_loaded_1790568322558.png"
    if not wildlife_png.exists():
        wildlife_png = BRAIN_DIR / "wildlife_scene_loaded_1790568072357.png"

    if wildlife_png.exists():
        im_wildlife = Image.open(wildlife_png).convert("RGB").resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        annotated_wildlife = create_banner_overlay(
            im_wildlife, "STEP 6: FAUNA & WILDLIFE", "Zebra Detected (84.1% Conf) • Real-Time Dynamic Confidence Slider"
        )
        for _ in range(35):
            all_frames.append(annotated_wildlife)

    # 6. Scene 5: Webcam & Ingestion Capabilities
    webcam_png = BRAIN_DIR / "live_webcam_tab_1790571569038.png"
    if webcam_png.exists():
        im_webcam = Image.open(webcam_png).convert("RGB").resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        annotated_webcam = create_banner_overlay(
            im_webcam, "STEP 7: MULTI-STREAM", "Live Webcam Stream HUD • Continuous Detection & Video Frame Ingestion"
        )
        for _ in range(35):
            all_frames.append(annotated_webcam)

    # 7. Outro Card (48 frames = 2.0s)
    outro_card = create_title_card(
        title="DEPLOY VEY RAXIS SENTINEL",
        subtitle="Production Computer Vision Microservice with OpenAPI, Docker & ONNX",
        badge="GET STARTED TODAY",
        tags=["Docker Ready", "ONNX Runtime", "FastAPI Endpoints", "GitHub: veyraxis-sentinel"],
    )
    for _ in range(48):
        all_frames.append(outro_card)

    print(f"Total compiled frames: {len(all_frames)} ({len(all_frames) / FPS:.1f} seconds at {FPS} FPS)")

    # -------------------------------------------------------------
    # Output 1: High-Definition MP4 Video (sentinel_demo.mp4)
    # -------------------------------------------------------------
    mp4_path = ASSETS_DIR / "sentinel_demo.mp4"
    print(f"Exporting MP4 video to: {mp4_path}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(mp4_path), fourcc, FPS, (TARGET_W, TARGET_H))

    for frame in all_frames:
        # Convert RGB PIL to BGR OpenCV
        cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(cv_frame)
    video_writer.release()
    print(f"MP4 export complete! Size: {mp4_path.stat().st_size / (1024 * 1024):.2f} MB")

    # -------------------------------------------------------------
    # Output 2: Optimized Animated GIF for GitHub README (sentinel_demo.gif)
    # -------------------------------------------------------------
    gif_path = ASSETS_DIR / "sentinel_demo.gif"
    print(f"Exporting optimized GIF to: {gif_path}...")

    # Downsample frames for lightweight GitHub README display (800x450, step=2)
    gif_w = 800
    gif_h = 450
    gif_frames = []
    for i, frame in enumerate(all_frames):
        if i % 2 == 0:  # 12 FPS effective for GIF
            downscaled = frame.resize((gif_w, gif_h), Image.Resampling.BILINEAR)
            # Quantize palette with dithering for small file size and crisp colors
            quantized = downscaled.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            gif_frames.append(quantized)

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / 12),
        loop=0,
        optimize=True,
    )
    print(f"GIF export complete! Size: {gif_path.stat().st_size / (1024 * 1024):.2f} MB")

    # -------------------------------------------------------------
    # Output 3: WebP Animated Video (sentinel_demo.webp)
    # -------------------------------------------------------------
    webp_path = ASSETS_DIR / "sentinel_demo.webp"
    print(f"Exporting animated WebP to: {webp_path}...")
    gif_frames[0].convert("RGB").save(
        webp_path,
        save_all=True,
        append_images=[f.convert("RGB") for f in gif_frames[1:]],
        duration=int(1000 / 12),
        loop=0,
        quality=85,
    )
    print(f"WebP export complete! Size: {webp_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    build_showcase()
