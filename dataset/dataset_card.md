# Dataset Card: Veyraxis Sentinel Industrial Object Detection

## 1. Overview
The Veyraxis Sentinel dataset is curated for real-time monitoring, workplace safety, and asset integrity surveillance in industrial, warehouse, and construction settings.

## 2. Target Classes
The dataset includes 5 core detection classes:
- **`0: person`**: Human workers, operators, and pedestrians.
- **`1: vehicle`**: Forklifts, trucks, automated guided vehicles (AGVs), and transport carts.
- **`2: helmet`**: Hard hats, protective helmets, and PPE headwear.
- **`3: fire`**: Open flames, hazardous combustions, and ignition events.
- **`4: damaged_component`**: Fractured machinery, warped structural beams, or compromised components.

## 3. Annotation Schema
Labels follow standard normalized YOLO bounding box syntax:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized within the range `[0.0, 1.0]`.

## 4. Split Distribution
- **Training**: 70% of frames (used exclusively for parameter updates and feature learning).
- **Validation**: 15% of frames (used for learning-rate scheduling and early-stopping checkpointing).
- **Test**: 15% of frames (strictly isolated for final mAP, latency, and error mode assessment).

## 5. Provenance and Data Governance
- Images are filtered to avoid duplicate frames or sequential redundancy.
- Cryptographic hash checks (`SHA-256`) are enforced to guarantee zero data leakage between splits.
