# 👁️ VigiLens: Comprehensive System Architecture, Algorithms & Operation Guide

> **VigiLens** (*Vigilance + Lens*) — Autonomous Vision Engine for Real-Time Spatial Object Detection, ByteTrack Kinematic Tracking, and Mathematical Perimeter Geofencing.

---

## 1. Executive Summary & Academic Scope

### 1.1 Project Overview
**VigiLens** is a production-grade, edge-deployable computer vision platform engineered for real-time visual surveillance, automated safety monitoring, and spatial hazard prevention. It integrates state-of-the-art single-stage anchor-free convolutional detectors (**YOLO11x / YOLO11m**), temporal multi-object tracking, and deterministic computational geometry algorithms.

### 1.2 Academic & Dissertation Relevance
Many computer vision academic projects stop at running pre-packaged object detection scripts. **VigiLens** elevates the academic rigor by introducing:
1. **Mathematical Spatial Geofencing**: Deterministic Jordan Curve ray-casting and 2D vector cross-product tripwire crossing analysis computed in real time.
2. **Temporal Coordinate Smoothing**: Exponential Moving Average (EMA) bounding box stabilization filtering spatial jitter.
3. **Decoupled Asynchronous Frame Capture**: Multi-threaded non-blocking camera reading eliminating USB I/O bottlenecks.
4. **Quantified Multi-Tier Benchmarking**: Rigorous latency, throughput, memory footprint, and mAP50–95 validation against official COCO test distributions.
5. **Production Microservice Architecture**: Complete REST API implementation conforming to OpenAPI 3.1 / Swagger standards.

---

## 2. Mathematical Foundations & Core Algorithms

```
                          ┌─────────────────────────┐
                          │   Live Camera / RTSP    │
                          └───────────┬─────────────┘
                                      │ Threaded USB Fetch
                                      ▼
                          ┌─────────────────────────┐
                          │     ThreadedCamera      │
                          │   (Zero I/O Blocking)   │
                          └───────────┬─────────────┘
                                      │ Latest Frame
                                      ▼
                          ┌─────────────────────────┐
                          │  YOLO11x Flagship CNN   │
                          │   (FP16 CUDA:0 Tensor)  │
                          └───────────┬─────────────┘
                                      │ Detections (x1, y1, x2, y2, conf, cls)
                                      ▼
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │   ByteTrack Tracker   │           │   Geofence Manager    │
        │ • Hungarian Matching  │           │ • Ray-Casting Point   │
        │ • EMA Box Smoothing   │           │ • Vector Cross-Prod   │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │ Tracked IDs                       │ Incursions & Crossings
                    └─────────────────┬─────────────────┘
                                      ▼
                          ┌─────────────────────────┐
                          │  Anti-Aliased Renderer  │
                          │  (FPS HUD, Pills, Zones)│
                          └─────────────────────────┘
```

### 2.1 Restricted Zone Geofencing (Jordan Curve Theorem & Ray-Casting)
To determine if a detected object (e.g., worker or vehicle) has breached a restricted safety zone defined by an arbitrary polygon $P = \{v_1, v_2, \dots, v_n\}$, VigiLens implements the **Ray-Casting Algorithm**:

* **Mathematical Concept**: A horizontal test ray is projected from the target object's reference coordinate $P(x, y)$ towards positive infinity $(+\infty, y)$.
* **Incursion Criterion**: The number of intersections $k$ between the ray and the polygon's edges is calculated:
  $$\text{Inside}(P) = \begin{cases} \text{True} & \text{if } k \equiv 1 \pmod 2 \\ \text{False} & \text{if } k \equiv 0 \pmod 2 \end{cases}$$
* **Edge Intersection Equation**: An edge connecting $v_i(x_i, y_i)$ and $v_j(x_j, y_j)$ intersects the horizontal ray at $y$ if:
  $$\min(y_i, y_j) < y \le \max(y_i, y_j) \quad \text{and} \quad x_{\text{intersect}} = x_i + \frac{y - y_i}{y_j - y_i}(x_j - x_i) > x$$

### 2.2 Directional Virtual Tripwires (2D Vector Cross Product)
To count objects entering or exiting across a virtual line segment $\overline{AB}$ between $A(x_1, y_1)$ and $B(x_2, y_2)$:
* Let previous object centroid be $P_{t-1}$ and current centroid be $P_t$.
* The line segments $\overline{AB}$ and $\overline{P_{t-1}P_t}$ intersect if and only if points $A$ and $B$ lie on opposite sides of $\overline{P_{t-1}P_t}$, and points $P_{t-1}$ and $P_t$ lie on opposite sides of $\overline{AB}$.
* **Orientation Test**: Computed using the 2D cross-product determinant:
  $$\text{Orient}(A, B, P) = (B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)$$
* The directional sign of $\text{Orient}(A, B, P_t)$ determines whether the passage was **Inbound** ($+$) or **Outbound** ($-$).

### 2.3 Temporal Exponential Moving Average (EMA) Bounding Box Smoothing
Standard CNN detectors predict slightly perturbed bounding box coordinates across consecutive frames due to quantization noise. VigiLens stabilizes bounding boxes using an exponential filter:
$$\hat{B}_t = \alpha B_t + (1 - \alpha) \hat{B}_{t-1}$$
Where:
* $B_t = (x_1, y_1, x_2, y_2)$ is the raw measurement at frame $t$.
* $\hat{B}_{t-1}$ is the filtered box from the preceding frame.
* $\alpha = 0.70$ is the calibrated smoothing coefficient, eliminating coordinate vibration while remaining responsive to rapid human movement.

---

## 3. Identifiable Object Classes (All 80 MS COCO Categories)

The underlying YOLO11 architecture is trained on the Microsoft COCO benchmark, covering **80 standard object categories**:

| ID | Class Name | Category | ID | Class Name | Category |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **0** | `person` | People & Wearables | **40** | `wine glass` | Food & Kitchen |
| **1** | `bicycle` | Vehicles | **41** | `cup` | Food & Kitchen |
| **2** | `car` | Vehicles | **42** | `fork` | Food & Kitchen |
| **3** | `motorcycle` | Vehicles | **43** | `knife` | Food & Kitchen |
| **4** | `airplane` | Vehicles | **44** | `spoon` | Food & Kitchen |
| **5** | `bus` | Vehicles | **45** | `bowl` | Food & Kitchen |
| **6** | `train` | Vehicles | **46** | `banana` | Food & Kitchen |
| **7** | `truck` | Vehicles | **47** | `apple` | Food & Kitchen |
| **8** | `boat` | Vehicles | **48** | `sandwich` | Food & Kitchen |
| **9** | `traffic light` | Traffic & Urban | **49** | `orange` | Food & Kitchen |
| **10** | `fire hydrant` | Traffic & Urban | **50** | `broccoli` | Food & Kitchen |
| **11** | `stop sign` | Traffic & Urban | **51** | `carrot` | Food & Kitchen |
| **12** | `parking meter` | Traffic & Urban | **52** | `hot dog` | Food & Kitchen |
| **13** | `bench` | Traffic & Urban | **53** | `pizza` | Food & Kitchen |
| **14** | `bird` | Animals | **54** | `donut` | Food & Kitchen |
| **15** | `cat` | Animals | **55** | `cake` | Food & Kitchen |
| **16** | `dog` | Animals | **56** | `chair` | Furniture & Indoors |
| **17** | `horse` | Animals | **57** | `couch` | Furniture & Indoors |
| **18** | `sheep` | Animals | **58** | `potted plant`| Furniture & Indoors |
| **19** | `cow` | Animals | **59** | `bed` | Furniture & Indoors |
| **20** | `elephant` | Animals | **60** | `dining table`| Furniture & Indoors |
| **21** | `bear` | Animals | **61** | `toilet` | Furniture & Indoors |
| **22** | `zebra` | Animals | **62** | `tv` | Electronics |
| **23** | `giraffe` | Animals | **63** | `laptop` | Electronics |
| **24** | `backpack` | People & Wearables | **64** | `mouse` | Electronics |
| **25** | `umbrella` | People & Wearables | **65** | `remote` | Electronics |
| **26** | `handbag` | People & Wearables | **66** | `keyboard` | Electronics |
| **27** | `tie` | People & Wearables | **67** | `cell phone` | Electronics |
| **28** | `suitcase` | People & Wearables | **68** | `microwave` | Electronics |
| **29** | `frisbee` | Sports & Outdoor | **69** | `oven` | Electronics |
| **30** | `skis` | Sports & Outdoor | **70** | `toaster` | Electronics |
| **31** | `snowboard` | Sports & Outdoor | **71** | `sink` | Furniture & Indoors |
| **32** | `sports ball`| Sports & Outdoor | **72** | `refrigerator`| Electronics |
| **33** | `kite` | Sports & Outdoor | **73** | `book` | Household & Personal |
| **34** | `baseball bat`| Sports & Outdoor | **74** | `clock` | Household & Personal |
| **35** | `baseball glove`| Sports & Outdoor | **75** | `vase` | Household & Personal |
| **36** | `skateboard`| Sports & Outdoor | **76** | `scissors` | Household & Personal |
| **37** | `surfboard` | Sports & Outdoor | **77** | `teddy bear` | Household & Personal |
| **38** | `tennis racket`| Sports & Outdoor | **78** | `hair drier` | Household & Personal |
| **39** | `bottle` | Food & Kitchen | **79** | `toothbrush` | Household & Personal |

> **Note on Out-of-Vocabulary Objects**: Objects outside these 80 categories (e.g., specific hand tools, pens, watches) cannot be identified without custom domain transfer fine-tuning.

---

## 4. What is "mAP 50–95 (54.7%)"? Full Academic Explanation

### 4.1 Definition
**mAP** stands for **mean Average Precision**. In object detection, each prediction is evaluated on two criteria:
1. **Classification Accuracy**: Did the model predict the correct category?
2. **Localization Accuracy (IoU)**: Does the bounding box cover the true object accurately?

### 4.2 Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$
* **IoU 0.50**: 50% overlap. Considered an acceptable coarse match.
* **IoU 0.75**: Strict match. The predicted box tightly wraps the object.
* **IoU 0.95**: 95% overlap. Near-perfect boundary alignment.

### 4.3 Why Average from 50 to 95?
Older competitions (like PASCAL VOC) only measured $\text{mAP}_{50}$. However, a box with only 50% overlap can be sloppy or misaligned. The modern MS COCO standard computes average precision across **10 separate IoU thresholds** in steps of 0.05:
$$\text{mAP@[0.50:0.95]} = \frac{1}{10} \sum_{k=0}^{9} \text{AP}_{\text{IoU} = 0.50 + 0.05k}$$

### 4.4 Why 54.7% is World-Class
In standard classification, 54% might seem low. In object detection over 80 varied classes on complex real-world imagery, **54.7% represents state-of-the-art flagship performance**:
* Historical baseline (Faster R-CNN): ~37.4%
* YOLOv5-X (2020): ~50.7%
* YOLOv8-X (2023): ~53.9%
* **YOLO11x (VigiLens Flagship): `54.7%`** (and over **`73%`** at loose $\text{mAP}_{50}$)

---

## 5. Model Architecture Benchmark Matrix

Empirical benchmarks recorded directly on target local hardware:
* **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, CUDA 12.4)
* **CPU**: 13th Gen Intel Core i7-13700HX

| Model File | Architecture Tier | Parameters | Model Size | COCO mAP (50–95) | P50 Latency (GPU) | Live Stream FPS | Primary Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`weights/yolo11x.pt`** 🏆 | **Extra Large** | **56.9 M** | **109.3 MB** | **`54.7%`** | **36.6 ms** | **~28 FPS** | **Default: Highest accuracy & detects most objects** |
| **`weights/yolo11m.pt`** ⚡ | **Medium** | **20.1 M** | **38.8 MB** | **`51.5%`** | **14.2 ms** | **~71 FPS** | **High speed / multi-camera RTSP** |
| **`weights/yolo11n.pt`** | **Nano** | **2.6 M** | **5.3 MB** | **`39.5%`** | **8.7 ms** | **~99 FPS** | Low-power CPU / Raspberry Pi |
| **`runs/checkpoints/best.onnx`** | **ONNX Engine** | **2.6 M** | **10.2 MB** | Baseline | **18.1 ms** (CPU) | **~55 FPS** (CPU) | Standalone CPU deployment |

---

## 6. One-Click Quickstart Guide (Windows)

VigiLens includes double-clickable launchers in the root folder that require zero terminal setup:

### 1. `start_webcam.bat` (Direct 1-Click Runner)
* **How to run**: Double-click `start_webcam.bat` in File Explorer.
* **What happens**:
  * An interactive prompt displays with default `[1] YOLO11-Extra Large (54.7% mAP)`.
  * Press **`[Enter]`** to launch immediately.
  * Opens your live webcam with **Hazard Zone Geofencing**, **ByteTrack Tracking**, **Calibrated Confidence HUD**, and **Threaded Non-blocking Video Capture**.
  * If no physical webcam is plugged in, it gracefully falls back to an urban traffic scene demonstration.

### 2. `run.bat` (Master Demonstration Menu)
* **How to run**: Double-click `run.bat` in File Explorer.
* **Available Modes**:
  1. Live Webcam + Restricted Hazard Zone Geofencing (DEFAULT)
  2. Live Webcam + Virtual Tripwire Directional Counter
  3. Live Webcam + Object Tracking & Accuracy Percentages
  4. Test Inference on Sample Images (Traffic / Pedestrians)
  5. Start FastAPI REST Microservice (Opens Swagger UI)
  6. Run Complete Verification Test Suite (36 Tests)
  7. Export PyTorch Model to ONNX Runtime Engine
  8. Switch Model Tier (Nano vs Medium vs Extra Large)

### 3. `start_api.bat` (REST API & Web UI)
* **How to run**: Double-click `start_api.bat` in File Explorer.
* **What happens**:
  * Boots the FastAPI backend on `http://localhost:8000`.
  * Automatically opens your default web browser to the interactive **Swagger UI** (`http://localhost:8000/docs`).

---

## 7. Command-Line Reference

```bash
# 1. Live Webcam with Maximum Accuracy (YOLO11x):
python scripts/predict.py --source 0 --display --track --model weights/yolo11x.pt

# 2. Live Webcam with Hazard Zone Geofencing:
python scripts/predict.py --source 0 --display --zone preset

# 3. Live Webcam with Virtual Tripwire Counter:
python scripts/predict.py --source 0 --display --tripwire preset

# 4. Custom Geofence Polygon Coordinates (x1,y1,x2,y2,x3,y3,x4,y4):
python scripts/predict.py --source 0 --display --zone DANGER_AREA:100,100,500,100,500,400,100,400

# 5. Run Detection on an Image:
python scripts/predict.py --source data/samples/traffic.jpg --conf 0.25

# 6. Run Complete Automated Test Suite:
pytest -v
```

---

## 8. REST API Endpoints

When the FastAPI server is running on `http://localhost:8000`:

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Redirects to Swagger UI (`/docs`). |
| `GET` | `/api/v1/health` | Health check, device placement, and uptime. |
| `GET` | `/api/v1/metrics` | Request counts, latency distributions, and system telemetry. |
| `POST` | `/api/v1/predict` | Multipart image upload returning JSON detections. |
| `POST` | `/api/v1/predict/visualize` | Multipart image upload returning annotated JPEG stream. |
| `POST` | `/api/v1/predict/batch` | Batch image upload (up to 16 images) returning JSON array. |
| `GET` | `/api/v1/models` | Lists available model profiles (Nano, Medium, Extra Large). |

---

## 9. Verification & Code Quality

* **Unit & Integration Tests**: **36 / 36 tests passing** (`pytest -v`).
* **Code Formatting & Linting**: **0 errors** (`ruff check .` clean).
* **Memory Management**: Zero unbounded queues; camera buffer set to 1; intermediate training checkpoints pruned.
