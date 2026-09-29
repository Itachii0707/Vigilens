# 👁️ VigiLens

[![CI](https://github.com/vigilens/vigilens/actions/workflows/ci.yml/badge.svg)](https://github.com/vigilens/vigilens/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17+-005CED.svg)](https://onnxruntime.ai/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **VigiLens (*Vigilance* + *Lens*) — A high-performance, scalable, and deployable AI system for real-time industrial object detection, spatial tracking, and visual intelligence.**

> 📖 **Comprehensive System Architecture & Defense Guide**: See [VIGILENS_MASTER_GUIDE.md](file:///c:/Projects/Vigilens/VIGILENS_MASTER_GUIDE.md) for full mathematical formulas (Ray-Casting, 2D vector cross-product tripwire, IoU mAP 50-95, all 80 classes, and system diagrams).

---

## 🎬 Live Interactive Studio Demo

![VigiLens Live Demo](assets/sentinel_demo.gif)

*Real-time object detection across 80 COCO categories, sub-35ms GPU latency, operational industry presets, and interactive bounding-box inspector.*

> 📹 **High-Definition Video**: [Download / Watch MP4 (720p HD)](assets/sentinel_demo.mp4) &nbsp;|&nbsp; 🖼️ **Lightweight WebP**: [sentinel_demo.webp](assets/sentinel_demo.webp)

---

## 1. Project Overview

**VigiLens** (`vigilens`) is an enterprise-grade computer vision platform built from the ground up to solve mission-critical object detection, worker safety (PPE compliance), and asset integrity surveillance in real-time.

Unlike image classification, **VigiLens** performs dense spatial localization and semantic identification, predicting:
1. **Object Class** (e.g. `person`, `vehicle`, `helmet`, `fire`, `damaged_component`)
2. **Bounding-Box Coordinates** (`[x1, y1, x2, y2]` in pixel space and normalized YOLO `[x_c, y_c, w, h]`)
3. **Calibrated Confidence Score** (`0.0` to `1.0`)
4. **Persistent Tracking ID** (via temporal Kalman/IoU & ByteTrack algorithms)

### Core Detection Classes
- `person`: Workers, operators, security personnel, and pedestrians.
- `vehicle`: Industrial forklifts, transport trucks, AGVs, and construction machinery.
- `helmet`: Hard hats, bump caps, and protective headgear.
- `fire`: Open flames, smoke points, and hazardous thermal/combustion anomalies.
- `damaged_component`: Fractured machinery, warped structural beams, or compromised components.

### Ingestion Streams Supported
- **Static Images** (`.jpg`, `.png`, `.webp`, `.bmp`)
- **Image Directories** (batch ingestion and bulk processing)
- **Recorded Video Files** (`.mp4`, `.avi`, `.mkv`, `.mov`)
- **Direct Hardware Webcams** (USB, DirectShow on Windows, V4L2 on Linux)
- **RTSP / IP Camera Streams** (H.264/H.265 RTSP streams with automated exponential backoff reconnection)

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Inputs["1. Ingestion Layer"]
        A1[Single Image] --> Ingest[StreamProcessor]
        A2[Image Directory] --> Ingest
        A3[Video File] --> Ingest
        A4[Webcam] --> Ingest
        A5[RTSP / IP Camera] --> Ingest
    end

    subgraph DataOps["2. DataOps & Validation"]
        D1[Raw Dataset] --> DVal[DatasetValidator]
        DVal --> DRep[Diagnostic Charts & Reports]
        DVal --> Aug[Albumentations Pipeline]
        Aug --> Prep[LetterboxPreprocessor]
    end

    subgraph CoreEngine["3. Veyraxis Core Engine"]
        Prep --> ModelRouter{Runtime Engine}
        ModelRouter -->|GPU CUDA / AMP| PTD[PyTorch Detector YOLO11 / YOLO26]
        ModelRouter -->|CPU / Cross-Platform| ORT[ONNX Runtime Engine]
        ModelRouter -->|NVIDIA TensorRT| TRT[TensorRT Engine]
        
        PTD --> PostProc[NMS & Coordinate Scaler]
        ORT --> PostProc
        TRT --> PostProc
        
        PostProc --> Tracker[ObjectTracker ByteTrack]
    end

    subgraph Serving["4. Serving & Deployment"]
        Tracker --> Vis[Visualizer & Alert Engine]
        Tracker --> API[FastAPI Microservice]
        API --> Health[GET /health]
        API --> Pred[POST /predict]
        API --> Batch[POST /predict/batch]
        API --> Metrics[GET /metrics]
        API --> Docker[Docker & Compose Container]
    end

    subgraph MLOps["5. MLOps & Tracking"]
        PTD --> MLF[MLflow Tracking]
        PTD --> TBoard[TensorBoard]
        PostProc --> Eval[Isolated Test Evaluator]
        Eval --> HTMLRep[Executive HTML Report]
    end
```

---

## 3. Architecture & Design Decisions

### 3.1 Why YOLO (YOLO26 / YOLO11) is Selected
Modern single-stage anchor-free architectures provide an optimal balance between spatial localization accuracy and sub-10ms inference latency. Using modified C3k2 and C2PSA attention blocks, the detector extracts multi-scale feature pyramids (P3, P4, P5) allowing simultaneous detection of tiny PPE helmets alongside large vehicles.

### 3.2 Model Tier Selection Policy
- **Small/Edge Deployment (`yolo26n` / `yolo11n` / `yolo26s`)**: Recommended for edge appliances (NVIDIA Jetson, Intel NUC, or CPU-only servers) where sub-10ms latency is mandatory.
- **Balanced Server Deployment (`yolo26m` / `yolo11m`)**: Recommended for multi-stream RTSP processing (4-8 concurrent 1080p cameras per GPU).
- **High-Accuracy GPU Deployment (`yolo26l` / `yolo26x`)**: Selected for high-resolution forensic inspection, aerial surveillance, or detecting fine-grained micro-fractures in industrial components.

### 3.3 Accuracy, Latency, and Memory Trade-Offs
| Architecture | Tier | Parameters (M) | FLOPs (G) | P50 Latency (GPU) | P50 Latency (CPU ONNX) | Target Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n / YOLO26n** | Edge / Nano | 2.6 M | 6.5 G | **8.7 ms** | **18.0 ms** | Edge cameras, Jetson, CPU |
| **YOLO11s / YOLO26s** | Small | 9.4 M | 21.5 G | 13.2 ms | 36.5 ms | 1080p real-time video |
| **YOLO11m / YOLO26m** | Balanced | 20.1 M | 68.0 G | 22.4 ms | 78.0 ms | Multi-stream RTSP servers |
| **YOLO11x / YOLO26x** | High-Accuracy| 56.9 M | 194.9 G | 42.1 ms | 185.0 ms | Offline forensic audit |

### 3.4 Transfer Learning Strategy
The training pipeline starts with weights pre-trained on large-scale natural distributions (COCO). The backbone preserves general visual primitives (edges, gradients, textures), while the decoupled detection head is fine-tuned to industrial classes. The pipeline supports freezing backbone layers for initial warmup epochs before full model fine-tuning.

### 3.5 Object Tracking (ByteTrack / Kalman)
Video frames and live streams contain temporal correlations. The system includes persistent object tracking using IoU-based Hungarian matching and ByteTrack. Each detected worker, vehicle, or hazard retains a unique `track_id` across frames, mitigating flickering and enabling dwell-time analytics.

### 3.6 Model Quantization & Format Selection
- **Development & Training**: PyTorch with Automatic Mixed Precision (`torch.cuda.amp` FP16) for fast convergence.
- **CPU Deployment**: **ONNX Runtime** utilizing Graph Optimization and multi-threaded SIMD execution.
- **NVIDIA GPU Deployment**: **TensorRT** FP16/INT8 engines providing maximum throughput.

### 3.7 Project Risks & Mitigations
- **Camera Network Dropouts**: Mitigated by `StreamProcessor`'s automatic reconnect loop with exponential backoff.
- **Decompression Bombs & Malicious Uploads**: Mitigated by PIL `MAX_IMAGE_PIXELS` capping, magic-byte inspection, and strict file size limits (15 MB).
- **GPU Out-of-Memory (OOM)**: Mitigated by FastAPI concurrency locks and dynamic batching.

---

## 4. Hardware & Benchmark Results

The system was benchmarked on the following local target hardware:
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM, CUDA 12.4)
- **CPU**: 13th Gen Intel Core i7-13700HX
- **OS**: Windows 11 64-bit / Linux container compatible

### Verified Benchmark Metrics (Live Run):
| Benchmark Metric | Native PyTorch (CUDA:0) | Standalone ONNX Runtime (CPU) |
| :--- | :--- | :--- |
| **Cold-Start Time** | 812.0 ms | **242.1 ms** |
| **P50 Inference Latency** | **8.74 ms** | **18.06 ms** |
| **P95 Inference Latency** | **9.40 ms** | **19.00 ms** |
| **Throughput** | **113.0 FPS** | **55.2 FPS** |
| **Peak GPU VRAM** | **48.7 MB** | 0 MB (Pure CPU) |
| **Model Disk Footprint** | 5.21 MB | 10.17 MB |
| **Parity Validation** | Baseline (Ground Truth) | **PASSED (Mean IoU = 1.0000)** |

---

## 5. Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- Git
- NVIDIA GPU Drivers (550+) with CUDA 12.x (optional for GPU acceleration; CPU is fully supported)

### Step 1: Clone and Set Up Environment
```bash
git clone https://github.com/vigilens/vigilens.git
cd vigilens

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies and package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quickstart: How to Run This Project

### ⚡ One-Click Windows Launchers (Recommended for Demos & Presentations)

No terminal typing required! Simply double-click any of the provided `.bat` launchers in the root folder:

| Launcher File | What It Does |
| :--- | :--- |
| **[`start_webcam.bat`](file:///c:/Projects/Vigilens/start_webcam.bat)** | **Instant 1-Click Run:** Launches live webcam detection + ByteTrack + Hazard Zone Geofencing immediately. |
| **[`run.bat`](file:///c:/Projects/Vigilens/run.bat)** | **Master Interactive Menu:** Double-click and hit `[Enter]` for default webcam or choose from 7 demonstration modes. |
| **[`start_api.bat`](file:///c:/Projects/Vigilens/start_api.bat)** | **REST API Launcher:** Boots the FastAPI microservice and auto-opens Swagger UI (`http://localhost:8000/docs`). |

---

### 💻 Manual Command-Line Execution

#### 1. Run Live Inference with Object Tracking (Webcam)
```bash
python scripts/predict.py --source 0 --display --track
```
*Press `q` anytime to stop the stream and view the count summary.*

#### 2. Run with Virtual Tripwire & Restricted Zone Geofencing
```bash
# Zone Intrusion Warning (alerts when person/vehicle enters hazard perimeter)
python scripts/predict.py --source 0 --display --zone preset

# Virtual Tripwire (counts inbound and outbound crossings)
python scripts/predict.py --source 0 --display --tripwire preset
```

#### 3. Test on Sample Images
```bash
python scripts/predict.py --source data/samples/traffic.jpg --conf 0.25
```
*Visual output saved to `outputs/predictions/pred_traffic.jpg`.*

#### 4. Launch the REST API & Interactive Swagger UI
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to test endpoints interactively.*

---

## 6. Dataset Structure & Validation

### Dataset Directory Layout
```
dataset/
├── images/
│   ├── train/               # Training images (.jpg, .png)
│   ├── val/                 # Validation images
│   └── test/                # Isolated test images
├── labels/
│   ├── train/               # YOLO annotations (.txt)
│   ├── val/
│   └── test/
├── data.yaml                # Dataset mapping configuration
├── dataset_card.md          # Dataset documentation & class specifications
└── README.md
```

### YOLO Annotation Format
Each `.txt` label file contains one row per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
*Coordinates are normalized between 0.0 and 1.0.*

### Dataset Validation Command
Validate images, check for corrupt headers, detect coordinate violations, and generate diagnostic charts:
```bash
python scripts/validate_dataset.py --data dataset/data.yaml --output-dir outputs/dataset_reports
```

**Artifacts Generated**:
- `dataset_summary.json`: High-level data statistics.
- `class_distribution.csv`: Frequency and percentage breakdown per class.
- `class_distribution.png`: Bar chart of class frequencies.
- `bbox_size_distribution.png`: Bounding box area histogram.
- `image_resolutions.png`: Resolution scatter plot.
- `problematic_annotations.md`: Detailed audit table of invalid or flagged lines.
- `annotated_samples/`: Visual image previews with drawn bounding boxes.

---

## 7. Model Training Pipeline

Train a model using reproducible configurations and automated logging:

```bash
# Standard training using configs/train.yaml
python scripts/train.py --config configs/train.yaml

# Training with CLI overrides (e.g. 25 epochs on GPU device 0)
python scripts/train.py --data dataset/data.yaml --epochs 25 --batch-size 8 --device 0
```

### Directory Artifacts Generated in `runs/`:
```
runs/
├── experiments/             # Experiment runs with epoch logs
├── checkpoints/             # Best (best.pt) and last (last.pt) weights
├── logs/                    # Training and validation logs
├── plots/                   # Loss curves, PR curves, F1 curves
├── predictions/             # Batch validation predictions
└── metrics/                 # JSON summary of final metrics
```

---

## 8. Model Evaluation & Error Diagnostics

Evaluate the trained checkpoint against an isolated test partition:

```bash
python scripts/evaluate.py \
  --model runs/checkpoints/best.pt \
  --data dataset/data.yaml \
  --output-dir outputs/evaluation \
  --conf 0.25 \
  --iou 0.50
```

### Generated Reports:
- `evaluation_report.html`: Standalone executive HTML report with summary cards and failure tables.
- `evaluation_results.json`: JSON payload containing P50/P95 latencies, FPS, mAP, and precision/recall.
- `confusion_matrix.png`: Normalized confusion matrix including background false positive analysis.
- `per_class_metrics.csv`: Tabular precision, recall, and AP per class.
- `annotated_predictions/`: Test images with Green (Ground Truth) vs Magenta (Prediction) comparison overlays.

---

## 9. Multi-Source Inference & Object Tracking

Run inference on any source with configurable confidence, NMS, and optional tracking:

### 1. Single Image
```bash
python scripts/predict.py --source dataset/images/test/sample_test_0000.jpg --model runs/checkpoints/best.pt
```

### 2. Directory of Images
```bash
python scripts/predict.py --source dataset/images/test/ --model runs/checkpoints/best.pt --output-dir outputs/predictions
```

### 3. Video File
```bash
python scripts/predict.py --source test_video.mp4 --model runs/checkpoints/best.pt --save-video --track
```

### 4. Local Webcam
```bash
python scripts/predict.py --source 0 --model runs/checkpoints/best.pt --display --track
```

### 5. RTSP / IP Camera Stream
```bash
python scripts/predict.py --source rtsp://admin:pass@192.168.1.100:554/live --track --output-dir outputs/stream_output
```

### 6. Tracking & Object Counting
When `--track` is enabled on videos, webcams, or RTSP streams, the system maintains persistent object IDs and outputs a live count summary:

```text
==================================================
🎯 VIGILENS TRACKING & COUNT SUMMARY:
 • Frames Processed : 450
 • Average Speed    : 34.2 FPS
 • Total Unique IDs : 19
 • Unique Counts by Category:
    - person            : 12
    - car               : 7
==================================================
```
An automated `count_summary.json` is exported to the output directory along with the annotated video stream.

### 7. Virtual Tripwire & Zone Geofencing
Define restricted danger areas or directional crossing lines using computational geometry (Ray-Casting algorithm):

```bash
# Restricted Hazard Zone (flashing warning border + intrusion logging)
python scripts/predict.py --source 0 --display --zone preset

# Virtual Tripwire (bi-directional inbound/outbound counter)
python scripts/predict.py --source 0 --display --tripwire preset
```
When violated, an automated `geofence_events.json` event log is recorded.

### Example Output JSON
```json
{
  "image_id": "frame_000001",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9412,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 420,
        "y2": 700
      },
      "track_id": 1
    }
  ],
  "latency_ms": 8.74
}
```

---

## 10. Model Export & Numerical Parity

Export PyTorch weights to ONNX, TensorRT, or OpenVINO with automated numerical parity verification:

```bash
python scripts/export_model.py \
  --model runs/checkpoints/best.pt \
  --format onnx \
  --imgsz 640 \
  --dynamic \
  --validate \
  --benchmark \
  --output-dir outputs/exports
```

---

## 11. FastAPI REST Microservice

The platform includes a production FastAPI service with input sanitization, MIME verification, file size limits, request ID tracing, and Prometheus-compatible metrics.

### Start the API Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at: `http://localhost:8000/docs`

### API Endpoints
- `GET /`: Auto-redirects to interactive Swagger API documentation (`/docs`).
- `GET /docs`: Interactive OpenAPI / Swagger UI (test endpoints, upload images, inspect JSON schemas).
- `GET /health`: Service health, loaded model name, device, and GPU VRAM usage.
- `POST /predict`: Upload image for real-time detection (returns JSON or annotated image with `?visualize=true`).
- `POST /predict/batch`: Upload multiple images for concurrent batch inference.
- `GET /metrics`: Telemetry counters, average latency, and 95th percentile latency.

### Example cURL Commands

#### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```

#### 2. Single Image Inference (JSON)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@dataset/images/test/sample_test_0000.jpg" \
  -F "conf_threshold=0.3"
```

#### 3. Single Image Inference (Annotated Image Return)
```bash
curl -X POST "http://localhost:8000/predict?visualize=true" \
  -F "file=@dataset/images/test/sample_test_0000.jpg" \
  --output annotated_result.jpg
```

#### 4. Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@dataset/images/test/sample_test_0000.jpg" \
  -F "files=@dataset/images/test/sample_test_0001.jpg"
```

#### 5. Telemetry Metrics
```bash
curl -X GET http://localhost:8000/metrics
```

---

## 12. Docker & Containerized Deployment

Run the complete stack (FastAPI Microservice + MLflow UI) via Docker Compose:

```bash
# Build and start services in background
docker-compose up -d --build

# View logs
docker-compose logs -f veyraxis-api

# Access FastAPI service: http://localhost:8000/docs
# Access MLflow UI:        http://localhost:5000
```

---

## 13. Testing & Code Quality

The test suite covers unit tests, integration tests, security boundaries, and end-to-end pipeline execution:

```bash
# Run full pytest suite (29 tests)
pytest -v

# Run lint checks (Ruff)
ruff check src tests scripts api

# Format code (Ruff)
ruff format src tests scripts api
```

---

## 14. Troubleshooting & Known Limitations

- **RTSP Connection Dropouts**: Ensure the network router supports TCP transport. `StreamProcessor` automatically attempts 5 reconnection cycles with exponential backoff before termination.
- **Decompression Bomb Warnings**: Images larger than 50 megapixels are rejected by `security.py` to prevent denial-of-service memory exhaustion.
- **CUDA Out-of-Memory**: Lower the training batch size in `configs/train.yaml` (e.g. from 16 to 8 or 4) or export to ONNX/TensorRT for inference.

---

## 15. License

Developed under the Apache 2.0 License. Designed and engineered for production computer vision deployments.
