"""Integration tests for FastAPI inference endpoints, security, and metrics."""

from fastapi.testclient import TestClient


def test_api_health(client: TestClient):
    """GET /health should return 200 and system device telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "device" in data
    assert "uptime_seconds" in data
    assert "classes" in data
    assert len(data["classes"]) > 0


def test_api_predict_single_image(client: TestClient, sample_image_bytes: bytes):
    """POST /predict should accept an image and return structured JSON detections."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert "request_id" in data
    assert "image_id" in data
    assert "detections" in data
    assert "inference_latency_ms" in data
    assert response.headers.get("X-Request-ID") == data["request_id"]


def test_api_predict_visualize(client: TestClient, sample_image_bytes: bytes):
    """POST /predict?visualize=true should return an annotated JPEG image."""
    response = client.post(
        "/predict?visualize=true",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_api_predict_batch(client: TestClient, sample_image_bytes: bytes):
    """POST /predict/batch should process multiple files and return batch detections."""
    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("frame_01.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("frame_02.jpg", sample_image_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 2
    assert len(data["results"]) == 2


def test_api_predict_invalid_mime(client: TestClient):
    """POST /predict with non-image data should be rejected with 415 Unsupported Media Type."""
    fake_payload = b"<?php echo 'malicious_code'; ?>"
    response = client.post(
        "/predict",
        files={"file": ("script.php", fake_payload, "text/plain")},
    )
    assert response.status_code == 415
    data = response.json()
    assert "error" in data or "detail" in data


def test_api_predict_empty_file(client: TestClient):
    """POST /predict with empty payload should be rejected with 413 or 400."""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code in (400, 413, 415)


def test_api_metrics(client: TestClient):
    """GET /metrics should expose inference counters and latency percentiles."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_detections_made" in data
    assert "average_latency_ms" in data
    assert "p95_latency_ms" in data


def test_api_root_redirect(client: TestClient):
    """GET / should redirect to /docs Swagger UI."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"
