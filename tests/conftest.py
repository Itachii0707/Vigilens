"""Shared pytest fixtures for Veyraxis Sentinel."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.dependencies import container
from api.main import app
from veyraxis.data.dataset import create_synthetic_sentinel_dataset


@pytest.fixture(scope="session")
def sample_image() -> np.ndarray:
    """Create a sample 640x640 synthetic BGR image with geometric shapes."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(img, (400, 400), 80, (0, 0, 255), -1)  # Red circle
    return img


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image: np.ndarray) -> bytes:
    """Encode sample image as JPEG byte buffer."""
    ret, buf = cv2.imencode(".jpg", sample_image)
    assert ret
    return buf.tobytes()


@pytest.fixture(scope="session")
def temp_dataset_dir() -> Generator[Path, None, None]:
    """Temporary synthetic dataset directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="sentinel_test_data_"))
    create_synthetic_sentinel_dataset(
        output_root=temp_dir,
        samples_per_split={"train": 4, "val": 2, "test": 2},
    )
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient with initialized ModelContainer."""
    # Ensure model container is initialized
    if container.engine is None:
        container.initialize()
    with TestClient(app) as test_client:
        yield test_client
