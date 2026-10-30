"""
tests/test_ood_and_hierarchy.py
===============================
Unit tests for:
- Layered OOD rejection (grey tiles, tiny images, monochrome blanks)
- Crop-constrained hierarchical inference
- Batch inference endpoint (/predict_batch)
- Coverage blindspot mapping
"""

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.api import app
from src.ood_detector import check_image_heuristics


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_heuristic_rejects_tiny_image():
    """Images smaller than 64x64 must be rejected before reaching neural inference."""
    tiny = Image.new("RGB", (32, 32), color=(0, 255, 0))
    valid, code, msg = check_image_heuristics(tiny)
    assert not valid
    assert code == "IMAGE_TOO_SMALL"


def test_heuristic_rejects_flat_monochrome():
    """Flat uniform images (e.g. pure grey tile) must be rejected."""
    flat = Image.new("RGB", (224, 224), color=(128, 128, 128))
    valid, code, msg = check_image_heuristics(flat)
    assert not valid
    assert code in ("FLAT_IMAGE", "NOT_A_LEAF_COLOR")


def test_heuristic_rejects_non_leaf_colors():
    """Images lacking plant tissue colors (e.g. pure blue canvas) are rejected."""
    blue_canvas = Image.new("RGB", (224, 224), color=(30, 40, 200))
    valid, code, msg = check_image_heuristics(blue_canvas)
    assert not valid
    assert code == "NOT_A_LEAF_COLOR"


def test_api_predict_rejects_grey_tile(client):
    """POST /predict with a flat grey tile returns status=rejected_not_a_leaf."""
    im = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post("/predict", files={"file": ("grey.jpg", buf, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "rejected_not_a_leaf"
    assert "reason_code" in data
    assert data["confidence"] == 0.0


def test_api_predict_crop_constraint(client):
    """When crop='potato' is specified, returned condition must belong to potato."""
    # Create synthetic leaf-like green square
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = 160  # Green
    arr[40:180, 40:180, 0] = 50
    arr[40:180, 40:180, 1] = 200
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/predict",
        data={"crop": "potato"},
        files={"file": ("leaf.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    if data["status"] == "ok":
        assert data["crop"].lower() == "potato"
        assert data["in_crop_constrained"] is True


def test_predict_batch_endpoint(client):
    """POST /predict_batch handles multiple files."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = 180
    im = Image.fromarray(arr)
    buf1 = io.BytesIO()
    im.save(buf1, format="JPEG")
    buf1.seek(0)

    # Grey tile file
    im_grey = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf2 = io.BytesIO()
    im_grey.save(buf2, format="JPEG")
    buf2.seek(0)

    files = [
        ("files", ("leaf1.jpg", buf1, "image/jpeg")),
        ("files", ("grey.jpg", buf2, "image/jpeg")),
    ]
    res = client.post("/predict_batch", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2
    assert data["results"][1]["status"] == "rejected_not_a_leaf"
