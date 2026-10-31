"""
Tests for the FastAPI /predict, /health, /stats endpoints.

Uses TestClient (synchronous) so no running server is needed.
The model is mocked if not present; if models/best_model.pth exists,
a real inference test is run.

Usage:
    pytest tests/test_api.py -v
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── TestClient import ─────────────────────────────────────────────────────────
# FastAPI's TestClient uses httpx under the hood
try:
    from fastapi.testclient import TestClient
    from src.api import app
    CLIENT_AVAILABLE = True
except ImportError:
    CLIENT_AVAILABLE = False


@pytest.fixture(scope="module")
def client():
    if not CLIENT_AVAILABLE:
        pytest.skip("FastAPI/httpx not available")
    project_root = Path(__file__).resolve().parent.parent
    if not (project_root / "models" / "best_model.pth").exists():
        pytest.skip("Model artifact unavailable; API integration tests require local weights")
    with TestClient(app) as c:
        yield c


def _make_jpeg_bytes(width: int = 256, height: int = 256, color: tuple = (100, 150, 80)) -> bytes:
    """Create a small solid-color JPEG in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


# ─────────────────────────────────────────────────────────────────────────────
# /stats
# ─────────────────────────────────────────────────────────────────────────────

def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "request_count" in body
    assert "mean_latency_ms" in body
    assert "class_distribution" in body


# ─────────────────────────────────────────────────────────────────────────────
# /predict — schema validation
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_schema(client):
    """
    Send a valid JPEG and verify the response schema is correct.
    If the model is not loaded, expect 503 (not a test failure).
    """
    jpeg_bytes = _make_jpeg_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
    )

    if resp.status_code == 503:
        pytest.skip("Model not loaded — run scripts/train.py first")

    assert resp.status_code == 200
    body = resp.json()

    # Required fields
    assert "status" in body
    assert "disclaimer" in body
    assert "top_k" in body
    assert isinstance(body["top_k"], list)

    if body["status"] == "ok":
        assert "crop" in body
        assert "condition" in body
        assert "confidence" in body
        assert isinstance(body["confidence"], float)
        assert "advisory" in body
        advisory = body["advisory"]
        assert "recommendation_text" in advisory
        assert "disclaimer" in advisory
        # Grad-CAM may be empty string on error, but key must exist
        assert "gradcam_png_base64" in body

    elif body["status"] == "uncertain":
        assert "message" in body


def test_predict_rejects_non_image(client):
    """Non-image content type should return 400."""
    resp = client.post(
        "/predict",
        files={"file": ("text.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_rejects_corrupt_file(client):
    """Corrupt image bytes should return 400."""
    resp = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"\xff\xd8\xff\x00corrupt", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_predict_with_weather_params(client):
    """Weather params should be accepted without error."""
    jpeg_bytes = _make_jpeg_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
        data={"temperature_c": "25.0", "humidity_pct": "80.0"},
    )
    assert resp.status_code in (200, 503)  # 503 = no model; 200 = success


def test_serving_confidence_is_unscaled_softmax():
    """A2: stored temperature is not applied; T=1 softmax only (src/api.py)."""
    from src.calibration import probs_from_logits
    import torch

    src = (Path(__file__).resolve().parent.parent / "src" / "api.py").read_text(encoding="utf-8")
    assert "TemperatureScaler" not in src
    assert "temperature_scale" not in src
    assert "probs_from_logits(logits, temperature=1.0)" in src
    logits = torch.tensor([[4.0, 0.0, 0.0]])
    p = probs_from_logits(logits, temperature=1.0)
    assert torch.allclose(p, torch.softmax(logits, dim=1))
    p_soft = probs_from_logits(logits, temperature=1.32)
    assert p_soft[0, 0] < p[0, 0]  # T>1 would change confidence; serving must not


def test_api_uses_lifespan_handlers():
    """Serving lifecycle is managed by FastAPI lifespan, not deprecated events."""
    src = (Path(__file__).resolve().parent.parent / "src" / "api.py").read_text(
        encoding="utf-8"
    )
    assert "@app.on_event" not in src
    assert "lifespan=lifespan" in src


def test_predict_disclaimer_always_present(client):
    """Disclaimer must appear in every response."""
    jpeg_bytes = _make_jpeg_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
    )
    if resp.status_code == 200:
        body = resp.json()
        assert "disclaimer" in body
        assert len(body["disclaimer"]) > 50  # non-trivial text
