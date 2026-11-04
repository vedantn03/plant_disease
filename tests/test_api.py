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
from conftest import api_model_skip_reason

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
    reason = api_model_skip_reason(project_root)
    if reason:
        pytest.skip(reason)
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
    assert "model_version" in body
    assert "model_source" in body
    assert "checkpoint_md5_prefix" in body
    assert "registry_match" in body


def test_metrics_names_after_prediction_or_skip(client):
    jpeg_bytes = _make_jpeg_bytes(color=(60, 150, 70))
    response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
        headers={"X-Traffic-Source": "replay"},
    )
    if response.status_code == 503:
        pytest.skip("Model unavailable; metrics integration requires local model weights")
    metrics = client.get("/metrics")
    if metrics.status_code == 503:
        pytest.skip("prometheus-client unavailable in the test environment")
    assert metrics.status_code == 200
    body = metrics.text
    assert "predict_requests_total" in body
    assert "predict_latency_seconds" in body
    assert "predict_confidence" in body
    assert "bg_check_total" in body
    assert "advice_total" in body
    assert "advice_withheld_total" in body
    assert "advice_gap_total" in body
    assert 'traffic_source="replay"' in body
    assert "model_info" in body




def test_metrics_endpoint_without_model_can_be_checked_with_skip_helper():
    project_root = Path(__file__).resolve().parent.parent
    reason = api_model_skip_reason(project_root)
    if reason:
        pytest.skip(reason)
    if not __import__("src.api", fromlist=["PROMETHEUS_AVAILABLE"]).PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus-client unavailable")


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
    assert {"quality", "ood_status", "bg_check", "bg_check_details"} <= body.keys()
    assert body["bg_check"] in {"consistent", "inconsistent", "unknown"}
    assert body["bg_check_details"]["result"] == body["bg_check"]

    if body["status"] == "ok":
        assert "crop" in body
        assert "condition" in body
        assert "confidence" in body
        assert isinstance(body["confidence"], float)
        assert "advisory" in body
        assert "advice_trace" in body
        assert "kb_version" in body
        assert body["advisory"]["tier"] in {"withheld", "scouting", "full"}
        assert body["advice_trace"]["kb_version"] == body["kb_version"]
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


def test_api_artifact_skip_helper_allows_complete_local_fixture(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "best_model.pth").write_bytes(b"fixture")
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    assert api_model_skip_reason(tmp_path) is None


def test_api_model_skip_helper_reports_missing_artifacts(tmp_path):
    assert api_model_skip_reason(tmp_path) is not None
