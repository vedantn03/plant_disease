"""CI-only API plumbing test using random weights stored in a temporary folder."""

from __future__ import annotations

import io
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import api  # noqa: E402


class TinyFixtureModel(nn.Module):
    """Small untrained network with the layer expected by the API's GradCAM wrapper."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer4(images)
        return self.fc(self.pool(features).flatten(1))


class FakeGradCAM:
    def __init__(self, _model) -> None:
        pass

    def generate(self, *_args, **_kwargs):
        return None

    def remove_hooks(self) -> None:
        return None


@pytest.fixture
def fixture_checkpoint(tmp_path: Path) -> Path:
    """Create untrained random weights; this fixture is not an accuracy demo."""
    torch.manual_seed(7)
    classes = {
        "Apple___Apple_scab": 0,
        "Apple___Black_rot": 1,
        "Apple___Cedar_apple_rust": 2,
        "Apple___healthy": 3,
        "Corn_(maize)___Common_rust_": 4,
    }
    model = TinyFixtureModel(num_classes=len(classes))
    checkpoint = tmp_path / "fixture_model.pth"
    torch.save({"class_to_idx": classes, "model_state_dict": model.state_dict()}, checkpoint)
    return checkpoint


def _image_bytes() -> bytes:
    image = Image.new("RGB", (96, 96), (40, 150, 60))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_temporary_model_path_is_refused_outside_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANT_DISEASE_TEST_MODEL", "temporary-fixture.pth")
    monkeypatch.setenv("ENV", "production")
    with pytest.raises(RuntimeError, match="only when ENV=test"):
        api._check_test_model_environment()


def test_fixture_model_exercises_health_predict_and_metrics(
    fixture_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if sys.platform == "win32":
        pytest.skip(
            "Starlette TestClient socketpair is blocked in this Windows sandbox; Ubuntu CI runs it"
        )
    torch.set_num_threads(1)
    monkeypatch.setenv("PLANT_DISEASE_TEST_MODEL", str(fixture_checkpoint))
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "absent-dataset"))
    assert api._check_test_model_environment() == str(fixture_checkpoint)

    cfg = SimpleNamespace(
        data=SimpleNamespace(image_size=8),
        background_check=SimpleNamespace(
            bg_check_enabled=False,
            abstain_on_bg_inconsistent=False,
            min_mask_area_fraction=0.05,
            max_mask_area_fraction=0.95,
        ),
        serving=SimpleNamespace(
            model_source="registry",
            registry_strict=False,
            log_file=str(tmp_path / "requests.jsonl"),
            top_k=5,
            confidence_threshold=0.0,
            max_upload_bytes=1024**2,
        ),
    )

    # The API contract is under test; leaf plausibility is not.
    with (
        patch.object(api, "load_config", return_value=cfg),
        patch.object(api.torch.cuda, "is_available", return_value=False),
        patch.object(
            api,
            "resolve_serving_checkpoint",
            side_effect=AssertionError("fixture path expected"),
        ),
        patch.object(
            api,
            "build_model",
            side_effect=lambda num_classes, pretrained: TinyFixtureModel(num_classes),
        ),
        patch.object(api, "GradCAM", FakeGradCAM),
        patch.object(
            api,
            "AdvisoryEngine",
            return_value=SimpleNamespace(
                get_advisory=lambda *_args, **_kwargs: {"recommendation_text": "fixture"}
            ),
        ),
        patch.object(api, "preprocess_image", return_value=torch.rand(1, 3, 8, 8)),
        patch.object(api, "check_image_heuristics", return_value=(True, None, None)),
        patch.object(api, "heatmap_to_base64_png", return_value="fixture-heatmap"),
    ):
        asyncio.run(api.startup())
        try:
            # Do not enter TestClient's lifespan: startup already loaded the temporary fixture.
            client = TestClient(api.app)
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["model_loaded"] is True
            assert health.json()["model_source"] == "test_fixture"

            prediction = client.post(
                "/predict",
                files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
            )
            assert prediction.status_code == 200
            body = prediction.json()
            assert body["status"] == "ok"
            assert isinstance(body["confidence"], float)
            assert isinstance(body["top_k"], list)
            assert "advisory" in body

            metrics = client.get("/metrics")
            assert metrics.status_code == 200
            metrics_text = metrics.text
        finally:
            asyncio.run(api.shutdown())
            api._state.update({"model": None, "grad_cam": None, "cfg": None, "log_file": None})
        assert "predict_requests_total" in metrics_text
        assert "predict_latency_seconds" in metrics_text
        assert "predict_confidence" in metrics_text
        assert "model_info" in metrics_text
