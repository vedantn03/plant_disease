"""Unit checks for additive presentation, XAI jobs, and artifact-backed insights."""

import threading
import time

import numpy as np
import pytest
import torch
from PIL import Image

from src.explainability import ExplanationJobStore, _overlay, _prediction_fn
from src.insights import build_insights, latest_mlflow_run
from src.presentation import build_prediction_views, select_translated_advice


def test_presentation_keeps_raw_values_outside_farmer_display():
    views = build_prediction_views(
        predicted_class="Tomato___Early_blight",
        confidence=0.91,
        status="ok",
        ood_status="accepted",
        bg_check="consistent",
        quality={"brightness": 100.0},
        coverage={"crops": {}},
        confidence_bands={"high_min": 0.85, "medium_min": 0.65, "provisional": True},
        pathogen_type="fungal",
        rules_fired=["none"],
        kb_version="test-hash",
        evidence_grade="C",
    )
    assert views["display"]["crop"]["label"] == "Tomato"
    assert views["display"]["condition"]["label"] == "Early Blight"
    assert views["display"]["confidence_band"] == "high"
    assert "raw_class" not in views["display"]
    assert views["debug"]["raw_class"] == "Tomato___Early_blight"


def test_translation_selects_draft_and_falls_back_per_field():
    english = ["Cited source text"]
    translated = ["मशीन मसुदा"]
    assert select_translated_advice(english, translated, "mr", "MACHINE_DRAFT") == (translated, False)
    assert select_translated_advice(english, [], "mr", "MISSING") == (english, True)
    assert select_translated_advice(english, translated, "en", "MACHINE_DRAFT") == (english, False)


def test_knowledge_base_translation_status_and_draft_are_source_linked():
    import re

    import yaml

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "kb" / "advisories.yaml").read_text(encoding="utf-8"))
    entries = data["advisories"]
    assert all(entry.get("translation_status") in {"MISSING", "MACHINE_DRAFT"} for entry in entries)
    draft_entries = [entry for entry in entries if entry.get("translation_status") == "MACHINE_DRAFT"]
    assert len(draft_entries) == 1
    entry = draft_entries[0]
    assert entry["status"] == "VERIFIED" and entry["source_url"].startswith("https://")
    english = " ".join(item.get("text", "") for item in entry["guidance"].get("scouting", []))
    marathi = " ".join(item.get("text", "") for item in entry.get("scouting_mr", []))
    assert sorted(re.findall(r"\d+(?:\.\d+)?", english)) == sorted(re.findall(r"\d+(?:\.\d+)?", marathi))
    assert not re.search(r"https?://", marathi)


def test_uncertain_prediction_is_retake_and_no_diagnosis():
    display = build_prediction_views(
        predicted_class="Tomato___Early_blight", confidence=0.99, status="uncertain",
        ood_status="accepted", quality=None, coverage=None, confidence_bands={},
        bg_check="consistent",
        pathogen_type=None, rules_fired=[], kb_version="v", evidence_grade="C",
    )["display"]
    assert display["diagnosis_state"] == "retake_photo"
    assert display["show_diagnosis"] is False


def test_explanation_job_lifecycle_cache_and_cancel():
    store = ExplanationJobStore(workers=1, cache_size=2, timeout_s=2, max_pending=1)
    try:
        first = store.submit(
            image_bytes=b"image", method="lime", params={"seed": 3}, model_version="2",
            work=lambda _cancel: {"overlay": "real-test-payload"},
        )
        deadline = time.monotonic() + 2
        while store.get(first.job_id).status not in {"done", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(first.job_id).status == "done"
        cached = store.submit(
            image_bytes=b"image", method="lime", params={"seed": 3}, model_version="2",
            work=lambda _cancel: (_ for _ in ()).throw(AssertionError("cache should avoid work")),
        )
        assert cached.status == "done"
        assert cached.result["cache_hit"] is True

        gate = threading.Event()
        pending = store.submit(
            image_bytes=b"other", method="shap", params={}, model_version="2",
            work=lambda _cancel: (gate.wait(1) or {"ok": True}),
        )
        assert store.cancel(pending.job_id).reason_code == "cancelled"
        gate.set()
    finally:
        store.shutdown()


def test_explanation_job_timeout_and_failure_are_explicit():
    store = ExplanationJobStore(workers=1, cache_size=0, timeout_s=0.1, max_pending=1)
    release = threading.Event()
    started = threading.Event()
    try:
        timed = store.submit(
            image_bytes=b"slow", method="lime", params={}, model_version="2",
            work=lambda _cancel: (started.set(), release.wait(1), {"ignored": True})[2],
        )
        assert started.wait(1)
        time.sleep(0.12)
        assert store.get(timed.job_id).status == "timeout"
        release.set()
        assert timed.future is not None and timed.future.result(timeout=1) is None

        failed = store.submit(
            image_bytes=b"bad", method="shap", params={}, model_version="2",
            work=lambda _cancel: (_ for _ in ()).throw(RuntimeError("synthetic explainer error")),
        )
        deadline = time.monotonic() + 1
        while store.get(failed.job_id).status not in {"failed", "done"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(failed.job_id).status == "failed"
        assert store.get(failed.job_id).message == "synthetic explainer error"
    finally:
        release.set()
        store.shutdown()


def test_overlay_matches_model_input_and_has_finite_fraction():
    image = Image.new("RGB", (40, 30), (40, 100, 30))
    heat = np.zeros((8, 10), dtype=np.float32)
    heat[:, 5:] = 1.0
    png, fraction, mask = _overlay(image, heat, 0.10)
    assert isinstance(png, str) and png
    assert isinstance(mask, str) and mask
    assert fraction == 0.10


def test_overlay_refuses_spatially_uniform_attributions():
    image = Image.new("RGB", (20, 20), (50, 100, 20))
    with pytest.raises(ValueError, match="spatially distinguishable"):
        _overlay(image, np.ones((4, 4), dtype=np.float32), 0.10)


def test_prediction_function_returns_finite_class_probability_matrix():
    class TinyModel(torch.nn.Module):
        def forward(self, tensor):
            mean = tensor.mean(dim=(1, 2, 3))
            return torch.stack((mean, -mean), dim=1)

    predict = _prediction_fn(TinyModel(), "cpu", 16, None, prepared_rgb=True)
    output = predict(np.zeros((2, 16, 16, 3), dtype=np.uint8))
    assert output.shape == (2, 2)
    assert np.isfinite(output).all()
    assert np.allclose(output.sum(axis=1), 1.0)


def test_lime_prediction_preprocessing_matches_serving_preprocessing():
    class TinyModel(torch.nn.Module):
        def forward(self, tensor):
            pooled = tensor.mean(dim=(2, 3))
            return torch.stack((pooled[:, 0], pooled[:, 1], pooled[:, 2]), dim=1)

    image = Image.fromarray(np.random.default_rng(19).integers(0, 256, (39, 61, 3), dtype=np.uint8))
    serving_tensor = __import__("src.model", fromlist=["preprocess_image"]).preprocess_image(image, 16)
    serving_logits = TinyModel()(serving_tensor)
    expected = torch.softmax(serving_logits, dim=1).numpy()
    from torchvision import transforms

    model_input = transforms.CenterCrop(16)(transforms.Resize(18)(image))
    pixels = np.asarray(model_input, dtype=np.uint8)
    lime_predict = _prediction_fn(TinyModel(), "cpu", 16, None, prepared_rgb=True)
    actual = lime_predict(pixels[None, ...])
    assert np.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_insights_missing_artifacts_are_explicit(tmp_path):
    payload = build_insights(
        tmp_path,
        serving={"model_loaded": False},
        process_stats={"request_count": 0},
        telemetry={"available": False, "records": [], "summary": {}},
        class_names=[],
        max_report_bytes=10000,
    )
    assert payload["data"]["class_count_from_serving_model"] is None
    assert payload["data"]["split_counts"]["status"] == "not generated yet"
    assert payload["data"]["quality_audit"] is None
    assert payload["reports"]["evaluation"] is None
    assert payload["pipeline"]["status"] == "not_configured"


def test_insights_exposes_only_supplied_drift_thresholds(tmp_path):
    payload = build_insights(
        tmp_path,
        serving={"model_loaded": False},
        process_stats={},
        telemetry={"available": False, "records": [], "summary": {}},
        class_names=[],
        max_report_bytes=1000,
        drift_config={"psi_threshold": 0.2, "ks_alpha": 0.05, "thresholds_provisional": True},
    )
    assert payload["monitoring_config"] == {
        "psi_threshold": 0.2,
        "ks_alpha": 0.05,
        "thresholds_provisional": True,
    }


def test_latest_mlflow_run_is_read_only_and_sanitized():
    experiment = type("Experiment", (), {"experiment_id": "exp-1", "name": "course-run"})()
    info = type("RunInfo", (), {
        "experiment_id": "exp-1", "start_time": 1234, "status": "FINISHED"
    })()
    data = type("RunData", (), {
        "params": {
            "arch": "resnet18", "seed": "42", "data_root": "D:\\private\\dataset",
            "git_commit": "private-run-id",
        },
        "metrics": {"val_macro_f1": 0.91, "train_loss": 0.12, "unreviewed_metric": 99.0},
    })()
    run = type("Run", (), {"info": info, "data": data})()

    class ReadOnlyClient:
        def __init__(self, *, tracking_uri):
            assert tracking_uri == "sqlite:///local.db"

        def search_experiments(self):
            return [experiment]

        def search_runs(self, **kwargs):
            assert kwargs["experiment_ids"] == ["exp-1"]
            assert kwargs["order_by"] == ["attributes.start_time DESC"]
            return [run]

    result = latest_mlflow_run("sqlite:///local.db", client_factory=ReadOnlyClient)
    assert result["status"] == "available"
    assert result["run"]["params"] == {"arch": "resnet18", "seed": "42"}
    assert result["run"]["metrics"] == {"val_macro_f1": 0.91, "train_loss": 0.12}
    assert result["run"]["started_at_ms"] == 1234


def test_latest_mlflow_run_missing_and_unavailable_states_are_explicit():
    assert latest_mlflow_run(None) == {"status": "not_configured", "run": None}

    class BrokenClient:
        def __init__(self, *, tracking_uri):
            raise OSError("private database path")

    result = latest_mlflow_run("sqlite:///local.db", client_factory=BrokenClient)
    assert result == {"status": "unavailable", "run": None, "reason": "OSError"}
