"""Pure UI parsing helpers are tested with synthetic API/report payloads."""

import json

from app.ui_helpers import (
    explanation_text,
    load_saved_mlops_evidence,
    parse_prediction_trust,
    prediction_explanation_values,
)


def test_explanation_languages_fall_back_to_english_until_reviewed():
    english, english_available = explanation_text("explanation_scope", "en")
    marathi, marathi_available = explanation_text("explanation_scope", "mr")
    hindi, hindi_available = explanation_text("explanation_scope", "hi")

    assert english
    assert english_available is True
    assert marathi == english
    assert hindi == english
    assert marathi_available is False
    assert hindi_available is False


def test_explanation_values_preserve_api_prediction_and_evidence():
    advice_trace = {"tier": "scout", "source_url": "https://source.example/page"}
    payload = {
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.8123,
        "gradcam_png_base64": "image-data",
        "advice_trace": advice_trace,
    }
    values = prediction_explanation_values(payload)

    assert values == {
        "label": "Tomato___Early_blight",
        "confidence": 0.8123,
        "gradcam_available": True,
        "advice_trace": advice_trace,
    }
    assert values["advice_trace"] is advice_trace


def test_prediction_trust_parser_uses_api_values_and_unknown_fallbacks():
    parsed = parse_prediction_trust({
        "crop": "apple",
        "condition": "healthy",
        "confidence": 0.91,
        "quality": {"brightness": 112.0, "blur_score": 4.0},
        "ood_status": "accepted",
        "bg_check": "inconsistent",
        "bg_check_details": {"confidence_change": -0.2},
    })

    assert parsed["label"] == "apple___healthy"
    assert parsed["confidence"] == 0.91
    assert parsed["quality"]["brightness"] == 112.0
    assert parsed["ood_status"] == "accepted"
    assert parsed["bg_check"] == "inconsistent"
    assert parsed["bg_check_details"]["confidence_change"] == -0.2
    assert parse_prediction_trust({})["bg_check"] == "unknown"
    assert parse_prediction_trust({})["confidence"] is None


def test_saved_evidence_loads_available_files_and_reports_missing_files(tmp_path):
    (tmp_path / "seed_variance.json").write_text(
        json.dumps({"models": {"champion": {"run_id": "run-a", "seed": 42}}}),
        encoding="utf-8",
    )
    (tmp_path / "candidate_gate_runb.json").write_text(
        json.dumps({"results": [{"name": "candidate", "run_id": "run-b", "seed": 43}]}),
        encoding="utf-8",
    )
    (tmp_path / "promotion_log.jsonl").write_text(
        json.dumps({"verdict": "DRY_RUN", "alias_moved": False}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "drift_latest.json").write_text(
        json.dumps({"status": "insufficient_data", "drift_flag": None}),
        encoding="utf-8",
    )

    evidence = load_saved_mlops_evidence(tmp_path, {"model_version": "2"})

    assert [row["run_id"] for row in evidence["models"]] == ["run-a", "run-b"]
    assert evidence["promotions"][0]["verdict"] == "DRY_RUN"
    assert evidence["drift"]["status"] == "insufficient_data"
    assert evidence["serving_model_version"] == "2"


def test_saved_evidence_missing_files_has_clear_messages(tmp_path):
    evidence = load_saved_mlops_evidence(tmp_path, None)
    assert evidence["models_message"] == "not generated yet"
    assert evidence["promotions_message"] == "not generated yet"
    assert evidence["drift_message"] == "not generated yet"
    assert evidence["serving_message"] == "not available from /health"
