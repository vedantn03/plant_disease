"""Pure UI parsing helpers are tested with synthetic API/report payloads."""

import json
import ast
import platform
from pathlib import Path

import pytest

from app.ui_helpers import (
    explanation_text,
    import_optional_explainer,
    load_saved_mlops_evidence,
    optional_explainer_available,
    parse_prediction_trust,
    prediction_explanation_values,
    saved_explainer_figures,
)


def test_explanation_language_changes_ui_copy_without_review_claim():
    english, english_available = explanation_text("explanation_scope", "en")
    marathi, marathi_available = explanation_text("explanation_scope", "mr")
    hindi, hindi_available = explanation_text("explanation_scope", "hi")

    assert english
    assert english_available is True
    assert marathi != english
    assert hindi != english
    assert marathi_available is False
    assert hindi_available is False
    assert "मराठी" in explanation_text("draft_translation_notice", "mr")[0]
    assert "मसौदा" in explanation_text("draft_translation_notice", "hi")[0]


def test_explanation_values_preserve_api_prediction_and_evidence():
    advice_trace = {"tier": "scout", "source_url": "https://source.example/page"}
    payload = {
        "crop": "tomato",
        "condition": "early_blight",
        "confidence": 0.8123,
        "gradcam_png_base64": "image-data",
        "advice_trace": advice_trace,
    }
    values = prediction_explanation_values(payload)

    assert values == {
        "label": "tomato___early_blight",
        "confidence": 0.8123,
        "gradcam_available": True,
        "advice_trace": advice_trace,
    }
    assert values["advice_trace"] is advice_trace


def test_selected_language_preserves_prediction_and_evidence_values():
    payload = {
        "crop": "tomato",
        "condition": "late_blight",
        "confidence": 0.8656,
        "gradcam_png_base64": "unchanged-figure-bytes",
        "advice_trace": {"evidence_grade": "C", "kb_version": "kb-hash"},
    }
    values = prediction_explanation_values(payload)
    lines = [
        explanation_text("prediction_line", locale, label=values["label"], confidence="86.6%")[0]
        for locale in ("en", "mr", "hi")
    ]

    assert values["label"] == "tomato___late_blight"
    assert values["confidence"] == payload["confidence"]
    assert values["gradcam_available"] is True
    assert values["advice_trace"] is payload["advice_trace"]
    for line in lines:
        assert values["label"] in line
        assert "86.6%" in line
    assert len(set(lines)) == 3


def test_missing_optional_explainer_is_handled(tmp_path, monkeypatch):
    import app.ui_helpers as ui_helpers

    monkeypatch.setattr(ui_helpers.importlib, "import_module", lambda _name: _raise_import())
    monkeypatch.setattr(ui_helpers.importlib.util, "find_spec", lambda _name: None)
    assert import_optional_explainer("missing_explainer") is None
    assert optional_explainer_available("missing_explainer") is False
    assert saved_explainer_figures(tmp_path, "shap") == []
    assert saved_explainer_figures(tmp_path, "lime") == []


def test_saved_explainer_figures_use_only_existing_offline_results(tmp_path):
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    shap_figure = shap_dir / "compare_example.png"
    lime_figure = tmp_path / "xai_trio_comparison.png"
    shap_figure.write_bytes(b"synthetic image bytes")
    lime_figure.write_bytes(b"synthetic image bytes")

    assert saved_explainer_figures(tmp_path, "shap") == [shap_figure]
    assert saved_explainer_figures(tmp_path, "lime") == [lime_figure]
    assert saved_explainer_figures(tmp_path, "unknown") == []


def _raise_import():
    raise ModuleNotFoundError("optional package is absent")


def test_app_smoke_without_api_and_no_nested_expanders(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    if platform.system() == "Windows":
        pytest.skip(
            "AppTest cannot initialize asyncio socketpair in this Windows sandbox; "
            "Ubuntu CI runs the headless app test."
        )

    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    app_path = Path(__file__).resolve().parents[1] / "app" / "ui.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run(timeout=15)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Diagnostic Screening",
        "Production Telemetry",
        "MLOps System",
    ]


def test_ui_has_no_nested_streamlit_expanders():
    app_path = Path(__file__).resolve().parents[1] / "app" / "ui.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    class ExpanderNestingCheck(ast.NodeVisitor):
        depth = 0

        @staticmethod
        def opens_expander(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "expander"
            )

        def visit_With(self, node: ast.With) -> None:
            opens = sum(self.opens_expander(item.context_expr) for item in node.items)
            assert self.depth + opens <= 1, "Streamlit expanders must not be nested"
            self.depth += opens
            for statement in node.body:
                self.visit(statement)
            self.depth -= opens
            for item in node.items:
                self.visit(item.context_expr)

    ExpanderNestingCheck().visit(tree)


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
