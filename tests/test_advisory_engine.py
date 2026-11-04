"""Deterministic unit tests for evidence grades and graded chemical advice."""

from datetime import date
import pytest

from src.advisory_engine import AdvisoryEngine, decide_advice, evidence_grade


def _entry():
    return {
        "id": "corn_rust",
        "crop": "corn_(maize)",
        "condition": "common_rust_",
        "pathogen_type": "fungal",
        "status": "VERIFIED",
        "reviewed_by": "",
        "retrieved_on": "2026-11-04",
        "expires_after_days": None,
        "source_url": "https://example.org/source",
        "guidance": {
            "scouting": [{"text": "Source-backed scouting note."}],
            "cultural": [],
            "chemical_options": [{"quote": "A direct source quotation.", "source_location": "p. 4"}],
            "label_reminder": "Consult the local label.",
        },
    }


@pytest.mark.parametrize(
    "status,ood,bg,confidence,cutoff",
    [
        ("uncertain", "accepted", "consistent", 0.99, 0.5),
        ("ok", "accepted", "inconsistent", 0.99, 0.5),
        ("ok", "rejected", "consistent", 0.99, 0.5),
        ("ok", "accepted", "consistent", 0.99, None),
    ],
)
def test_chemical_options_never_escape_full_tier(status, ood, bg, confidence, cutoff):
    advice, trace = decide_advice(
        _entry(), status=status, confidence=confidence, ood_status=ood,
        quality={"brightness": 100.0, "blur_score": 10.0}, bg_check=bg,
        confidence_cutoff=cutoff, kb_version="test-version",
    )
    assert trace["tier"] != "full"
    assert advice["chemical_options"] == []


def test_full_tier_can_show_only_cited_quote():
    advice, trace = decide_advice(
        _entry(), status="ok", confidence=0.9, ood_status="accepted",
        quality={"brightness": 100.0, "blur_score": 10.0}, bg_check="consistent",
        confidence_cutoff=0.8, kb_version="test-version",
    )
    assert trace["tier"] == "full"
    assert all("quote" in item and "source_location" in item for item in advice["chemical_options"])


def test_rejected_prediction_withholds_and_consults_local_agronomist():
    advice, trace = decide_advice(
        _entry(), status="rejected", confidence=0.0, ood_status="rejected",
        quality=None, bg_check="unknown", confidence_cutoff=0.5, kb_version="test-version",
    )
    assert trace["tier"] == "withheld"
    assert advice["message"] == "Consult a local agronomist."
    assert advice["chemical_options"] == []


def test_advice_trace_contains_required_provenance_fields():
    advice, trace = decide_advice(
        _entry(), status="uncertain", confidence=0.4, ood_status="accepted",
        quality={"brightness": 100.0, "blur_score": 10.0}, bg_check="unknown",
        confidence_cutoff=0.8, kb_version="2026-11-04:hash",
    )
    assert {
        "tier", "evidence_grade", "kb_entry_id", "source_url", "retrieved_on",
        "kb_version", "rules_fired", "bg_check",
    } <= set(trace)
    assert advice["weather_note"] == "risk unknown"


def test_expired_review_downgrades_to_grade_b():
    entry = _entry()
    entry.update(reviewed_by="reviewer", retrieved_on="2020-01-01", expires_after_days=10)
    assert evidence_grade(entry, today=date(2020, 2, 1)) == "B"


def test_confusable_hints_have_no_unsourced_differentiation(tmp_path):
    csv_path = tmp_path / "confused_pairs.csv"
    csv_path.write_text(
        "true_label,predicted_label,count\n"
        "Corn___Northern_Leaf_Blight,Corn___Cercospora,10\n",
        encoding="utf-8",
    )
    kb = tmp_path / "kb.yaml"
    kb.write_text("advisories: []\n", encoding="utf-8")
    engine = AdvisoryEngine(kb, tmp_path / "missing-weather.yaml", csv_path)
    hints = engine.get_confusable_hints("Corn___Cercospora")
    assert hints == [{
        "label": "Corn___Northern_Leaf_Blight",
        "hint": "May be confused with Corn___Northern_Leaf_Blight.",
        "how_to_tell_apart": None,
    }]
