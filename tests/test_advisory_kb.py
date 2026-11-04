"""Schema, provenance, freshness, and structured-content checks for advisory KB v2."""

import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from src.advisory_engine import evidence_grade, is_well_formed_http_url

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "kb" / "advisories.yaml"
REQUIRED = {
    "id", "crop", "condition", "pathogen_type", "source_url", "publisher",
    "retrieved_on", "reviewed_by", "expires_after_days", "guidance", "status",
}
FUNGICIDE_TERMS = ("fungicide", "mancozeb", "chlorothalonil", "azoxystrobin")
NO_FUNGICIDE_TYPES = {"viral", "pest"}
NO_FUNGICIDE_CONDITIONS = {"haunglongbing_(citrus_greening)"}


@pytest.fixture(scope="module")
def advisories():
    return yaml.safe_load(KB_PATH.read_text(encoding="utf-8"))["advisories"]


def test_kb_loads_and_has_v2_required_fields(advisories):
    assert advisories
    for entry in advisories:
        assert REQUIRED <= set(entry)
        assert {"scouting", "cultural", "chemical_options", "label_reminder"} <= set(entry["guidance"])


def test_verified_entries_require_provenance_and_retrieval_date(advisories):
    for entry in advisories:
        if entry["status"] == "VERIFIED":
            assert entry["id"] and entry["publisher"] and entry["retrieved_on"]
            assert is_well_formed_http_url(entry["source_url"])
            date.fromisoformat(entry["retrieved_on"])


def test_all_nonempty_urls_are_well_formed(advisories):
    for entry in advisories:
        if entry["source_url"]:
            assert is_well_formed_http_url(entry["source_url"])


def test_reviewed_entry_expiration_changes_evidence_grade():
    entry = {
        "status": "VERIFIED", "reviewed_by": "reviewer", "retrieved_on": "2020-01-01",
        "expires_after_days": 30,
    }
    assert evidence_grade(entry, today=date(2020, 1, 2)) == "A"
    assert evidence_grade(entry, today=date(2020, 3, 1)) == "B"


def test_unreviewed_and_placeholder_evidence_grades():
    assert evidence_grade({"status": "VERIFIED", "reviewed_by": ""}) == "B"
    assert evidence_grade({"status": "PLACEHOLDER"}) == "C"


def test_no_plantdisease_cornell_attribution(advisories):
    for entry in advisories:
        assert "plantdisease.net" not in entry.get("publisher", "").lower()
        assert "plantdisease.net" not in entry.get("source_name", "").lower()


def test_viral_pest_hlb_have_no_fungicide_options(advisories):
    for entry in advisories:
        if entry["pathogen_type"] in NO_FUNGICIDE_TYPES or entry["condition"] in NO_FUNGICIDE_CONDITIONS:
            assert entry["guidance"]["chemical_options"] == []


def test_dosage_or_chemical_options_must_be_source_quotes(advisories):
    dosage = re.compile(r"\b(?:\d+(?:\.\d+)?\s*(?:g|kg|ml|mL|L|%|per\s+liter|per\s+ha))\b", re.I)
    for entry in advisories:
        options = entry["guidance"]["chemical_options"]
        if options:
            assert entry["source_url"]
            for option in options:
                assert option.get("quote")
                assert option.get("source_location")
                assert not dosage.search(option["quote"]) or entry["source_url"]


def test_untraceable_entries_are_placeholders_without_guidance(advisories):
    for entry in advisories:
        if entry["status"] == "PLACEHOLDER":
            assert entry["guidance"]["scouting"] == []
            assert entry["guidance"]["cultural"] == []
            assert entry["guidance"]["chemical_options"] == []
            assert entry["guidance"]["label_reminder"] == "Consult a local agronomist."


def test_only_retained_corn_rust_text_is_verified(advisories):
    verified = [entry for entry in advisories if entry["status"] == "VERIFIED"]
    assert [(item["crop"], item["condition"]) for item in verified] == [
        ("corn_(maize)", "common_rust_")
    ]


def test_audit_inventory_matches_every_kb_entry_once(advisories):
    audit = (ROOT / "docs" / "ADVISORY_AUDIT.md").read_text(encoding="utf-8")
    inventory = audit.split("## Entry-by-entry inventory", 1)[1].split(
        "## Manual source checks requested", 1
    )[0]
    rows = []
    for line in inventory.splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in {"Crop", ""}:
            continue
        assert len(cells) == 8, f"Malformed audit inventory row: {line}"
        rows.append(cells)

    expected = {(entry["crop"], entry["condition"]) for entry in advisories}
    actual = [(row[0], row[1]) for row in rows]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected
    assert len(rows) == len(advisories)

    audit_counts = {"VERIFIED": 0, "UNVERIFIED-SOURCE": 0, "PLACEHOLDER": 0}
    active_counts = {"VERIFIED": 0, "PLACEHOLDER": 0}
    for row in rows:
        audit_counts[row[2]] += 1
        active_counts[row[3]] += 1
        assert row[4] != "—" if row[2] == "UNVERIFIED-SOURCE" else True
    assert audit_counts == {"VERIFIED": 1, "UNVERIFIED-SOURCE": 25, "PLACEHOLDER": 12}
    assert active_counts == {"VERIFIED": 1, "PLACEHOLDER": 37}
    manual_urls = audit.split("<!-- UNVERIFIED_URLS_BEGIN -->", 1)[1].split(
        "<!-- UNVERIFIED_URLS_END -->", 1
    )[0]
    listed_urls = re.findall(r"https?://\S+", manual_urls)
    unverified_urls = [row[4] for row in rows if row[2] == "UNVERIFIED-SOURCE"]
    assert len(listed_urls) == 25
    assert set(listed_urls) == set(unverified_urls)
    assert "`plantdisease.net` is not a Cornell domain" in audit
