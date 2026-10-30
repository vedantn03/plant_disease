"""
Tests for the advisory knowledge base (kb/advisories.yaml).

Verifies:
  1. YAML loads without error.
  2. Schema: all required fields present for each entry.
  3. No PLACEHOLDER entry has digits in recommendation_text.
  4. Viral/pest/HLB entries contain no fungicide keywords.
  5. VERIFIED entries have non-empty source_url.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KB_PATH = Path(__file__).resolve().parent.parent / "kb" / "advisories.yaml"

REQUIRED_FIELDS = {
    "crop", "condition", "pathogen_type", "recommendation_text",
    "caution_text", "source_name", "source_url", "last_verified", "status"
}

FUNGICIDE_KEYWORDS = {
    "fungicide", "mancozeb", "chlorothalonil", "azoxystrobin",
    "propiconazole", "tebuconazole", "metalaxyl", "cymoxanil",
}

# Pathogen types that must NOT mention fungicides
NO_FUNGICIDE_TYPES = {"viral", "pest"}
# Specific conditions that must not mention fungicides
NO_FUNGICIDE_CONDITIONS = {"haunglongbing_(citrus_greening)"}


@pytest.fixture(scope="module")
def advisories():
    assert KB_PATH.exists(), f"KB file not found: {KB_PATH}"
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("advisories", [])


def test_kb_loads(advisories):
    assert len(advisories) > 0, "No advisories found in KB"


def test_schema_all_entries(advisories):
    for i, entry in enumerate(advisories):
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"Entry {i} ({entry.get('crop')}/{entry.get('condition')}) missing fields: {missing}"


def test_no_digits_in_placeholder_recommendation(advisories):
    """PLACEHOLDER entries must not contain digits in recommendation_text."""
    for entry in advisories:
        if entry.get("status") == "PLACEHOLDER":
            rec_text = entry.get("recommendation_text", "")
            digits = re.findall(r"\d+", rec_text)
            assert not digits, (
                f"PLACEHOLDER entry {entry['crop']}/{entry['condition']} "
                f"has digits in recommendation_text: {digits}"
            )


def test_no_fungicide_in_viral_pest_hlb(advisories):
    """Viral, pest, and HLB entries must not contain fungicide wording."""
    for entry in advisories:
        ptype = entry.get("pathogen_type", "")
        condition = entry.get("condition", "").lower()

        is_no_fungicide = (
            ptype in NO_FUNGICIDE_TYPES or
            condition in NO_FUNGICIDE_CONDITIONS
        )

        if is_no_fungicide:
            rec = (entry.get("recommendation_text", "") + " " +
                   entry.get("caution_text", "")).lower()

            # Explicit warnings against fungicide use are allowed and expected.
            # Strip out disclaimers/warnings that mention fungicides are not appropriate/effective/applied.
            rec_clean = rec
            rec_clean = re.sub(r"do\s+not\s+apply\s+fungicides?", "", rec_clean)
            rec_clean = re.sub(r"fungicide[s]?\s+(treatment\s+)?(is|are)\s+not\s+(appropriate|effective)", "", rec_clean)
            rec_clean = re.sub(r"fungicides?\s+are\s+not\s+effective", "", rec_clean)

            for keyword in FUNGICIDE_KEYWORDS:
                assert keyword not in rec_clean, (
                    f"Entry {entry['crop']}/{entry['condition']} (type={ptype}) "
                    f"contains forbidden fungicide keyword: '{keyword}'"
                )


def test_verified_entries_have_source_url(advisories):
    for entry in advisories:
        if entry.get("status") == "VERIFIED":
            assert entry.get("source_url", "").startswith("http"), (
                f"VERIFIED entry {entry['crop']}/{entry['condition']} has no valid source_url"
            )


def test_pathogen_type_valid_values(advisories):
    valid = {"fungal", "bacterial", "viral", "pest", "none", "unknown"}
    for entry in advisories:
        ptype = entry.get("pathogen_type", "")
        assert ptype in valid, (
            f"Entry {entry['crop']}/{entry['condition']} has invalid pathogen_type: '{ptype}'"
        )
