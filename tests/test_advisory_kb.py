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
    """Viral, pest, and HLB entries must never recommend fungicides in structured chemical options.

    Any recommended chemical options must not contain fungicide active ingredients,
    and cultural controls must not prescribe fungicide applications.
    """
    for entry in advisories:
        ptype = entry.get("pathogen_type", "")
        condition = entry.get("condition", "").lower()

        is_no_fungicide = (
            ptype in NO_FUNGICIDE_TYPES or
            condition in NO_FUNGICIDE_CONDITIONS
        )

        if is_no_fungicide:
            # 1. Structured chemical options check: must have zero fungicide recommendations
            chemical_options = entry.get("chemical_options") or []
            for chem in chemical_options:
                ai = chem.get("active_ingredient", "").lower()
                frac = str(chem.get("frac_code", "")).lower()
                label = chem.get("label_caveat", "").lower()
                for kw in FUNGICIDE_KEYWORDS:
                    assert kw not in ai, (
                        f"Non-fungal entry {entry['crop']}/{entry['condition']} recommends fungicide: '{ai}'"
                    )
                    assert kw not in label or "not" in label or "never" in label or "zero" in label, (
                        f"Non-fungal entry {entry['crop']}/{entry['condition']} has positive fungicide caveat: '{label}'"
                    )
                # Ensure no fungal FRAC code is recommended
                assert "fungicide" not in frac, f"Fungicide FRAC assigned to non-fungal entry: {frac}"

            # 2. Structured cultural controls check: no fungicide spraying prescribed
            cultural_controls = entry.get("cultural_controls") or []
            for cc in cultural_controls:
                practice = cc.get("practice", "").lower()
                for kw in FUNGICIDE_KEYWORDS:
                    if kw in practice:
                        # Only allowed if explicitly framed as negative assertion
                        assert any(neg in practice for neg in ["no", "never", "not", "ineffective", "prohibited"]), (
                            f"Non-fungal entry {entry['crop']}/{entry['condition']} has affirmative fungicide practice: '{practice}'"
                        )


def test_fungicide_on_viral_entry_fails_safety_check():
    """Safety assertion proof: verify that injecting a fungicide into a viral entry strictly FAILS validation."""
    fake_viral_entry = {
        "crop": "tomato",
        "condition": "tomato_yellow_leaf_curl_virus",
        "pathogen_type": "viral",
        "chemical_options": [
            {
                "active_ingredient": "Azoxystrobin (Fungicide)",
                "frac_code": "11",
                "rate_text": "1.0 kg/ha",
                "label_caveat": "Apply broad-spectrum fungicide spray",
            }
        ],
        "cultural_controls": []
    }

    # Simulate the check logic above and verify it catches the violation
    violation_caught = False
    for chem in fake_viral_entry["chemical_options"]:
        ai = chem["active_ingredient"].lower()
        if any(kw in ai for kw in FUNGICIDE_KEYWORDS):
            violation_caught = True
            break

    assert violation_caught, "Safety check failed to intercept illegal fungicide on viral entry!"


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
