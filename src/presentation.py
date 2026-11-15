"""Farmer-facing presentation fields derived from serving and coverage data."""

from __future__ import annotations

from typing import Any

from src.utils import parse_class_name


def _humanize(value: str) -> str:
    cleaned = value.replace("_", " ").replace("(", "(").strip()
    return " ".join(word.capitalize() for word in cleaned.split())


def select_translated_advice(
    english: list[Any], translated: list[Any] | None, locale: str, status: str
) -> tuple[list[Any], bool]:
    """Select draft Marathi only when present; report per-field English fallback."""
    if locale != "mr":
        return english, False
    if status == "MACHINE_DRAFT" and translated:
        return translated, False
    return english, bool(english)


def build_prediction_views(
    *,
    predicted_class: str | None,
    confidence: float | None,
    status: str,
    ood_status: str,
    bg_check: str,
    quality: dict[str, Any] | None,
    coverage: dict[str, Any] | None,
    confidence_bands: dict[str, Any] | None,
    pathogen_type: str | None,
    rules_fired: list[str] | None,
    kb_version: str,
    evidence_grade: str | None,
) -> dict[str, dict[str, Any]]:
    """Build additive UI and developer fields; preserve raw values in `debug`."""
    crop_key = None
    condition_key = None
    if predicted_class:
        try:
            crop_key, condition_key = parse_class_name(predicted_class)
        except ValueError:
            crop_key, condition_key = None, None

    band_config = confidence_bands or {}
    high_min = float(band_config.get("high_min", 1.0))
    medium_min = float(band_config.get("medium_min", 1.0))
    if status in {"uncertain", "rejected", "rejected_not_a_leaf", "abstained"}:
        confidence_band = "uncertain"
        diagnosis_state = "retake_photo"
        next_steps = ["retake_photo", "consult_local_expert"]
    else:
        value = confidence if confidence is not None else -1.0
        confidence_band = "high" if value >= high_min else "medium" if value >= medium_min else "low"
        if confidence_band == "low" or ood_status != "accepted" or bg_check == "inconsistent":
            diagnosis_state = "retake_photo"
            next_steps = ["retake_photo", "consult_local_expert"]
        else:
            diagnosis_state = "healthy" if condition_key == "healthy" else "disease_likely"
            next_steps = ["inspect_other_leaves", "consult_local_expert"]

    crop_label = _humanize(crop_key) if crop_key else None
    condition_label = _humanize(condition_key) if condition_key else None
    coverage_warning = None
    crops = coverage.get("crops", {}) if isinstance(coverage, dict) else {}
    if crop_key:
        crop_entry = next(
            (value for name, value in crops.items() if name.lower() == crop_key.lower()), None
        )
        classes = crop_entry.get("classes", []) if isinstance(crop_entry, dict) else []
        if len(classes) == 1:
            only_class = str(classes[0]).split("___", maxsplit=1)[-1]
            coverage_warning = {
                "status": "single_condition_only",
                "available_condition": _humanize(only_class),
                "message_key": "single_condition_coverage",
            }

    display = {
        "crop": {"key": crop_key, "label": crop_label},
        "condition": {"key": condition_key, "label": condition_label},
        "show_diagnosis": diagnosis_state != "retake_photo",
        "diagnosis_state": diagnosis_state,
        "confidence_band": confidence_band,
        "confidence_band_provisional": bool(band_config.get("provisional", True)),
        "coverage_warning": coverage_warning,
        "pathogen_type": pathogen_type or "unknown",
        "next_step_keys": next_steps,
    }
    debug = {
        "raw_class": predicted_class,
        "raw_confidence": confidence,
        "ood_status": ood_status,
        "quality": quality,
        "rules_fired": list(rules_fired or []),
        "kb_version": kb_version,
        "evidence_grade": evidence_grade or "C",
    }
    return {"display": display, "debug": debug}
