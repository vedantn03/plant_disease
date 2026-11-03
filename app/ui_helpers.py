"""Pure parsing helpers for API trust responses and saved MLOps reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_prediction_trust(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract only fields the API returned; absent values remain unavailable."""
    crop = payload.get("crop")
    condition = payload.get("condition")
    label = payload.get("predicted_class")
    if not label and crop and condition:
        label = f"{crop}___{condition}"
    if not label:
        label = condition or "not available"
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        quality = None
    details = payload.get("bg_check_details")
    if not isinstance(details, dict):
        details = {}
    bg_check = payload.get("bg_check")
    if bg_check not in {"consistent", "inconsistent", "unknown"}:
        bg_check = "unknown"
    return {
        "label": label,
        "confidence": float(confidence) if confidence is not None else None,
        "quality": quality,
        "ood_status": payload.get("ood_status") or "unknown",
        "bg_check": bg_check,
        "bg_check_details": details,
    }


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, "not generated yet"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read saved report: {exc}"


def _model_rows(reports_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seed_data, seed_error = _read_json(reports_dir / "seed_variance.json")
    if isinstance(seed_data, dict):
        for role, model in seed_data.get("models", {}).items():
            if not isinstance(model, dict):
                continue
            run_id = str(model.get("run_id") or "not available")
            if run_id != "not available":
                seen_run_ids.add(run_id)
            rows.append({"role": role, **model})

    candidate_files = sorted(reports_dir.glob("candidate_gate_*.json"))
    for report_path in candidate_files:
        report, _ = _read_json(report_path)
        if not isinstance(report, dict):
            continue
        for candidate in report.get("results", []):
            if not isinstance(candidate, dict):
                continue
            run_id = str(candidate.get("run_id") or "not available")
            if run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            rows.append({
                "role": candidate.get("name") or "candidate",
                "run_id": run_id,
                "seed": candidate.get("seed", "not available"),
                "clean_macro_f1": candidate.get("clean_macro_f1", "not available"),
                "within_crop_macro_f1": candidate.get("within_crop_macro_f1", "not available"),
                "min_recall": candidate.get("min_recall", {}).get("value", "not available"),
                "ece": candidate.get("ece", "not available"),
                "report": report_path.name,
            })
    if rows:
        return rows, None
    if seed_error:
        return [], seed_error
    return [], "not generated yet"


def load_saved_mlops_evidence(reports_dir: Path, health: dict[str, Any] | None) -> dict[str, Any]:
    """Load only committed/generated report files and the supplied /health response."""
    model_rows, model_message = _model_rows(reports_dir)

    promotion_path = reports_dir / "promotion_log.jsonl"
    promotions = []
    promotion_message = None
    if not promotion_path.exists():
        promotion_message = "not generated yet"
    else:
        try:
            for line in promotion_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    promotions.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            promotions = []
            promotion_message = f"could not read saved report: {exc}"
        if not promotions and promotion_message is None:
            promotion_message = "not generated yet"

    drift, drift_message = _read_json(reports_dir / "drift_latest.json")
    serving_version = None
    if isinstance(health, dict):
        serving_version = health.get("model_version")

    return {
        "models": model_rows,
        "models_message": model_message,
        "promotions": promotions,
        "promotions_message": promotion_message,
        "drift": drift,
        "drift_message": drift_message,
        "serving_model_version": serving_version,
        "serving_message": None if serving_version is not None else "not available from /health",
    }
