"""Source-aware, conservative advisory selection and provenance helpers."""

from __future__ import annotations

import hashlib
import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from src.utils import get_logger, parse_class_name

log = get_logger("advisory")
DISCLAIMER = (
    "Decision support only; not a substitute for a local agronomist or official product label."
)


def evidence_grade(entry: dict[str, Any], today: date | None = None) -> str:
    if entry.get("status") == "PLACEHOLDER":
        return "C"
    if entry.get("status") != "VERIFIED":
        return "C"
    if not entry.get("reviewed_by"):
        return "B"
    expires = entry.get("expires_after_days")
    retrieved = entry.get("retrieved_on")
    if not expires or not retrieved:
        return "B"
    try:
        retrieved_date = date.fromisoformat(retrieved)
    except (TypeError, ValueError):
        return "B"
    today = today or date.today()
    if today > retrieved_date + timedelta(days=int(expires)):
        return "B"
    return "A"


def decide_advice(
    entry: dict[str, Any] | None,
    *,
    status: str,
    confidence: float | None,
    ood_status: str,
    quality: dict[str, Any] | None,
    bg_check: str,
    confidence_cutoff: float | None,
    kb_version: str,
    rules_fired: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return allowed advice and its trace; chemical options are top-tier only."""
    grade = evidence_grade(entry or {})
    refused = status in {"rejected", "rejected_not_a_leaf", "abstained"}
    sufficient_measurements = bool(quality) and all(
        isinstance(quality.get(key), (int, float)) for key in ("brightness", "blur_score")
    )
    confidence_ok = (
        confidence_cutoff is not None
        and confidence is not None
        and confidence >= confidence_cutoff
    )
    full = (
        not refused
        and status == "ok"
        and grade in {"A", "B"}
        and ood_status == "accepted"
        and bg_check == "consistent"
        and sufficient_measurements
        and confidence_ok
    )
    if refused:
        tier = "withheld"
        text = "Consult a local agronomist."
    elif full:
        tier = "full"
        text = "Cited advice is available below."
    else:
        tier = "scouting"
        text = "Use scouting and cultural guidance only; retake the photo."

    raw = (entry or {}).get("guidance") or {}
    fired = list(rules_fired or [])
    if grade == "C":
        fired.append("evidence_grade_c_no_verified_entry")
    if confidence_cutoff is None:
        fired.append("confidence_cutoff_not_selected")
    if bg_check != "consistent":
        fired.append(f"background_sensitivity_{bg_check}")
    if status == "uncertain":
        fired.append("prediction_uncertain_retake_photo")
    advice = {
        "tier": tier,
        "message": text,
        "scouting": raw.get("scouting", []) if tier != "withheld" else [],
        "cultural": raw.get("cultural", []) if tier != "withheld" else [],
        "chemical_options": raw.get("chemical_options", []) if full else [],
        "label_reminder": raw.get("label_reminder", "Consult a local agronomist."),
        "weather_note": "risk unknown",
        "disclaimer": DISCLAIMER,
    }
    trace = {
        "tier": tier,
        "evidence_grade": grade,
        "kb_entry_id": (entry or {}).get("id"),
        "source_url": (entry or {}).get("source_url") or None,
        "retrieved_on": (entry or {}).get("retrieved_on") or None,
        "kb_version": kb_version,
        "rules_fired": fired,
        "bg_check": bg_check if bg_check in {"consistent", "inconsistent", "unknown"} else "unknown",
    }
    return advice, trace


class AdvisoryEngine:
    def __init__(
        self,
        kb_path: str | Path | None = None,
        weather_rules_path: str | Path | None = None,
        confused_pairs_path: str | Path | None = None,
    ):
        root = Path(__file__).resolve().parent.parent
        self.kb_path = Path(kb_path or root / "kb" / "advisories.yaml")
        self.weather_rules_path = Path(weather_rules_path or root / "kb" / "weather_rules.yaml")
        self.confused_pairs_path = Path(confused_pairs_path or root / "reports" / "confused_pairs.csv")
        self._entries: dict[str, dict[str, Any]] = {}
        self._confusable: dict[str, list[dict[str, Any]]] = {}
        self._last_mtime = 0.0
        self._load()

    @staticmethod
    def _make_key(crop: str, condition: str) -> str:
        return f"{crop.lower()}||{condition.lower()}"

    @property
    def kb_version(self) -> str:
        digest = hashlib.sha256()
        for path in (self.kb_path, self.weather_rules_path):
            if path.exists():
                digest.update(path.read_bytes())
        return f"{date.today().isoformat()}:{digest.hexdigest()}"

    def _load(self) -> None:
        if not self.kb_path.exists():
            self._entries = {}
            return
        data = yaml.safe_load(self.kb_path.read_text(encoding="utf-8")) or {}
        self._entries = {
            self._make_key(entry["crop"], entry["condition"]): entry
            for entry in data.get("advisories", [])
        }
        self._last_mtime = self.kb_path.stat().st_mtime
        self._load_confusable_pairs()
        log.info("Loaded %s advisory entries", len(self._entries))

    def _load_confusable_pairs(self) -> None:
        self._confusable = {}
        if not self.confused_pairs_path.is_file():
            return
        with self.confused_pairs_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                true_label = row.get("true_label") or row.get("actual") or row.get("true")
                predicted = row.get("predicted_label") or row.get("predicted") or row.get("prediction")
                if not true_label or not predicted:
                    continue
                self._confusable.setdefault(predicted, []).append({
                    "label": true_label,
                    "hint": f"May be confused with {true_label}.",
                    "how_to_tell_apart": None,
                })

    def get_entry(self, crop: str, condition: str) -> dict[str, Any] | None:
        if self.kb_path.exists() and self.kb_path.stat().st_mtime > self._last_mtime:
            self._load()
        return self._entries.get(self._make_key(crop, condition))

    def get_confusable_hints(self, class_name: str) -> list[dict[str, Any]]:
        return list(self._confusable.get(class_name, []))

    def get_advisory(self, crop: str, condition: str, **_weather: Any) -> dict[str, Any]:
        entry = self.get_entry(crop, condition)
        if not entry:
            return {"status": "PLACEHOLDER", "guidance": {}, "weather_note": "risk unknown"}
        return entry

    def get_advisory_from_class_name(self, class_name: str, **_weather: Any) -> dict[str, Any]:
        try:
            crop, condition = parse_class_name(class_name)
        except ValueError:
            return {}
        return self.get_advisory(crop, condition)

    def version_metadata(self) -> dict[str, Any]:
        return {"kb_version": self.kb_version, "entry_count": len(self._entries)}


def is_well_formed_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
