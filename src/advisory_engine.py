"""
advisory_engine.py
==================
Rule-based crop-protection advisory engine for the plant disease pipeline.

Loads kb/advisories.yaml and kb/weather_rules.yaml, returning a structured,
auditable advisory for a given (crop, condition) pair.

Design principles:
  - Scope: CROP-PROTECTION only. Fertilizer recommendations from leaf images are PROHIBITED.
  - Advisories come ONLY from cited university extension or regulator trials (never model-generated).
  - Weather inputs add ONLY verified epidemiological context notes; never change chemical doses.
  - Viral, HLB, and pest conditions explicitly exclude synthetic fungicide advice.
  - Every response includes region disclosure and the legal advisory disclaimer.
"""

from pathlib import Path
from typing import Optional
import yaml

from src.utils import get_logger, parse_class_name

log = get_logger("advisory")

DISCLAIMER = (
    "DISCLAIMER: This advisory is for demonstration and decision-support purposes only and is NOT a "
    "substitute for professional agronomic advice. Always consult a licensed agronomist, "
    "local agricultural extension office, or official registered product labels before applying any "
    "treatment. Dosages and active ingredients vary by country, state, and specific label registration. "
    "This tool does not diagnose field conditions."
)

FUNGICIDE_KEYWORDS = {
    "fungicide", "mancozeb", "chlorothalonil", "copper", "azoxystrobin",
    "propiconazole", "tebuconazole", "metalaxyl", "cymoxanil", "difenoconazole",
}


class AdvisoryEngine:
    """
    Loads YAML knowledge bases and serves structured agronomic advisories.
    """

    def __init__(
        self,
        kb_path: str | Path | None = None,
        weather_rules_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        if kb_path is None:
            kb_path = root / "kb" / "advisories.yaml"
        if weather_rules_path is None:
            weather_rules_path = root / "kb" / "weather_rules.yaml"

        self.kb_path = Path(kb_path)
        self.weather_rules_path = Path(weather_rules_path)
        self._entries: dict[str, dict] = {}
        self._weather_rules: list[dict] = []
        self._last_mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        if self.kb_path.exists():
            try:
                self._last_mtime = self.kb_path.stat().st_mtime
            except Exception:
                pass
            with open(self.kb_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for entry in data.get("advisories", []):
                key = self._make_key(entry["crop"], entry["condition"])
                self._entries[key] = entry
            log.info(f"Loaded {len(self._entries)} advisory entries from {self.kb_path}")

        if self.weather_rules_path.exists():
            with open(self.weather_rules_path, "r", encoding="utf-8") as f:
                w_data = yaml.safe_load(f)
            self._weather_rules = w_data.get("rules", [])
            log.info(f"Loaded {len(self._weather_rules)} weather risk rules from {self.weather_rules_path}")

    def _make_key(self, crop: str, condition: str) -> str:
        return f"{crop.lower()}||{condition.lower()}"

    def get_advisory(
        self,
        crop: str,
        condition: str,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
    ) -> dict:
        if self.kb_path.exists():
            try:
                current_mtime = self.kb_path.stat().st_mtime
                if current_mtime > self._last_mtime:
                    self._load()
            except Exception:
                pass

        key = self._make_key(crop, condition)
        entry = self._entries.get(key)

        if entry is None:
            return self._not_found(crop, condition)

        region = entry.get("region", "US-Extension / International")
        result = {
            "crop": entry.get("crop"),
            "condition": entry.get("condition"),
            "pathogen_type": entry.get("pathogen_type", "unknown"),
            "status": entry.get("status", "PLACEHOLDER"),
            "region": region,
            "region_caveat": f"Source region: {region}; verify local regulatory registrations, label rates, and pre-harvest intervals (PHI) before use.",
            "recommendation_text": entry.get("recommendation_text", ""),
            "caution_text": entry.get("caution_text", ""),
            "cultural_controls": entry.get("cultural_controls", []),
            "chemical_options": entry.get("chemical_options", []),
            "weather_note": "",
            "weather_urgency": "low",
            "source_name": entry.get("source_name", ""),
            "source_url": entry.get("source_url", ""),
            "last_verified": entry.get("last_verified", ""),
            "disclaimer": DISCLAIMER,
        }

        # Check weather rules
        if temperature_c is not None and humidity_pct is not None:
            # 1. Check entry-specific weather rule
            if entry.get("weather_rule"):
                w_rule = entry["weather_rule"]
                if "source" in w_rule or "source_id" in w_rule:
                    if (
                        w_rule.get("humidity_min", 0) <= humidity_pct
                        and w_rule.get("temp_min", -100) <= temperature_c <= w_rule.get("temp_max", 100)
                    ):
                        result["weather_note"] = w_rule.get("note", "")
                        result["weather_urgency"] = w_rule.get("urgency", "elevated")

            # 2. Check global weather rules from weather_rules.yaml
            if not result["weather_note"]:
                for rule in self._weather_rules:
                    if (
                        rule.get("crop", "").lower() == crop.lower()
                        and rule.get("condition", "").lower() == condition.lower()
                    ):
                        if (
                            humidity_pct >= rule.get("humidity_min", 0)
                            and rule.get("temp_min", -100) <= temperature_c <= rule.get("temp_max", 100)
                        ):
                            result["weather_note"] = rule.get("note", "")
                            result["weather_urgency"] = rule.get("urgency", "elevated")
                            break

        return result

    def _not_found(self, crop: str, condition: str) -> dict:
        return {
            "crop": crop,
            "condition": condition,
            "pathogen_type": "unknown",
            "status": "PLACEHOLDER",
            "region": "global",
            "region_caveat": "Consult a certified local agronomist or extension office.",
            "recommendation_text": (
                "No verified agronomic recommendation is currently recorded for this condition. "
                "Consult local extension publications."
            ),
            "caution_text": "",
            "cultural_controls": [],
            "chemical_options": [],
            "weather_note": "",
            "weather_urgency": "low",
            "source_name": "",
            "source_url": "",
            "last_verified": "",
            "disclaimer": DISCLAIMER,
        }

    def get_advisory_from_class_name(
        self,
        class_name: str,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
    ) -> dict:
        try:
            crop, condition = parse_class_name(class_name)
        except ValueError:
            return self._not_found(class_name, "unknown")
        return self.get_advisory(crop, condition, temperature_c, humidity_pct)
