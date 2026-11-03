"""SIMULATION ONLY: summarize an existing legacy simulation drift artifact.

This script never creates synthetic observations and never starts retraining.
Use scripts/drift_exporter.py for real request-log monitoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.utils import get_logger

log = get_logger("drift_trigger_simulation")


def build_simulation_summary(
    drift_metrics: dict[str, Any], *, psi_threshold: float, ks_alpha: float
) -> dict[str, Any]:
    violations = []
    for feature, metric in drift_metrics.items():
        psi = float(metric.get("psi", 0.0))
        p_value = float(metric.get("p_value", metric.get("ks_pvalue", 1.0)))
        if psi > psi_threshold and p_value < ks_alpha:
            violations.append({"feature": feature, "psi": psi, "ks_p_value": p_value})
    detected = bool(violations)
    return {
        "classification": "SIMULATION",
        "drift_detected": detected,
        "action_required": "SIMULATION_DRIFT_DETECTED" if detected else "SIMULATION_NO_DRIFT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "violations": violations,
        "automated_retraining": False,
        "thresholds_provisional": True,
        "notes": "SIMULATION output only; this artifact does not trigger retraining or registry changes.",
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml", require_data_root=False)
    input_path = PROJECT_ROOT / "reports" / "drift_metrics.json"
    if not input_path.exists():
        message = (
            "SIMULATION ONLY: legacy reports/drift_metrics.json is missing. "
            "No synthetic data was generated. Use scripts/drift_exporter.py for real logs."
        )
        log.error(message)
        print(message, file=sys.stderr)
        return 2

    drift_metrics = json.loads(input_path.read_text(encoding="utf-8"))
    summary = build_simulation_summary(
        drift_metrics,
        psi_threshold=float(cfg.drift.psi_threshold),
        ks_alpha=float(cfg.drift.ks_alpha),
    )
    output_path = PROJECT_ROOT / "reports" / "retraining_trigger.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("SIMULATION summary written to %s", output_path)
    print("SIMULATION ONLY")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
