"""Copy observed seed ranges from the saved WP1 report into the gate YAML."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPREAD_SOURCES = {
    "within_crop_macro_f1": "within_crop_macro_f1",
    "minimum_class_recall": "min_recall",
    "bg_drop_per_image_mean": "bg_drop_per_image_mean",
    "ece": "ece",
}


def sync_spreads(seed_report: Path, thresholds_path: Path, *, write: bool = True) -> dict[str, Any]:
    """Populate each criterion's observed spread from its named report metric."""
    report = json.loads(seed_report.read_text(encoding="utf-8"))
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Seed variance report must contain a metrics object")

    config = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    criteria = config.get("criteria", {})
    updated = {}
    for criterion, metric_name in SPREAD_SOURCES.items():
        metric = metrics.get(metric_name)
        spread = metric.get("range") if isinstance(metric, dict) else None
        if not isinstance(spread, (int, float)):
            raise ValueError(f"Missing numeric metrics.{metric_name}.range in {seed_report}")
        try:
            report_source = seed_report.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            report_source = seed_report.name
        expected_source = f"{report_source}#/metrics/{metric_name}/range"
        row = criteria.get(criterion)
        if not isinstance(row, dict):
            raise ValueError(f"Missing threshold criterion {criterion!r}")
        row["observed_seed_spread"] = float(spread)
        row["observed_seed_spread_source"] = expected_source
        updated[criterion] = float(spread)

    if write:
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=thresholds_path.parent,
                prefix=f".{thresholds_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, thresholds_path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-report", default="reports/seed_variance.json")
    parser.add_argument("--thresholds", default="configs/gates/gate_thresholds.yaml")
    parser.add_argument("--check", action="store_true", help="Show values without writing")
    args = parser.parse_args()
    try:
        values = sync_spreads(
            PROJECT_ROOT / args.seed_report,
            PROJECT_ROOT / args.thresholds,
            write=not args.check,
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"seed_spreads": values, "written": not args.check}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
