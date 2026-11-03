"""Export sliding-window feature drift statistics from real API request logs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus_client import CollectorRegistry, Gauge, start_http_server

from src.config_loader import load_config

FEATURE_LOG_FIELDS = {
    "brightness": "brightness",
    "blur": "blur_score",
    "confidence": "confidence",
}
LOG = logging.getLogger("drift_exporter")


def read_log_window(log_path: Path, window_size: int) -> list[dict[str, Any]]:
    """Return the most recent valid records, bounded by configured window size."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    records: deque[dict[str, Any]] = deque(maxlen=window_size)
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            try:
                feature_values = [float(record[FEATURE_LOG_FIELDS[name]]) for name in FEATURE_LOG_FIELDS]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in feature_values):
                continue
            records.append(record)
    return list(records)


def compute_psi(
    reference_counts: list[int],
    current_values: list[float],
    quantile_bin_edges: list[float],
    epsilon: float,
) -> float:
    """Compute PSI using baseline quantile cuts and inclusive tail buckets."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    edges = [-np.inf, *quantile_bin_edges, np.inf]
    current_counts, _ = np.histogram(np.asarray(current_values, dtype=np.float64), bins=edges)
    if len(reference_counts) != len(current_counts):
        raise ValueError("Baseline count bins do not match the quantile bin edges")
    ref = np.asarray(reference_counts, dtype=np.float64)
    cur = current_counts.astype(np.float64)
    ref_pct = (ref + epsilon) / (ref.sum() + len(ref) * epsilon)
    cur_pct = (cur + epsilon) / (cur.sum() + len(cur) * epsilon)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def evaluate_window(
    baseline: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    minimum_sample_count: int,
    psi_threshold: float,
    ks_alpha: float,
    psi_epsilon: float,
    drift_rule: str,
    thresholds_provisional: bool,
) -> dict[str, Any]:
    """Compare a request window to the stored TRAIN baseline, without fake series."""
    feature_records = [
        record
        for record in records
        if all(
            isinstance(record.get(FEATURE_LOG_FIELDS[name]), (int, float))
            and math.isfinite(float(record[FEATURE_LOG_FIELDS[name]]))
            for name in FEATURE_LOG_FIELDS
        )
    ]
    n = len(feature_records)
    replay_count = sum(record.get("traffic_source", "api") == "replay" for record in feature_records)
    replay_share = replay_count / n if n else None
    result: dict[str, Any] = {
        "status": "insufficient_data" if n < minimum_sample_count else "evaluated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_size": n,
        "minimum_sample_count": minimum_sample_count,
        "replay_share": replay_share,
        "thresholds_provisional": thresholds_provisional,
        "drift_flag": None,
        "features": {},
    }
    if n < minimum_sample_count:
        result["reason"] = "insufficient data for drift verdict"
        return result

    if drift_rule != "psi_and_ks":
        raise ValueError(f"Unsupported drift rule: {drift_rule}")

    feature_flags = []
    for feature, field in FEATURE_LOG_FIELDS.items():
        try:
            baseline_feature = baseline["features"][feature]
            reference = np.asarray(baseline_feature["reference_values"], dtype=np.float64)
            current = np.asarray([record[field] for record in feature_records], dtype=np.float64)
            ks_result = ks_2samp(reference, current)
            psi = compute_psi(
                baseline_feature["bin_counts"],
                current.tolist(),
                baseline_feature["quantile_bin_edges"],
                psi_epsilon,
            )
            flagged = psi > psi_threshold and float(ks_result.pvalue) < ks_alpha
            feature_flags.append(flagged)
            result["features"][feature] = {
                "psi": psi,
                "ks_p_value": float(ks_result.pvalue),
                "sample_count": n,
                "drift_flag": flagged,
            }
        except KeyError as exc:
            raise ValueError(f"Baseline is missing feature data: {exc}") from exc

    result["drift_flag"] = int(any(feature_flags))
    result["rule"] = "any feature: PSI > configured threshold AND KS p-value < ks_alpha"
    result["psi_threshold"] = psi_threshold
    result["ks_alpha"] = ks_alpha
    result["traffic_source_note"] = "Replay traffic is staged traffic, not real users."
    return result


def create_metrics(registry: CollectorRegistry | None = None) -> dict[str, Any]:
    """Create isolated Prometheus collectors, suitable for process or unit tests."""
    registry = registry or CollectorRegistry()
    return {
        "registry": registry,
        "psi": Gauge("drift_psi", "Population Stability Index by feature", ["feature"], registry=registry),
        "ks_p_value": Gauge(
            "drift_ks_p_value", "KS two-sample p-value by feature", ["feature"], registry=registry
        ),
        "flag": Gauge(
            "drift_flag", "Provisional feature drift verdict", ["state"], registry=registry
        ),
        "window_size": Gauge("drift_window_size", "Valid request count in sliding window", registry=registry),
        "replay_share": Gauge(
            "drift_replay_share", "Fraction of window requests marked as replay", registry=registry
        ),
    }


def update_metrics(metrics: dict[str, Any], report: dict[str, Any]) -> None:
    """Publish observations and remove verdict/statistic series when unavailable."""
    metrics["window_size"].set(report["window_size"])
    if report["replay_share"] is not None:
        metrics["replay_share"].set(report["replay_share"])
    else:
        metrics["replay_share"].clear()
    metrics["psi"].clear()
    metrics["ks_p_value"].clear()
    metrics["flag"].clear()
    if report["status"] != "evaluated":
        return
    for feature, values in report["features"].items():
        metrics["psi"].labels(feature=feature).set(values["psi"])
        metrics["ks_p_value"].labels(feature=feature).set(values["ks_p_value"])
    metrics["flag"].labels(state="current").set(report["drift_flag"])


def collect_once(
    project_root: Path,
    cfg: Any,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    drift_cfg = cfg.drift
    log_path = project_root / str(drift_cfg.request_log_path)
    baseline_path = project_root / str(drift_cfg.baseline_path)
    records = read_log_window(log_path, int(drift_cfg.window_size))
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report = evaluate_window(
            baseline,
            records,
            minimum_sample_count=int(drift_cfg.minimum_sample_count),
            psi_threshold=float(drift_cfg.psi_threshold),
            ks_alpha=float(drift_cfg.ks_alpha),
            psi_epsilon=float(drift_cfg.psi_epsilon),
            drift_rule=str(drift_cfg.drift_rule),
            thresholds_provisional=bool(drift_cfg.thresholds_provisional),
        )
    else:
        report = {
            "status": "baseline_missing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window_size": len(records),
            "minimum_sample_count": int(drift_cfg.minimum_sample_count),
            "replay_share": None,
            "thresholds_provisional": bool(drift_cfg.thresholds_provisional),
            "drift_flag": None,
            "reason": f"Baseline file not found: {baseline_path}",
        }
    update_metrics(metrics, report)
    report_path = project_root / str(drift_cfg.latest_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Evaluate once without starting HTTP")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml", require_data_root=False)
    metrics = create_metrics()
    if args.once:
        print(json.dumps(collect_once(PROJECT_ROOT, cfg, metrics), indent=2))
        return 0

    port = int(cfg.drift.exporter_port)
    start_http_server(port, registry=metrics["registry"])
    LOG.info("Drift exporter listening on port %s", port)
    try:
        while True:
            try:
                report = collect_once(PROJECT_ROOT, cfg, metrics)
                LOG.info("Drift evaluation status=%s window_size=%s", report["status"], report["window_size"])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                LOG.exception("Drift evaluation failed: %s", exc)
            time.sleep(int(cfg.drift.poll_interval_seconds))
    except KeyboardInterrupt:
        LOG.info("Drift exporter stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
