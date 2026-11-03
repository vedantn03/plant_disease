"""Drift math and exporter checks use synthetic arrays/log records only."""

import json
import numpy as np
from prometheus_client import generate_latest

from scripts.compute_baseline_features import build_baseline_payload
from scripts.drift_exporter import (
    compute_psi,
    create_metrics,
    evaluate_window,
    read_log_window,
    update_metrics,
)


def _records(values, source="api"):
    return [
        {
            "brightness": float(value),
            "blur_score": float(value),
            "confidence": float(value),
            "traffic_source": source,
        }
        for value in values
    ]


def _baseline(values):
    values = [float(value) for value in values]
    payload = build_baseline_payload(
        {"brightness": values, "blur": values, "confidence": values},
        requested_sample_size=len(values),
        seed=42,
        dvc_data_hash="synthetic-test-hash",
        psi_bins=4,
        psi_epsilon=1e-6,
        checkpoint_md5="synthetic-test-checkpoint-hash",
    )
    return payload


def test_psi_and_ks_detect_shift_for_synthetic_arrays():
    baseline = _baseline(np.linspace(0.0, 1.0, 100).tolist())
    records = _records(np.linspace(8.0, 9.0, 100).tolist(), source="replay")

    psi = compute_psi(
        baseline["features"]["brightness"]["bin_counts"],
        [record["brightness"] for record in records],
        baseline["features"]["brightness"]["quantile_bin_edges"],
        1e-6,
    )
    report = evaluate_window(
        baseline,
        records,
        minimum_sample_count=30,
        psi_threshold=0.2,
        ks_alpha=0.05,
        psi_epsilon=1e-6,
        drift_rule="psi_and_ks",
        thresholds_provisional=True,
    )

    assert psi > 0.2
    assert report["features"]["brightness"]["ks_p_value"] < 0.05
    assert report["drift_flag"] == 1
    assert report["replay_share"] == 1.0


def test_insufficient_window_has_no_drift_verdict():
    report = evaluate_window(
        _baseline([0.0, 1.0, 2.0, 3.0]),
        _records([1.0, 2.0]),
        minimum_sample_count=3,
        psi_threshold=0.2,
        ks_alpha=0.05,
        psi_epsilon=1e-6,
        drift_rule="psi_and_ks",
        thresholds_provisional=True,
    )

    assert report["status"] == "insufficient_data"
    assert report["drift_flag"] is None
    assert report["features"] == {}


def test_log_window_keeps_only_latest_valid_records(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _records(range(5))),
        encoding="utf-8",
    )

    records = read_log_window(log_path, window_size=3)

    assert [record["brightness"] for record in records] == [2.0, 3.0, 4.0]


def test_baseline_file_schema_contains_required_metadata_and_distributions(tmp_path):
    baseline = _baseline([1.0, 2.0, 3.0, 4.0])
    path = tmp_path / "baseline_features.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["kind"] == "train_split_feature_baseline"
    assert loaded["sample_size"] == 4
    assert loaded["seed"] == 42
    assert loaded["dvc_data_hash"]
    for feature in ("brightness", "blur", "confidence"):
        assert loaded["features"][feature]["summary"]
        assert "quantile_bin_edges" in loaded["features"][feature]
        assert "bin_counts" in loaded["features"][feature]
        assert "reference_values" in loaded["features"][feature]


def test_exporter_metrics_render_only_observed_window_values():
    baseline = _baseline(np.linspace(0.0, 1.0, 40).tolist())
    report = evaluate_window(
        baseline,
        _records(np.linspace(4.0, 5.0, 40).tolist(), source="replay"),
        minimum_sample_count=30,
        psi_threshold=0.2,
        ks_alpha=0.05,
        psi_epsilon=1e-6,
        drift_rule="psi_and_ks",
        thresholds_provisional=True,
    )
    metrics = create_metrics()
    update_metrics(metrics, report)
    rendered = generate_latest(metrics["registry"]).decode("utf-8")

    for metric in (
        "drift_psi",
        "drift_ks_p_value",
        "drift_flag",
        "drift_window_size",
        "drift_replay_share",
    ):
        assert f"# HELP {metric} " in rendered
    assert 'drift_flag{state="current"} 1.0' in rendered

    update_metrics(metrics, {**report, "status": "insufficient_data", "drift_flag": None})
    insufficient_render = generate_latest(metrics["registry"]).decode("utf-8")
    assert 'drift_flag{state="current"}' not in insufficient_render
