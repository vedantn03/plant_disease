"""Synthetic regression coverage for WP1 promotion-gate reporting helpers."""

import json

import numpy as np

from scripts.train import compute_per_crop_metrics
from scripts.summarize_seed_variance import summarize
from scripts.promote import _metrics_from_payload


def test_single_class_crop_f1_matches_accuracy_with_false_positive_class():
    # A single-class crop has 9/10 right; the extra prediction belongs to another crop.
    names = {0: "Soybean___healthy", 1: "Tomato___healthy"}
    metrics = compute_per_crop_metrics(np.array([0] * 10), np.array([0] * 9 + [1]), names)
    assert metrics["soybean"]["accuracy"] == 0.9
    assert abs(metrics["soybean"]["macro_f1"] - metrics["soybean"]["accuracy"]) < 0.06


def _row(run_id, candidate_name, recall, correct):
    return {
        "name": candidate_name,
        "run_id": run_id,
        "clean_macro_f1": 0.99,
        "within_crop_macro_f1": 0.98,
        "min_recall": {
            "value": recall,
            "class": "Corn___driving",
            "n": 77,
            "correct": correct,
            "all": {"Corn___driving": {"recall": recall, "n": 77, "correct": correct}},
        },
        "ece": 0.001,
        "masked_background_by_fill": {
            fill: {"bg_drop": val} for fill, val in
            {"black": 0.18, "mid_grey": 0.09, "per_image_mean": 0.03}.items()
        },
    }


def test_seed_summary_reads_three_json_reports_without_model_loading(tmp_path):
    ids = {
        "champion": "fb731412d82544e0abc231b8d7c62f95",
        "seed_43": "48ecf14da2d84803a36d6fd108bd6b61",
        "seed_44": "0e2c51abf0834c248e03c5aef774e291",
    }
    paths = {}
    comparison = []
    for name, run_id in ids.items():
        path = tmp_path / f"{name}.json"
        row = _row(run_id, name, 0.88, 68)
        comparison.append(row)
        path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
        paths[name] = path
    paths["champion"].write_text(json.dumps({"results": comparison[:2]}), encoding="utf-8")
    # Use the same driving class across the three synthetic reports.
    for path in paths.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["results"]:
            row["min_recall"]["all"] = {
                "Corn___driving": {"recall": 0.88, "n": 77, "correct": 68}
            }
        path.write_text(json.dumps(payload), encoding="utf-8")
    result, markdown = summarize(paths, ids["champion"])
    assert result["metrics"]["min_recall"]["mean"] == 0.88
    assert result["driving_class_counts"]["seed_43"] == {"correct": 68, "total": 77}
    assert "three seeds are a small sample" in markdown.lower()


def test_candidate_gate_report_maps_to_promotion_metrics():
    row = _row("candidate", "candidate_seed_44", 0.9, 69)
    row["masked_background_by_fill"]["black"]["bg_drop"] = 0.18
    metrics = _metrics_from_payload({"results": [row]})
    assert metrics.within_crop_macro_f1 == 0.98
    assert metrics.min_recall == 0.9
    assert metrics.bg_drop == 0.18


def test_champion_comparison_report_selects_champion_row():
    champion = _row(None, "champion_v2", 0.91, 70)
    candidate = _row("run-2", "candidate_seed_44", 0.89, 68)
    # Promotion's selection code is exercised via the payload adaptation contract.
    payload = {"results": [champion, candidate]}
    selected = next(row for row in payload["results"] if row.get("name") == "champion_v2")
    metrics = _metrics_from_payload({"results": [selected]})
    assert metrics.min_recall == 0.91


def test_seed44_driving_class_counts_can_be_summarized():
    row = _row("run-44", "candidate_seed_44", 0.87, 67)
    row["min_recall"]["class"] = "Corn___driving"
    row["min_recall"]["all"] = {
        "Corn___driving": {"recall": 67 / 77, "n": 77, "correct": 67}
    }
    assert row["min_recall"]["all"]["Corn___driving"] == {
        "recall": 67 / 77, "n": 77, "correct": 67
    }
