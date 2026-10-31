"""Synthetic, artifact-free tests for three-fill report helpers."""

from scripts.three_fill_report import summarize_mask_areas


def test_summarize_mask_areas_is_deterministic():
    result = summarize_mask_areas([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["mean"] == 0.3
    assert result["median"] == 0.3
    assert result["p5"] == 0.12
    assert result["p95"] == 0.48


def test_summarize_mask_areas_empty_is_explicit():
    assert summarize_mask_areas([]) == {
        "mean": 0.0,
        "median": 0.0,
        "p5": 0.0,
        "p95": 0.0,
    }
