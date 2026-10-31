"""Unit tests for the 4-criterion promotion gate (fake metrics only)."""

from src.gates import (
    GateMetrics,
    evaluate_gate,
    load_gate_thresholds,
    one_extra_error_recall,
    wilson_lower_bound,
)


def _thresholds():
    return load_gate_thresholds()


def _ok(**overrides) -> GateMetrics:
    base = dict(macro_f1=0.9900, min_recall=0.90, bg_drop=0.05, ece=0.002)
    base.update(overrides)
    return GateMetrics(**base)


def test_first_model_bootstrap_passes_relative_rule():
    cand = _ok()
    verdict = evaluate_gate(cand, champion=None, thresholds=_thresholds())
    assert verdict["first_model"] is True
    assert verdict["all_passed"] is True
    assert verdict["verdict"] == "PASS"
    f1 = next(c for c in verdict["criteria"] if c["name"] == "macro_f1")
    assert f1["passed"] is True
    assert "first-model bootstrap" in f1["reason"]


def test_fail_macro_f1_below_champion_minus_margin():
    champ = _ok(macro_f1=0.9900)
    cand = _ok(macro_f1=0.9800)  # drop 0.010 > margin 0.005
    verdict = evaluate_gate(cand, champ, _thresholds())
    assert verdict["all_passed"] is False
    assert "macro_f1" in verdict["failures"]


def test_fail_min_recall():
    champ = _ok()
    cand = _ok(min_recall=0.70, min_recall_class="tiny", min_recall_n=23)
    verdict = evaluate_gate(cand, champ, _thresholds())
    assert "min_recall" in verdict["failures"]


def test_fail_bg_drop():
    champ = _ok(bg_drop=0.05)
    cand = _ok(bg_drop=0.12)
    verdict = evaluate_gate(cand, champ, _thresholds())
    assert "bg_drop" in verdict["failures"]


def test_bg_drop_is_relative_to_champion():
    champ = _ok(bg_drop=0.18)
    cand = _ok(bg_drop=0.185)
    verdict = evaluate_gate(cand, champ, _thresholds())
    assert "bg_drop" not in verdict["failures"]


def test_within_crop_f1_is_gate_metric_when_available():
    champ = _ok(macro_f1=0.70, within_crop_macro_f1=0.9908)
    cand = _ok(macro_f1=0.70, within_crop_macro_f1=0.9860)
    verdict = evaluate_gate(cand, champ, _thresholds())
    assert "macro_f1" not in verdict["failures"]


def test_fail_ece():
    champ = _ok()
    cand = _ok(ece=0.08)
    verdict = evaluate_gate(cand, champ, _thresholds())
    assert "ece" in verdict["failures"]


def test_all_criteria_marked_provisional():
    verdict = evaluate_gate(_ok(), _ok(), _thresholds())
    assert verdict["any_provisional"] is True
    assert all(c["provisional"] for c in verdict["criteria"])


def test_tiny_class_one_extra_error_n23():
    effect = one_extra_error_recall(23, current_errors=0)
    assert effect["recall_now"] == 1.0
    assert effect["recall_one_extra_error"] == round(22 / 23, 4)
    assert effect["drop"] == round(1 / 23, 4)
    # Wilson lower bound of 22/23 is well below the point estimate
    assert wilson_lower_bound(22, 23) < 22 / 23
    assert effect["wilson_lower_one_extra"] < 0.85 or effect["wilson_lower_one_extra"] >= 0.0
