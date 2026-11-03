"""Synthetic unit tests for VAL promotion gates and threshold approval."""

import hashlib

import yaml

from src.gates import (
    GateMetrics,
    evaluate_gate,
    load_gate_thresholds,
    one_extra_error_recall,
    thresholds_sha256,
    verify_threshold_approval,
    wilson_lower_bound,
)


def _thresholds():
    return load_gate_thresholds()


def _ok(**overrides) -> GateMetrics:
    base = dict(
        macro_f1=0.9900,
        within_crop_macro_f1=0.9900,
        min_recall=0.90,
        bg_drop=0.05,
        bg_drop_per_image_mean=0.05,
        ece=0.002,
        per_class_recall={"class_a": {"n": 40, "correct": 36, "recall": 0.90}},
    )
    base.update(overrides)
    return GateMetrics(**base)


def _criterion(verdict, name):
    return next(item for item in verdict["criteria"] if item["name"] == name)


def test_first_model_bootstrap_passes_relative_rule_but_remains_provisional():
    verdict = evaluate_gate(_ok(), champion=None, thresholds=_thresholds())
    assert verdict["first_model"] is True
    assert verdict["all_passed"] is True
    assert verdict["verdict"] == "PASS"
    assert _criterion(verdict, "within_crop_macro_f1")["passed"] is True
    assert verdict["any_provisional"] is True


def test_within_crop_f1_uses_champion_minus_margin():
    champion = _ok(within_crop_macro_f1=0.9900)
    candidate = _ok(within_crop_macro_f1=0.9849)
    verdict = evaluate_gate(candidate, champion, _thresholds(), approved=True)
    assert "within_crop_macro_f1" in verdict["failures"]


def test_mean_fill_drop_is_only_background_gate():
    champion = _ok(bg_drop=0.10, bg_drop_per_image_mean=0.02)
    candidate = _ok(bg_drop=0.90, bg_drop_per_image_mean=0.05)
    verdict = evaluate_gate(candidate, champion, _thresholds(), approved=True)
    assert _criterion(verdict, "bg_drop_per_image_mean")["passed"] is True
    candidate.bg_drop_per_image_mean = 0.0501
    verdict = evaluate_gate(candidate, champion, _thresholds(), approved=True)
    assert _criterion(verdict, "bg_drop_per_image_mean")["passed"] is False


def test_tiny_class_uses_wilson_lower_bound_for_n_23():
    candidate = _ok(per_class_recall={
        "tiny_class": {"n": 23, "correct": 22, "recall": 22 / 23},
        "common_class": {"n": 40, "correct": 36, "recall": 0.90},
    })
    verdict = evaluate_gate(candidate, _ok(), _thresholds(), approved=True)
    limiting = _criterion(verdict, "minimum_class_recall")
    assert "wilson_95_used=True" in limiting["reason"]
    assert limiting["observed"] == wilson_lower_bound(22, 23, z=1.96)
    assert limiting["passed"] is False


def test_ece_ceiling_is_applied():
    verdict = evaluate_gate(_ok(ece=0.0101), _ok(), _thresholds(), approved=True)
    assert "ece" in verdict["failures"]


def test_threshold_approval_requires_exact_current_hash(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("schema_version: 1\ncriteria: {}\n", encoding="utf-8")
    approval = tmp_path / "approval.yaml"
    digest = hashlib.sha256(thresholds.read_bytes()).hexdigest()
    approval.write_text(yaml.safe_dump({
        "approved": True,
        "approved_by": "course-reviewer",
        "approved_on": "2026-11-03",
        "thresholds_sha256": digest,
    }), encoding="utf-8")

    assert verify_threshold_approval(approval, thresholds)[0] is True
    thresholds.write_text("schema_version: 1\ncriteria: {changed: true}\n", encoding="utf-8")
    valid, reason, current_hash = verify_threshold_approval(approval, thresholds)
    assert valid is False
    assert "does not match" in reason
    assert current_hash == thresholds_sha256(thresholds)


def test_unapproved_threshold_file_refuses_apply(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("schema_version: 1\n", encoding="utf-8")
    approval = tmp_path / "approval.yaml"
    approval.write_text(
        "approved: false\napproved_by: ''\napproved_on: ''\nthresholds_sha256: ''\n"
    )
    valid, reason, _ = verify_threshold_approval(approval, thresholds)
    assert valid is False
    assert "approved: true" in reason


def test_apply_requires_committed_approval_and_threshold_files(tmp_path, monkeypatch):
    import src.gates as gates

    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("schema_version: 1\n", encoding="utf-8")
    approval = tmp_path / "approval.yaml"
    digest = hashlib.sha256(thresholds.read_bytes()).hexdigest()
    approval.write_text(yaml.safe_dump({
        "approved": True,
        "approved_by": "reviewer",
        "approved_on": "2026-11-03",
        "thresholds_sha256": digest,
    }), encoding="utf-8")

    class _Result:
        returncode = 0

    monkeypatch.setattr(gates.subprocess, "run", lambda *_args, **_kwargs: _Result())
    valid, _, _ = verify_threshold_approval(
        approval, thresholds, require_committed=True, project_root=tmp_path
    )
    assert valid is True

    class _UntrackedResult:
        returncode = 1

    monkeypatch.setattr(
        gates.subprocess, "run", lambda *_args, **_kwargs: _UntrackedResult()
    )
    valid, reason, _ = verify_threshold_approval(
        approval, thresholds, require_committed=True, project_root=tmp_path
    )
    assert valid is False
    assert "tracked and committed" in reason

    calls = 0

    def dirty_after_tracking(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        result = type("Result", (), {})()
        result.returncode = 0 if calls % 2 else 1
        return result

    monkeypatch.setattr(gates.subprocess, "run", dirty_after_tracking)
    valid, reason, _ = verify_threshold_approval(
        approval, thresholds, require_committed=True, project_root=tmp_path
    )
    assert valid is False
    assert "uncommitted changes" in reason


def test_tiny_class_one_extra_error_n23():
    effect = one_extra_error_recall(23, current_errors=0)
    assert effect["recall_now"] == 1.0
    assert effect["recall_one_extra_error"] == round(22 / 23, 4)
    assert effect["drop"] == round(1 / 23, 4)
    assert effect["wilson_lower_one_extra"] < 22 / 23
