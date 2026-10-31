"""Dry-run promotion never moves @champion while thresholds are provisional."""

import json

from src.gates import GateMetrics, evaluate_gate, load_gate_thresholds
from src.registry import append_promotion_log, promote_alias


class _FakeClient:
    def __init__(self) -> None:
        self.aliases: dict[tuple[str, str], str] = {}

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[(name, alias)] = str(version)


def test_dry_run_does_not_call_set_alias(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr("src.registry.get_client", lambda: fake)
    # promote_alias with dry_run True must not touch the fake
    result = promote_alias("3", alias="champion", dry_run=True, client=fake)
    assert result["moved"] is False
    assert result["dry_run"] is True
    assert fake.aliases == {}


def test_apply_would_set_alias_on_fake_client():
    fake = _FakeClient()
    result = promote_alias("3", alias="champion", dry_run=False, client=fake)
    assert result["moved"] is True
    assert list(fake.aliases.values()) == ["3"]


def test_provisional_gate_blocks_live_promotion_policy():
    thresholds = load_gate_thresholds()
    verdict = evaluate_gate(
        GateMetrics(macro_f1=0.99, min_recall=0.9, bg_drop=0.04, ece=0.002),
        None,
        thresholds,
    )
    assert verdict["any_provisional"] is True
    # scripts/promote.py --apply exits 2 when this flag is set


def test_append_promotion_log_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "promotion_log.jsonl"
    monkeypatch.setattr("src.registry.PROMOTION_LOG", log_path)
    append_promotion_log({"action": "promote", "dry_run": True, "verdict": {"verdict": "PASS"}})
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["dry_run"] is True
    assert "timestamp" in rec
