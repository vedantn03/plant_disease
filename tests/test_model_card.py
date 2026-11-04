from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gen_model_card import generate


def test_model_card_uses_snapshots_and_marks_missing_inputs(tmp_path: Path) -> None:
    (tmp_path / "eval_summary.json").write_text(
        json.dumps({"generated_at": "2026-11-04", "macro_f1": 0.91}), encoding="utf-8"
    )
    card, datasheet = generate(tmp_path)
    assert "Source date: 2026-11-04" in card
    assert "`macro_f1`: `0.91`" in card
    assert "Not generated: missing `reports/snapshot/robustness_three_fill.json`" in card
    assert "Not generated: missing `reports/snapshot/seed_variance.json`" in datasheet
    assert "license and redistribution rights must be verified" in card


def test_model_card_rejects_absolute_paths_and_usernames(tmp_path: Path) -> None:
    (tmp_path / "registry_export.json").write_text(
        json.dumps({"artifact": "D:\\private\\Vedant\\model.pth"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="absolute Windows path or username"):
        generate(tmp_path)


def test_model_card_includes_corn_weak_class_only_from_evaluation_snapshot(
    tmp_path: Path,
) -> None:
    (tmp_path / "eval_summary.json").write_text(
        json.dumps({
            "date": "2026-11-04",
            "weakest_classes": [{
                "class": "Corn Cercospora Gray leaf spot",
                "recall": 0.75,
            }],
        }),
        encoding="utf-8",
    )
    card, _ = generate(tmp_path)
    assert "Corn weak-class evidence" in card
    assert "Corn Cercospora Gray leaf spot" in card
    assert '"recall": 0.75' in card
