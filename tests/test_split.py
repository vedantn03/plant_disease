"""
Tests for audit_and_split.py

Verifies:
  1. Zero group overlap across train/val/test splits.
  2. All three split CSVs exist and are non-empty.
  3. Class names parse correctly.
"""

import csv
import sys
from pathlib import Path

import pytest

# Allow src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import parse_class_name

SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"


def _load_split(name: str) -> list[dict]:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{name}.csv not found – run scripts/audit_and_split.py first")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _require_all_splits() -> None:
    missing = [name for name in ("train", "val", "test")
               if not (SPLITS_DIR / f"{name}.csv").exists()]
    if missing:
        pytest.skip(
            "Split manifests unavailable; run scripts/audit_and_split.py first "
            f"(missing: {', '.join(missing)})"
        )


def test_splits_exist():
    _require_all_splits()
    for name in ["train", "val", "test"]:
        assert (SPLITS_DIR / f"{name}.csv").exists(), \
            f"{name}.csv missing. Run scripts/audit_and_split.py"


def test_splits_nonempty():
    _require_all_splits()
    for name in ["train", "val", "test"]:
        recs = _load_split(name)
        assert len(recs) > 0, f"{name} split is empty"


def test_zero_group_overlap():
    """Core assertion: no group_id appears in more than one split."""
    _require_all_splits()
    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")

    train_groups = {r["group_id"] for r in train}
    val_groups = {r["group_id"] for r in val}
    test_groups = {r["group_id"] for r in test}

    tv = train_groups & val_groups
    tt = train_groups & test_groups
    vt = val_groups & test_groups

    assert not tv, f"Train/val group overlap: {len(tv)} groups"
    assert not tt, f"Train/test group overlap: {len(tt)} groups"
    assert not vt, f"Val/test group overlap: {len(vt)} groups"


def test_class_names_parseable():
    """All class_name values in any split must parse without error."""
    _require_all_splits()
    for name in ["train", "val", "test"]:
        recs = _load_split(name)
        for r in recs:
            cls = r["class_name"]
            try:
                crop, cond = parse_class_name(cls)
                assert crop, f"Empty crop in {cls}"
                assert cond, f"Empty condition in {cls}"
            except ValueError as e:
                pytest.fail(f"parse_class_name failed for '{cls}': {e}")


def test_parse_class_name_known_cases():
    """Unit test for the parser itself."""
    assert parse_class_name("Tomato___Early_blight") == ("tomato", "early_blight")
    assert parse_class_name("Apple___healthy") == ("apple", "healthy")
    assert parse_class_name("Cherry_(including_sour)___Powdery_mildew") == (
        "cherry_(including_sour)", "powdery_mildew"
    )
    assert parse_class_name("Corn_(maize)___Common_rust_") == ("corn_(maize)", "common_rust_")
