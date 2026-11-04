from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inspect_registry import sanitized_snapshot
from scripts.snapshot_reports import _sanitize, _write_snapshot


def test_registry_snapshot_omits_host_paths_and_arbitrary_tags() -> None:
    source = {
        "model": "VedantPlantDiseaseClassifier",
        "tracking_uri": "D:\\private\\Vedant\\mlflow.db",
        "alias": {"name": "champion", "version": "2", "error": "private"},
        "versions": [{
            "version": "2",
            "run_id": "run-safe",
            "source": "D:\\private\\Vedant\\artifact",
            "tags": {"gate_verdict": "PASS", "local_path": "C:\\Users\\Vedant\\x"},
        }],
    }
    result = sanitized_snapshot(source)
    text = json.dumps(result)
    assert "Vedant" not in text
    assert "D:\\" not in text
    assert "Vedant" not in text
    assert "local_path" not in text
    assert result["versions"][0]["artifact_source_kind"] == "local_file_uri"


def test_snapshot_sanitizer_redacts_machine_paths_and_usernames() -> None:
    result = _sanitize({
        "path": "D:\\VIT\\project\\reports",
        "owner": "Vedant",
        "run_id": "abcdef0123456789abcdef0123456789",
    })
    assert "D:\\" not in str(result)
    assert "Vedant" not in str(result)
    assert "<redacted-local-path>" in result["path"]
    assert result["run_id"] == "<redacted-identifier>"


def test_snapshot_copy_enforces_size_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "snapshot.json"
    source.write_text(json.dumps({"value": "content"}), encoding="utf-8")
    with pytest.raises(ValueError, match="byte limit"):
        _write_snapshot(source, target, max_bytes=5)
    assert not target.exists()
