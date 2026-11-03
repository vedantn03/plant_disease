"""Tests for safe local and registry checkpoint resolution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model_loader import resolve_serving_checkpoint
from src.model_loader import verify_manifest


def _cfg(model_source: str):
    return SimpleNamespace(
        serving=SimpleNamespace(model_source=model_source),
        mlflow=SimpleNamespace(
            tracking_uri="sqlite:///mlflow.db",
            registered_model_name="PlantDiseaseClassifier",
            champion_alias="champion",
        ),
    )


def test_local_resolution_targets_champion_without_writing():
    path, provenance = resolve_serving_checkpoint(_cfg("local"), Path("/project"))
    assert path == Path("/project/models/best_model.pth")
    assert provenance["model_source"] == "local"
    assert provenance["registry_match"] == "unknown"
    assert Path(provenance["model_uri"]) == path


def test_unknown_model_source_fails_explicitly():
    with pytest.raises(ValueError, match="model_source"):
        resolve_serving_checkpoint(_cfg("unknown"), Path("/project"))


def test_manifest_hash_and_registry_mismatch(tmp_path):
    import hashlib
    import json

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest = tmp_path / "champion_manifest.json"
    digest = hashlib.md5(b"synthetic checkpoint", usedforsecurity=False).hexdigest()
    manifest.write_text(json.dumps({"version": "2", "run_id": "run-a", "checkpoint_md5": digest}))

    ok = verify_manifest(checkpoint, manifest, "2", "run-a")
    bad_identity = verify_manifest(checkpoint, manifest, "3", "run-b")
    assert ok["registry_match"] is True
    checkpoint.write_bytes(b"changed")
    bad_hash = verify_manifest(checkpoint, manifest, "2", "run-a")
    assert bad_hash["registry_match"] is False
    assert bad_identity["registry_match"] is False


def test_missing_manifest_returns_unknown_and_hash(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"synthetic")
    result = verify_manifest(checkpoint, tmp_path / "missing.json")
    assert result["registry_match"] == "unknown"
    assert result["checkpoint_md5"]


def test_registry_resolution_loads_mounted_checkpoint_and_reports_mismatch(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType

    checkpoint = tmp_path / "models" / "best_model.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"local mounted checkpoint")
    (checkpoint.parent / "champion_manifest.json").write_text(json.dumps({
        "version": "2", "run_id": "run-v2", "checkpoint_md5": "wrong"
    }), encoding="utf-8")

    class FakeClient:
        def get_model_version_by_alias(self, _name, _alias):
            return type(
                "Version", (),
                {"version": "2", "run_id": "run-v2", "source": "file:///host/path"},
            )()

    mlflow_module = ModuleType("mlflow")
    mlflow_module.set_tracking_uri = lambda _uri: None
    mlflow_module.tracking = ModuleType("mlflow.tracking")
    mlflow_module.tracking.MlflowClient = FakeClient
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow_module.tracking)
    path, provenance = resolve_serving_checkpoint(_cfg("registry"), tmp_path)
    assert path == checkpoint
    assert provenance["registry_match"] is False
    assert provenance["artifact_source"] == "file:///host/path"


def test_registry_resolution_uses_versioned_manifest_entry(tmp_path):
    import hashlib
    import json

    checkpoint = tmp_path / "models" / "candidates" / "candidate.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"candidate checkpoint fixture")
    digest = hashlib.md5(checkpoint.read_bytes(), usedforsecurity=False).hexdigest()
    (tmp_path / "models" / "champion_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "active_version": "2",
        "versions": {
            "2": {"run_id": "run-2", "checkpoint_path": "models/best_model.pth",
                  "md5": "unused", "gate_verdict": "BASELINE"},
            "3": {"run_id": "run-3", "checkpoint_path": "models/candidates/candidate.pth",
                  "md5": digest, "gate_verdict": "PASS"},
        },
    }), encoding="utf-8")

    class FakeRegistryClient:
        def get_model_version_by_alias(self, _name, _alias):
            return type("Version", (), {
                "version": "3", "run_id": "run-3", "source": "runs:/run-3/model"
            })()

    path, provenance = resolve_serving_checkpoint(
        _cfg("registry"), tmp_path, registry_client=FakeRegistryClient()
    )
    assert path == checkpoint
    assert provenance["registry_match"] is True
    assert provenance["gate_verdict"] == "PASS"
