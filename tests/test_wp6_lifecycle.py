"""Synthetic tests for seed spread sync and manifest-backed rollback."""

import json

import yaml

from scripts.sync_gate_spreads import sync_spreads
from scripts.promote import _apply_promotion
from src.model_loader import file_md5
from src.registry import rollback_champion


def test_sync_spreads_copies_ranges_from_report(tmp_path):
    report = tmp_path / "seed_variance.json"
    report.write_text(json.dumps({"metrics": {
        "within_crop_macro_f1": {"range": 0.11},
        "min_recall": {"range": 0.22},
        "bg_drop_per_image_mean": {"range": 0.33},
        "ece": {"range": 0.44},
    }}), encoding="utf-8")
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(yaml.safe_dump({"criteria": {
        name: {"observed_seed_spread": None} for name in (
            "within_crop_macro_f1", "minimum_class_recall",
            "bg_drop_per_image_mean", "ece",
        )
    }}), encoding="utf-8")

    values = sync_spreads(report, thresholds)
    saved = yaml.safe_load(thresholds.read_text(encoding="utf-8"))
    assert values["minimum_class_recall"] == 0.22
    assert saved["criteria"]["bg_drop_per_image_mean"]["observed_seed_spread"] == 0.33


class _FakeRegistryClient:
    def __init__(self):
        self.alias_version = "4"

    def search_model_versions(self, _query):
        return [type("Version", (), {"version": version})() for version in ("2", "3", "4")]

    def set_registered_model_alias(self, _name, _alias, version):
        self.alias_version = str(version)

    def get_model_version(self, _name, version):
        run_id = {"2": "run-2", "3": "run-3"}[str(version)]
        return type("Version", (), {"version": str(version), "run_id": run_id})()

    def get_model_version_by_alias(self, _name, _alias):
        run_id = {"2": "run-2", "3": "run-3"}[self.alias_version]
        return type(
            "Version", (), {"version": self.alias_version, "run_id": run_id}
        )()


def test_apply_uses_fake_registry_and_updates_manifest_without_touching_champion(
    tmp_path, monkeypatch
):
    import scripts.promote as promote
    import src.registry as registry

    mlflow_cfg = type("ML", (), {
        "registered_model_name": "PlantDiseaseClassifier", "champion_alias": "champion"
    })()
    cfg = type("Cfg", (), {"mlflow": mlflow_cfg})()
    monkeypatch.setattr(promote, "load_config", lambda **_kwargs: cfg)
    monkeypatch.setattr(registry, "load_config", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        promote, "checkpoint_dvc_status", lambda *_args, **_kwargs: (True, "fixture verified")
    )

    models = tmp_path / "models"
    models.mkdir()
    champion_path = models / "best_model.pth"
    champion_path.write_bytes(b"synthetic champion checkpoint")
    before_md5 = file_md5(champion_path)
    candidate_path = models / "candidate.pth"
    candidate_path.write_bytes(b"synthetic candidate checkpoint")
    manifest_path = models / "champion_manifest.json"
    manifest_path.write_text(json.dumps({
        "active_version": "2",
        "versions": {"2": {
            "run_id": "run-2",
            "checkpoint_path": "models/best_model.pth",
            "md5": before_md5,
            "gate_verdict": "BASELINE",
        }},
    }), encoding="utf-8")
    client = _FakeRegistryClient()
    client.alias_version = "2"

    result = _apply_promotion(
        candidate=None,
        candidate_row={"run_id": "run-3"},
        verdict={"all_passed": True},
        version="3",
        checkpoint_path=candidate_path,
        client=client,
        manifest_path=manifest_path,
        project_root=tmp_path,
        thresholds_hash="approved-threshold-hash",
        champion_run_id="run-2",
        champion_version="2",
    )

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["alias_moved"] is True
    assert client.alias_version == "3"
    assert saved_manifest["active_version"] == "3"
    assert saved_manifest["versions"]["3"]["gate_verdict"] == "PASS"
    assert file_md5(champion_path) == before_md5


def test_apply_refuses_if_champion_changed_since_dry_run(tmp_path, monkeypatch):
    import scripts.promote as promote
    import src.registry as registry

    cfg = type("Cfg", (), {"mlflow": type("ML", (), {
        "registered_model_name": "PlantDiseaseClassifier", "champion_alias": "champion"
    })()})()
    monkeypatch.setattr(promote, "load_config", lambda **_kwargs: cfg)
    monkeypatch.setattr(registry, "load_config", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        promote, "checkpoint_dvc_status", lambda *_args, **_kwargs: (True, "fixture verified")
    )
    client = _FakeRegistryClient()
    client.alias_version = "3"
    candidate_path = tmp_path / "candidate.pth"
    candidate_path.write_bytes(b"synthetic candidate checkpoint")
    manifest_path = tmp_path / "champion_manifest.json"
    manifest_path.write_text(json.dumps({"active_version": "3", "versions": {}}), encoding="utf-8")
    result = _apply_promotion(
        candidate=None,
        candidate_row={"run_id": "run-3"},
        verdict={"all_passed": True},
        version="3",
        checkpoint_path=candidate_path,
        client=client,
        manifest_path=manifest_path,
        project_root=tmp_path,
        thresholds_hash="approved-threshold-hash",
        champion_run_id="run-2",
        champion_version="2",
    )
    assert result["alias_moved"] is False
    assert result["refuse_reason"] == "champion_version_changed_since_dry_run"


def test_rollback_refuses_when_no_prior_pass_version(tmp_path, monkeypatch):
    import src.registry as registry

    client = _FakeRegistryClient()
    monkeypatch.setattr(registry, "load_config", lambda **_kwargs: type("Cfg", (), {
        "mlflow": type("ML", (), {
            "registered_model_name": "PlantDiseaseClassifier", "champion_alias": "champion"
        })()
    })())
    current_version = type("V", (), {"version": "4"})()
    monkeypatch.setattr(registry, "get_champion_version", lambda _client: current_version)
    monkeypatch.setattr(registry, "append_promotion_log", lambda _record: None)
    manifest = tmp_path / "champion_manifest.json"
    manifest.write_text(json.dumps({"active_version": "4", "versions": {
        "2": {"gate_verdict": "BASELINE"},
        "3": {"gate_verdict": "REJECT"},
    }}), encoding="utf-8")

    result = rollback_champion(
        dry_run=False, client=client, manifest_path=manifest, project_root=tmp_path
    )
    assert result["ok"] is False
    assert "Refusing" in result["reason"]
    assert client.alias_version == "4"


def test_rollback_selects_prior_dvc_verified_pass_and_updates_manifest(tmp_path, monkeypatch):
    import src.registry as registry

    client = _FakeRegistryClient()
    cfg = type("Cfg", (), {"mlflow": type("ML", (), {
        "registered_model_name": "PlantDiseaseClassifier", "champion_alias": "champion"
    })()})()
    monkeypatch.setattr(registry, "load_config", lambda **_kwargs: cfg)
    current_version = type("V", (), {"version": "4"})()
    monkeypatch.setattr(registry, "get_champion_version", lambda _client: current_version)
    monkeypatch.setattr(registry, "append_promotion_log", lambda _record: None)
    monkeypatch.setattr(
        registry,
        "checkpoint_dvc_status",
        lambda *_args, **_kwargs: (True, "fixture DVC pointer verified"),
    )
    checkpoint = tmp_path / "models" / "candidate.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic rollback checkpoint")
    digest = file_md5(checkpoint)
    manifest = tmp_path / "models" / "champion_manifest.json"
    manifest.write_text(json.dumps({"active_version": "4", "versions": {
        "3": {"checkpoint_path": "models/candidate.pth", "md5": digest, "gate_verdict": "PASS"},
        "4": {"checkpoint_path": "models/candidate.pth", "md5": digest, "gate_verdict": "PASS"},
    }}), encoding="utf-8")

    result = rollback_champion(
        dry_run=False, client=client, manifest_path=manifest, project_root=tmp_path
    )
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["from_version"] == 4
    assert result["to_version"] == 3
    assert client.alias_version == "3"
    assert updated["active_version"] == "3"
