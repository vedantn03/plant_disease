"""Dataset-root validation stays enabled outside the serving entry point."""

from types import SimpleNamespace

import pytest

from src import api
from src.config_loader import load_config


def _run_synchronous_coroutine(coroutine):
    """Run a coroutine whose implementation has no await points."""
    with pytest.raises(StopIteration):
        coroutine.send(None)


def _write_config(path):
    path.write_text("data:\n  root: ''\n  image_size: 224\n", encoding="utf-8")
    return path


def test_config_requires_data_root_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "")
    config_path = _write_config(tmp_path / "config.yaml")

    with pytest.raises(EnvironmentError, match="DATA_ROOT is not set"):
        load_config(config_path)


def test_config_can_skip_data_root_validation_for_serving_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "")
    config_path = _write_config(tmp_path / "config.yaml")

    config = load_config(config_path, require_data_root=False)

    assert config.data.root == ""
    assert config.data.image_size == 224


def test_api_startup_uses_dataset_free_serving_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "")
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "configs" / "config.yaml").write_text(
        "data:\n  root: ''\n  image_size: 224\n"
        "serving:\n  model_source: local\n  registry_strict: false\n"
        "  log_file: logs/requests.jsonl\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "__file__", str(project / "src" / "api.py"))
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: False)

    real_load_config = load_config
    def serving_config(path, *, require_data_root=True):
        assert require_data_root is False
        return real_load_config(path, require_data_root=require_data_root)

    monkeypatch.setattr(api, "load_config", serving_config)
    monkeypatch.setattr(
        api,
        "resolve_serving_checkpoint",
        lambda _cfg, _root: (
            project / "models" / "best_model.pth",
            {"model_source": "local", "registry_match": "unknown"},
        ),
    )
    monkeypatch.setattr(api, "AdvisoryEngine", lambda _path: SimpleNamespace())

    _run_synchronous_coroutine(api.startup())
    try:
        assert api._state["cfg"].data.root == ""
        assert api._state["model"] is None  # checkpoint is intentionally absent in this fixture
        assert (project / "logs" / "requests.jsonl").exists()
    finally:
        _run_synchronous_coroutine(api.shutdown())
