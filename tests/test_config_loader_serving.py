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


def _clear_loader_environment(monkeypatch):
    """Prevent process and .env values from affecting config-loader tests."""
    for name in ("DATA_ROOT", "MLFLOW_TRACKING_URI", "SERVING_MODEL_SOURCE"):
        monkeypatch.setenv(name, "")

    # load_dotenv reads the repository's .env file, which is outside tmp_path.
    monkeypatch.setattr("src.config_loader.load_dotenv", lambda **_kwargs: None)


def test_config_requires_data_root_by_default(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = _write_config(tmp_path / "config.yaml")

    with pytest.raises(EnvironmentError, match="DATA_ROOT is not set"):
        load_config(config_path)


def test_config_can_skip_data_root_validation_for_serving_only(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = _write_config(tmp_path / "config.yaml")

    config = load_config(config_path, require_data_root=False)

    assert config.data.root == ""
    assert config.data.image_size == 224


def test_api_startup_uses_dataset_free_serving_config(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
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
    monkeypatch.setattr(api, "AdvisoryEngine", lambda *_paths: SimpleNamespace())

    _run_synchronous_coroutine(api.startup())
    try:
        assert api._state["cfg"].data.root == ""
        assert api._state["model"] is None  # checkpoint is intentionally absent in this fixture
        assert (project / "logs" / "requests.jsonl").exists()
    finally:
        _run_synchronous_coroutine(api.shutdown())


def test_empty_mlflow_override_is_independent_of_environment(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = _write_config(tmp_path / "config.yaml")
    baseline = load_config(config_path, require_data_root=False)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    with_empty_override = load_config(config_path, require_data_root=False)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", " ")
    with_whitespace_override = load_config(config_path, require_data_root=False)

    assert baseline == with_empty_override == with_whitespace_override
    assert "mlflow" not in baseline


def test_mlflow_environment_override_creates_missing_section(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = _write_config(tmp_path / "config.yaml")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example")

    config = load_config(config_path, require_data_root=False)

    assert config.mlflow.tracking_uri == "http://tracking.example"


def test_serving_environment_override_creates_missing_section(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = _write_config(tmp_path / "config.yaml")
    monkeypatch.setenv("SERVING_MODEL_SOURCE", "registry")

    config = load_config(config_path, require_data_root=False)

    assert config.serving.model_source == "registry"


def test_environment_overrides_apply_when_sections_exist(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  root: ''\n  image_size: 224\n"
        "mlflow:\n  tracking_uri: file:./old.db\n"
        "serving:\n  model_source: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///override.db")
    monkeypatch.setenv("SERVING_MODEL_SOURCE", "registry")
    monkeypatch.setenv("DATA_ROOT", "D:/datasets/plant_disease")

    config = load_config(config_path, require_data_root=False)

    assert config.mlflow.tracking_uri == "sqlite:///override.db"
    assert config.serving.model_source == "registry"
    assert config.data.root == "D:/datasets/plant_disease"


def test_data_root_environment_override_creates_missing_section(tmp_path, monkeypatch):
    _clear_loader_environment(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("serving:\n  model_source: local\n", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", "D:/datasets/plant_disease")

    config = load_config(config_path)

    assert config.data.root == "D:/datasets/plant_disease"
