"""Tests for safe local and registry checkpoint resolution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model_loader import resolve_serving_checkpoint


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
    assert Path(provenance["model_uri"]) == path


def test_unknown_model_source_fails_explicitly():
    with pytest.raises(ValueError, match="model_source"):
        resolve_serving_checkpoint(_cfg("unknown"), Path("/project"))
