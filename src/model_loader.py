"""
Serving checkpoint resolution.

Local checkpoint loading remains the default. Registry-backed loading is an
explicit configuration choice and resolves the configured MLflow alias to the
registered run artifact without writing to models/best_model.pth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _registry_checkpoint(source: str) -> Path:
    """Download a registered run artifact and return its sole .pth file."""
    import mlflow

    artifact_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=source))
    checkpoints = (
        [artifact_dir]
        if artifact_dir.is_file() and artifact_dir.suffix == ".pth"
        else sorted(artifact_dir.rglob("*.pth"))
    )
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one .pth checkpoint under MLflow artifact {source!r}; "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def resolve_serving_checkpoint(cfg: Any, project_root: Path) -> tuple[Path, dict[str, str]]:
    """
    Resolve the serving checkpoint according to ``cfg.serving.model_source``.

    Returns the checkpoint path and provenance fields suitable for health/log
    reporting. Registry mode never replaces the local champion file.
    """
    source = str(getattr(cfg.serving, "model_source", "local")).lower()
    if source == "local":
        path = project_root / "models" / "best_model.pth"
        return path, {"model_source": "local", "model_uri": str(path)}

    if source != "registry":
        raise ValueError(f"Unsupported serving.model_source={source!r}; use 'local' or 'registry'.")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    name = str(cfg.mlflow.registered_model_name)
    alias = str(cfg.mlflow.champion_alias)
    try:
        version = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise RuntimeError(f"Could not resolve MLflow alias {name}@{alias}: {exc}") from exc

    source_uri = str(version.source)
    checkpoint = _registry_checkpoint(source_uri)
    return checkpoint, {
        "model_source": "registry",
        "model_uri": f"models:/{name}@{alias}",
        "model_version": str(version.version),
        "run_id": str(version.run_id),
        "artifact_source": source_uri,
    }
