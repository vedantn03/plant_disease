"""
Serving checkpoint resolution.

The mounted checkpoint is always loaded locally. In registry mode the configured
MLflow alias supplies identity metadata that is checked against the committed
manifest; no registry artifact is downloaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_md5(path: Path) -> str:
    """Return the MD5 digest used by DVC for an existing checkpoint file."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    registry_version: str | None = None,
    registry_run_id: str | None = None,
) -> dict[str, str | bool | None]:
    """Compare checkpoint MD5 and optional registry identity with a committed manifest."""
    if not manifest_path.exists():
        return {
            "registry_match": "unknown",
            "checkpoint_md5": file_md5(checkpoint_path) if checkpoint_path.exists() else None,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_md5 = file_md5(checkpoint_path) if checkpoint_path.exists() else None
    identity_match = (
        (registry_version is None or str(manifest.get("version")) == str(registry_version))
        and (registry_run_id is None or manifest.get("run_id") == registry_run_id)
    )
    hash_match = actual_md5 is not None and actual_md5.lower() == str(
        manifest.get("checkpoint_md5", "")
    ).lower()
    return {
        "registry_match": bool(identity_match and hash_match),
        "checkpoint_md5": actual_md5,
        "manifest_version": str(manifest.get("version")),
        "manifest_run_id": manifest.get("run_id"),
    }


def resolve_serving_checkpoint(cfg: Any, project_root: Path) -> tuple[Path, dict[str, str]]:
    """
    Resolve the serving checkpoint according to ``cfg.serving.model_source``.

    Returns the checkpoint path and provenance fields suitable for health/log
    reporting. Registry mode never replaces the local champion file.
    """
    source = str(getattr(cfg.serving, "model_source", "local")).lower()
    if source not in {"local", "registry"}:
        raise ValueError(f"Unsupported serving.model_source={source!r}; use 'local' or 'registry'.")

    path = project_root / "models" / "best_model.pth"
    manifest_path = project_root / "models" / "champion_manifest.json"
    registry_info: dict[str, Any] = {}
    if source == "registry":
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = MlflowClient()
        name = str(cfg.mlflow.registered_model_name)
        alias = str(cfg.mlflow.champion_alias)
        try:
            version = client.get_model_version_by_alias(name, alias)
            registry_info = {
                "registry_version": str(version.version),
                "registry_run_id": str(version.run_id),
                "artifact_source": str(version.source),
                "model_uri": f"models:/{name}@{alias}",
            }
        except Exception as exc:
            registry_info = {"registry_error": str(exc), "model_uri": f"models:/{name}@{alias}"}

    verification = verify_manifest(
        path,
        manifest_path,
        registry_info.get("registry_version"),
        registry_info.get("registry_run_id"),
    )
    if source == "local":
        verification["registry_match"] = "unknown"
    if source == "registry" and "registry_error" in registry_info:
        verification["registry_match"] = False
    return path, {
        "model_source": source,
        "model_uri": registry_info.get("model_uri", path.as_posix()),
        **registry_info,
        **verification,
    }
