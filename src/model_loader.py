"""Resolve a registry alias to a verified, mounted DVC checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def file_md5(path: Path) -> str:
    """Return the MD5 digest used by DVC for an existing checkpoint file."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read the version map, accepting the prior single-version v2 schema."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    versions = payload.get("versions")
    if isinstance(versions, dict):
        return payload
    version = str(payload.get("version", ""))
    if not version:
        raise ValueError("Champion manifest has no version map")
    return {
        "schema_version": 1,
        "active_version": version,
        "versions": {
            version: {
                "run_id": payload.get("run_id"),
                "checkpoint_path": "models/best_model.pth",
                "md5": payload.get("md5", payload.get("checkpoint_md5")),
                "gate_verdict": payload.get("gate_verdict", "BASELINE"),
            }
        },
    }


def atomic_write_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the manifest after writing and syncing a sibling temp file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def safe_checkpoint_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest checkpoint_path must stay inside the project") from exc
    return path


def verify_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    registry_version: str | None = None,
    registry_run_id: str | None = None,
) -> dict[str, str | bool | None]:
    """Compare a checkpoint digest and optional registry identity to the manifest."""
    if not manifest_path.exists():
        return {
            "registry_match": "unknown",
            "checkpoint_md5": file_md5(checkpoint_path) if checkpoint_path.exists() else None,
        }
    try:
        manifest = read_manifest(manifest_path)
        version = str(registry_version or manifest.get("active_version", ""))
        entry = manifest["versions"][version]
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return {"registry_match": False, "checkpoint_md5": None, "registry_error": str(exc)}
    actual_md5 = file_md5(checkpoint_path) if checkpoint_path.exists() else None
    expected_md5 = str(entry.get("md5", "")).lower()
    identity_match = (
        (registry_version is None or version == str(registry_version))
        and (registry_run_id is None or entry.get("run_id") == registry_run_id)
    )
    hash_match = actual_md5 is not None and actual_md5.lower() == expected_md5
    return {
        "registry_match": bool(identity_match and hash_match),
        "checkpoint_md5": actual_md5,
        "manifest_version": version,
        "manifest_run_id": entry.get("run_id"),
        "manifest_gate_verdict": entry.get("gate_verdict"),
    }


def resolve_serving_checkpoint(
    cfg: Any,
    project_root: Path,
    registry_client: Any | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve alias identity, then load bytes only from its manifest checkpoint path."""
    source = str(getattr(cfg.serving, "model_source", "local")).lower()
    if source not in {"local", "registry"}:
        raise ValueError(f"Unsupported serving.model_source={source!r}; use 'local' or 'registry'.")

    manifest_path = project_root / "models" / "champion_manifest.json"
    manifest: dict[str, Any] | None = None
    registry_info: dict[str, Any] = {}
    if source == "registry":
        try:
            if registry_client is None:
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
                registry_client = MlflowClient()
            model_name = str(cfg.mlflow.registered_model_name)
            alias = str(cfg.mlflow.champion_alias)
            registry_version = registry_client.get_model_version_by_alias(model_name, alias)
            registry_info = {
                "registry_version": str(registry_version.version),
                "registry_run_id": str(registry_version.run_id),
                "artifact_source": str(getattr(registry_version, "source", "unknown")),
                "model_uri": f"models:/{model_name}@{alias}",
            }
        except Exception as exc:
            registry_info = {"registry_error": str(exc), "model_uri": "registry alias unavailable"}

    try:
        manifest = read_manifest(manifest_path)
        version = str(registry_info.get("registry_version") or manifest["active_version"])
        entry = manifest["versions"][version]
        path = safe_checkpoint_path(project_root, str(entry["checkpoint_path"]))
        actual_md5 = file_md5(path) if path.is_file() else None
        expected_md5 = str(entry.get("md5", "")).lower()
        run_id_match = (
            source != "registry"
            or entry.get("run_id") == registry_info.get("registry_run_id")
        )
        match = actual_md5 is not None and actual_md5.lower() == expected_md5 and run_id_match
        error = None
        if actual_md5 is None:
            error = f"Manifest checkpoint is missing: {path}"
        elif actual_md5.lower() != expected_md5:
            error = "Checkpoint MD5 does not match the champion manifest"
        elif not run_id_match:
            error = "Registry alias run_id does not match the champion manifest"
        if source == "registry" and registry_info.get("registry_error"):
            match = False
            error = registry_info["registry_error"]
        registry_info.update({
            "model_source": source,
            "model_uri": registry_info.get("model_uri", path.as_posix()),
            "checkpoint_md5": actual_md5,
            "manifest_version": version,
            "manifest_run_id": entry.get("run_id"),
            "manifest_checkpoint_path": entry.get("checkpoint_path"),
            "gate_verdict": entry.get("gate_verdict"),
            "registry_match": match if source == "registry" else "unknown",
        })
        if error:
            registry_info["registry_error"] = error
        return path, registry_info
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        fallback = project_root / "models" / "best_model.pth"
        registry_info.update({
            "model_source": source,
            "model_uri": registry_info.get("model_uri", fallback.as_posix()),
            "registry_match": False if source == "registry" else "unknown",
            "checkpoint_md5": None,
            "registry_error": f"Could not resolve checkpoint from manifest: {exc}",
        })
        return fallback, registry_info
