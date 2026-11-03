"""
src/registry.py
===============
MLflow Model Registry helpers using *aliases*, not stages.

MLflow 2.9+ deprecates registry stages in favour of aliases and tags.
Docs: https://mlflow.org/docs/latest/ml/model-registry/
       https://www.mlflow.org/docs/latest/ml/model-registry/workflow.md
       ("As of MLflow 2.9.0, Model Stages have been deprecated...")

This module never writes models/best_model.pth.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from src.config_loader import load_config, _project_root
from src.model_loader import atomic_write_manifest, file_md5, read_manifest, safe_checkpoint_path
from src.promotion_safety import checkpoint_dvc_status
from src.utils import get_logger

log = get_logger("registry")
PROJECT_ROOT = _project_root()
PROMOTION_LOG = PROJECT_ROOT / "reports" / "promotion_log.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def dvc_data_hash() -> str:
    manifest = PROJECT_ROOT / "data" / "splits" / "manifest_hashes.json"
    if not manifest.exists():
        return "unknown"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def config_hash() -> str:
    return sha256_file(PROJECT_ROOT / "configs" / "config.yaml")[:16]


def default_tags(extra: dict[str, str] | None = None) -> dict[str, str]:
    tags = {
        "dvc_data_hash": dvc_data_hash(),
        "git_commit": git_commit(),
        "config_hash": config_hash(),
        "split": "val",
    }
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def get_client() -> MlflowClient:
    cfg = load_config(require_data_root=False)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return MlflowClient()


def get_champion_version(client: MlflowClient | None = None) -> Any | None:
    cfg = load_config(require_data_root=False)
    client = client or get_client()
    try:
        return client.get_model_version_by_alias(
            cfg.mlflow.registered_model_name,
            cfg.mlflow.champion_alias,
        )
    except MlflowException:
        return None
    except Exception:
        return None


def append_promotion_log(record: dict[str, Any]) -> None:
    PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(PROMOTION_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def set_version_tags(name: str, version: str, tags: dict[str, str], client: MlflowClient | None = None) -> None:
    client = client or get_client()
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, str(value))


def register_candidate_version(
    source: str,
    run_id: str,
    tags: dict[str, str] | None = None,
    client: MlflowClient | None = None,
) -> Any:
    """Register a model artifact as a candidate with provenance tags."""
    cfg = load_config()
    client = client or get_client()
    name = cfg.mlflow.registered_model_name
    try:
        client.create_registered_model(name)
    except MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise

    version_tags = default_tags({
        "status": "candidate",
        "gate_verdict": "PENDING",
        "robustness_metrics": "not_computed",
        **(tags or {}),
    })
    return client.create_model_version(
        name=name,
        source=source,
        run_id=run_id,
        tags=version_tags,
    )


def promote_alias(
    version: str,
    alias: str | None = None,
    dry_run: bool = True,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """
    Point an alias at `version`. dry_run=True never calls set_registered_model_alias.
    """
    cfg = load_config(require_data_root=False)
    alias = alias or cfg.mlflow.champion_alias
    client = client or get_client()
    name = cfg.mlflow.registered_model_name
    action = {
        "model": name,
        "alias": alias,
        "target_version": str(version),
        "dry_run": dry_run,
        "moved": False,
    }
    if dry_run:
        log.info(f"[dry-run] would set alias {name}@{alias} -> v{version}")
        return action
    client.set_registered_model_alias(name, alias, str(version))
    action["moved"] = True
    log.info(f"Set alias {name}@{alias} -> v{version}")
    return action


def rollback_champion(
    dry_run: bool = True,
    client: MlflowClient | None = None,
    *,
    manifest_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Move the alias to the highest prior DVC-verified PASS version."""
    cfg = load_config(require_data_root=False)
    client = client or get_client()
    name = cfg.mlflow.registered_model_name
    current = get_champion_version(client)
    if current is None:
        msg = "Champion alias is not set; refusing rollback."
        record = {"action": "rollback", "ok": False, "reason": msg, "dry_run": dry_run}
        append_promotion_log(record)
        return record

    project_root = project_root or PROJECT_ROOT
    manifest_path = manifest_path or project_root / "models" / "champion_manifest.json"
    try:
        manifest = read_manifest(manifest_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        msg = f"Could not read champion manifest: {exc}"
        record = {"action": "rollback", "ok": False, "reason": msg, "dry_run": dry_run}
        append_promotion_log(record)
        return record

    current_ver = int(current.version)
    registered_versions = {
        int(mv.version) for mv in client.search_model_versions(f"name='{name}'")
    }
    eligible: list[tuple[int, str]] = []
    for version_text, entry in manifest.get("versions", {}).items():
        try:
            version_num = int(version_text)
        except ValueError:
            continue
        if version_num >= current_ver or version_num not in registered_versions:
            continue
        if entry.get("gate_verdict") != "PASS":
            continue
        try:
            checkpoint = safe_checkpoint_path(project_root, str(entry.get("checkpoint_path", "")))
        except ValueError:
            continue
        if not checkpoint.is_file() or file_md5(checkpoint) != str(entry.get("md5", "")).lower():
            continue
        dvc_ok, _ = checkpoint_dvc_status(checkpoint, git_root=project_root)
        if dvc_ok:
            eligible.append((version_num, version_text))

    if not eligible:
        msg = (
            "No prior DVC-verified version with gate_verdict=PASS. Refusing rollback."
        )
        log.error(msg)
        record = {"action": "rollback", "ok": False, "reason": msg, "dry_run": dry_run}
        append_promotion_log(record)
        return record

    target_num, target_version = max(eligible)
    move = promote_alias(target_version, dry_run=dry_run, client=client)
    record = {
        "action": "rollback",
        "ok": True,
        "from_version": current_ver,
        "to_version": target_num,
        **move,
    }
    if not dry_run:
        prior_manifest = json.loads(json.dumps(manifest))
        try:
            manifest["active_version"] = target_version
            atomic_write_manifest(manifest_path, manifest)
        except Exception as exc:
            promote_alias(str(current_ver), dry_run=False, client=client)
            atomic_write_manifest(manifest_path, prior_manifest)
            record.update({"ok": False, "reason": f"Manifest update failed; alias restored: {exc}"})
    append_promotion_log(record)
    return record
