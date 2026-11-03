"""Read-only MLflow registry inspection for the configured model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config_loader import load_config


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def inspect_registry() -> dict[str, Any]:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    name = str(cfg.mlflow.registered_model_name)
    alias_name = str(cfg.mlflow.champion_alias)

    try:
        versions = list(client.search_model_versions(f"name='{name}'"))
    except MlflowException as exc:
        return {"model": name, "error": str(exc), "versions": []}

    try:
        champion = client.get_model_version_by_alias(name, alias_name)
        alias = {"name": alias_name, "version": str(champion.version)}
    except Exception as exc:
        alias = {"name": alias_name, "version": None, "error": str(exc)}

    inspected = []
    for version in sorted(versions, key=lambda item: int(item.version)):
        run_payload: dict[str, Any] = {}
        artifact = {
            "source": version.source,
            "source_uri": version.source,
            "retrievable": False,
            "error": None,
        }
        try:
            run = client.get_run(version.run_id)
            run_payload = {
                "run_id": version.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }
        except Exception as exc:
            run_payload = {"run_id": version.run_id, "error": str(exc)}

        try:
            downloaded = mlflow.artifacts.download_artifacts(artifact_uri=version.source)
            artifact["retrievable"] = Path(downloaded).exists()
            artifact["downloaded_path"] = str(downloaded)
        except Exception as exc:
            artifact["error"] = str(exc)

        inspected.append({
            "version": str(version.version),
            "creation_timestamp": version.creation_timestamp,
            "last_updated_timestamp": version.last_updated_timestamp,
            "current_stage": version.current_stage,
            "source": version.source,
            "run_id": version.run_id,
            "tags": dict(version.tags or {}),
            "run": run_payload,
            "artifact": artifact,
        })
        inspected[-1]["artifact_source_uri"] = str(version.source)

    return {
        "tracking_uri": cfg.mlflow.tracking_uri,
        "model": name,
        "alias": alias,
        "versions": inspected,
        "container_compatibility": {
            "status": "not_verified",
            "reason": (
                "This inspection runs on the host; source URI readability inside a Linux "
                "container requires a container-side check."
            ),
        },
        "notes": [
            "Read-only inspection: no tags, aliases, runs, or artifacts are modified.",
            "A retrievable artifact is not automatically approved for promotion.",
        ],
    }


def main() -> int:
    try:
        payload = inspect_registry()
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
