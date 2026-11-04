"""Read-only MLflow registry inspection for the configured model."""

from __future__ import annotations

import json
import argparse
import re
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


def inspect_registry(
    include_artifacts: bool = True,
    tracking_uri_override: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(require_data_root=False)
    tracking_uri = tracking_uri_override or cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
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

        if include_artifacts:
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
        "tracking_uri": tracking_uri,
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


def sanitized_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep registry identity and compatibility category, dropping paths and arbitrary tags."""
    def clean(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        value = re.sub(r"Vedant", "<redacted-user>", value, flags=re.IGNORECASE)
        return re.sub(
            r"(?:[A-Za-z]:[\\/][^\s,;]*|file:(?:/{1,3})[^\s,;]*)",
            "<redacted-local-path>",
            value,
            flags=re.IGNORECASE,
        )

    versions = []
    for item in payload.get("versions", []):
        source = str(item.get("source", ""))
        if re.match(r"^[A-Za-z]:[\\/]", source) or source.lower().startswith("file:"):
            source_kind = "local_file_uri"
        else:
            source_kind = "remote_or_relative_uri"
        tags = item.get("tags") or {}
        versions.append({
            "version": clean(str(item.get("version", "unknown"))),
            "run_id": clean(str(item.get("run_id", "unknown"))),
            "creation_timestamp": item.get("creation_timestamp"),
            "current_stage": item.get("current_stage"),
            "gate_verdict": clean(tags.get("gate_verdict", "unknown")),
            "artifact_source_kind": source_kind,
        })
    alias = payload.get("alias") or {}
    return {
        "model": clean(payload.get("model", "unknown")),
        "alias": {
            "name": clean(alias.get("name")),
            "version": clean(str(alias["version"])) if alias.get("version") is not None else None,
        },
        "versions": versions,
        "container_compatibility": payload.get("container_compatibility", {}),
        "source_note": "Paths, arbitrary tags, params, metrics, and usernames are omitted.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Write a sanitized JSON export without downloading artifact files.",
    )
    parser.add_argument(
        "--tracking-uri",
        help="Optional store URI override (useful when this script is run from another checkout).",
    )
    args = parser.parse_args()
    try:
        payload = inspect_registry(
            include_artifacts=args.snapshot is None,
            tracking_uri_override=args.tracking_uri,
        )
        if args.snapshot:
            safe_payload = sanitized_snapshot(payload)
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot.write_text(
                json.dumps(_jsonable(safe_payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                display_path = args.snapshot.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                display_path = args.snapshot.name
            print(f"Wrote sanitized registry snapshot: {display_path}")
        else:
            print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
