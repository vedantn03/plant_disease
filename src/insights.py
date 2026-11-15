"""Read-only, artifact-backed payload for the evaluator dashboard."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _read_promotions(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except OSError:
        return None
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        observed = row.get("observed") or {}
        tags = row.get("tags") or {}
        version = row.get("version") or row.get("candidate_version")
        seed = observed.get("seed") or tags.get("seed")
        candidate = row.get("candidate_run_id") or row.get("run_id")
        key = str(version or (f"seed-{seed}" if seed is not None else candidate or "unknown"))
        timestamp = str(row.get("timestamp", ""))
        if key not in latest or timestamp >= str(latest[key].get("timestamp", "")):
            latest[key] = row
    return sorted(latest.values(), key=lambda row: str(row.get("timestamp", "")), reverse=True)


def _split_counts(root: Path) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    train_counts: Counter[str] = Counter()
    for name in ("train", "val", "test"):
        manifest = root / "data" / "splits" / f"{name}.csv"
        if not manifest.is_file():
            splits[name] = {"status": "not generated yet"}
            continue
        rows = 0
        try:
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows += 1
                    class_name = row.get("class_name")
                    if name == "train" and class_name:
                        train_counts[class_name] += 1
        except OSError:
            splits[name] = {"status": "unavailable"}
            continue
        splits[name] = {"status": "available", "rows": rows}
    train_status = splits["train"]["status"]
    return {
        "status": train_status,
        "splits": splits,
        "class_counts": dict(train_counts),
    }


def _pipeline(root: Path) -> dict[str, Any]:
    dvc_path = root / "dvc.yaml"
    if not dvc_path.is_file():
        return {"status": "not_configured", "stages": []}
    try:
        document = yaml.safe_load(dvc_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"status": "unavailable", "stages": []}
    stages = []
    for name, stage in (document.get("stages") or {}).items():
        outputs = []
        for output in stage.get("outs", []) or []:
            path = output if isinstance(output, str) else output.get("path")
            if not path:
                continue
            outputs.append({"path": path, "present": (root / path).exists()})
        stages.append({
            "name": name,
            "status": "artifacts_present" if outputs and all(x["present"] for x in outputs)
            else "artifacts_missing_or_not_generated",
            "outputs": outputs,
        })
    return {"status": "available", "stages": stages}


def latest_mlflow_run(
    tracking_uri: str | None,
    *,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Read a sanitized latest-run summary; never expose params that can contain paths or IDs."""
    if not tracking_uri:
        return {"status": "not_configured", "run": None}
    try:
        if client_factory is None:
            from mlflow.tracking import MlflowClient

            client_factory = MlflowClient
        client = client_factory(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        experiment_names = {item.experiment_id: item.name for item in experiments}
        experiment_ids = list(experiment_names)
        if not experiment_ids:
            return {"status": "no_experiments", "run": None}
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {"status": "no_runs", "run": None}
        run = runs[0]
        safe_params = {
            key: str(value)
            for key, value in run.data.params.items()
            if key in {
                "seed", "arch", "pretrained", "num_classes", "epochs", "batch_size",
                "lr", "weight_decay", "amp", "imbalance_strategy", "subsample_per_class",
                "device",
            }
        }
        safe_metrics = {
            key: float(value)
            for key, value in run.data.metrics.items()
            if key in {
                "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1",
                "test_loss", "test_acc", "test_macro_f1", "test_weighted_f1",
                "test_accuracy", "test_f1_macro", "test_f1_weighted",
                "best_val_macro_f1", "within_crop_macro_f1_excl_single_class",
                "ece_before", "ece_after", "temperature",
            }
        }
        return {
            "status": "available",
            "run": {
                "experiment": experiment_names.get(run.info.experiment_id, "unknown"),
                "started_at_ms": run.info.start_time,
                "status": run.info.status,
                "params": safe_params,
                "metrics": safe_metrics,
            },
        }
    except Exception as exc:
        # Keep filesystem paths, connection strings, and database details out of the UI.
        return {"status": "unavailable", "run": None, "reason": type(exc).__name__}


def build_insights(
    project_root: str | Path,
    *,
    serving: dict[str, Any],
    process_stats: dict[str, Any],
    telemetry: dict[str, Any],
    class_names: list[str],
    max_report_bytes: int,
    drift_config: dict[str, Any] | None = None,
    mlflow_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build snapshots and live values without running DVC, CI, or registry actions."""
    root = Path(project_root)
    snapshot = root / "reports" / "snapshot"
    report_files = {
        "evaluation": "eval_summary.json",
        "seed_robustness": "seed_variance.json",
        "candidate_gate": "candidate_gate_comparison.json",
        "drift": "drift_latest.json",
        "background_sensitivity": "bg_check_rates.json",
        "advisory_audit": "advisory_audit.json",
        "registry": "registry_export.json",
        "dataset_audit": "dataset_audit.json",
    }
    reports = {key: _read_json(snapshot / filename) for key, filename in report_files.items()}
    explanation_report_path = root / "reports" / "explain_eval.md"
    try:
        with explanation_report_path.open("rb") as handle:
            explanation_report = handle.read(max(0, max_report_bytes)).decode(
                "utf-8", errors="replace"
            )
    except OSError:
        explanation_report = None
    reports["explainability_evaluation"] = explanation_report
    promotion_log = _read_promotions(root / "reports" / "promotion_log.jsonl")
    drift = reports["drift"]
    if drift is not None:
        drift = dict(drift)
        drift["data_kind"] = (
            "simulation" if drift.get("replay_share", 0) or "replay" in str(
                drift.get("traffic_source_note", "")
            ).lower() else "live_or_unclassified"
        )
        reports["drift"] = drift

    class_count = len(set(class_names)) if class_names else None
    crops = sorted({name.split("___", maxsplit=1)[0] for name in class_names})
    coverage = _read_yaml(root / "kb" / "coverage.yaml")
    model_card = root / "docs" / "MODEL_CARD.md"
    limitations = []
    if model_card.is_file():
        content = model_card.read_text(encoding="utf-8", errors="replace")
        marker = "## Limitations"
        if marker in content:
            limitations = [
                line[2:].strip() for line in content.split(marker, 1)[1].splitlines()
                if line.startswith("- ")
            ]

    workflows = list((root / ".github" / "workflows").glob("*.yml"))
    monitoring_files = [
        root / "monitoring" / "prometheus.yml",
        root / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml",
    ]
    return {
        "serving": serving,
        "process": process_stats,
        "log_window": telemetry,
        "data": {
            "status": "serving model classes" if class_names else "serving model unavailable",
            "class_count_from_serving_model": class_count,
            "crop_count_from_serving_model": len(crops) if class_names else None,
            "crops": crops,
            "split_counts": _split_counts(root),
            "coverage_artifact": coverage,
            "quality_audit": reports["dataset_audit"],
        },
        "reports": reports,
        "monitoring_config": drift_config or {},
        "mlflow": mlflow_summary or {"status": "not_connected", "run": None},
        "promotion_history": promotion_log,
        "pipeline": _pipeline(root),
        "components": [
            {"name": "CI", "status": "BUILT" if workflows else "NOT_CONNECTED", "detail": "Workflow files exist; live job status is not connected."},
            {"name": "Airflow", "status": "SIMULATED", "detail": "Repository runner is a simulation."},
            {"name": "SageMaker", "status": "DOCUMENTED_ONLY", "detail": "Local mapping/documentation only."},
            {"name": "Prometheus/Grafana", "status": "BUILT" if all(path.is_file() for path in monitoring_files) else "NOT_CONNECTED", "detail": "Configured locally; runtime is not verified by this API."},
            {"name": "DVC", "status": "BUILT" if (root / "dvc.yaml").is_file() else "NOT_CONNECTED", "detail": "Stage declarations are read-only."},
        ],
        "model_card_limitations": limitations,
        "source_paths": {
            "evaluation": "reports/snapshot/eval_summary.json",
            "seed_robustness": "reports/snapshot/seed_variance.json",
            "candidate_gate": "reports/snapshot/candidate_gate_comparison.json",
            "drift": "reports/snapshot/drift_latest.json",
            "advisory_audit": "reports/snapshot/advisory_audit.json",
            "registry": "reports/snapshot/registry_export.json",
            "explainability_evaluation": "reports/explain_eval.md",
        },
    }
