"""Evaluate a VAL candidate; apply is gated by a committed approval hash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.gates import (
    GateMetrics,
    evaluate_gate,
    load_gate_thresholds,
    verify_threshold_approval,
)
from src.model_loader import atomic_write_manifest, file_md5, read_manifest
from src.promotion_safety import checkpoint_dvc_status
from src.registry import append_promotion_log, get_client, promote_alias
from src.utils import get_logger

log = get_logger("promote")
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "gates" / "gate_thresholds.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs" / "gates" / "approval.yaml"


def _select_row(
    payload: dict[str, Any], *, run_id: str | None = None, name: str | None = None
) -> dict[str, Any]:
    """Select one evaluated VAL row deterministically from a comparison report."""
    rows = payload.get("results")
    if not isinstance(rows, list):
        return payload
    if run_id:
        match = next((row for row in rows if row.get("run_id") == run_id), None)
        if match is None:
            raise ValueError(f"Run ID {run_id} is absent from the metrics report")
        return match
    if name:
        match = next((row for row in rows if row.get("name") == name), None)
        if match is None:
            raise ValueError(f"Model row {name!r} is absent from the metrics report")
        return match
    candidates = [row for row in rows if row.get("name") != "champion_v2"]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and len(rows) == 1:
        return rows[0]
    raise ValueError("Select a metrics row with --candidate-run-id or --candidate-name")


def _metrics_from_payload(payload: dict[str, Any]) -> GateMetrics:
    """Extract VAL metrics, including all class supports and mean-fill drop."""
    row = _select_row(payload)
    min_recall = row["min_recall"]
    fills = row["masked_background_by_fill"]
    if "per_image_mean" not in fills:
        raise ValueError("VAL report lacks masked_background_by_fill.per_image_mean")
    per_class = min_recall.get("all", {})
    return GateMetrics(
        macro_f1=float(row["clean_macro_f1"]),
        within_crop_macro_f1=float(row["within_crop_macro_f1"]),
        min_recall=float(min_recall["value"]),
        min_recall_class=min_recall.get("class"),
        min_recall_n=int(min_recall["n"]) if min_recall.get("n") is not None else None,
        min_recall_correct=(
            int(min_recall["correct"]) if min_recall.get("correct") is not None else None
        ),
        per_class_recall={
            str(class_name): {
                "recall": float(values["recall"]),
                "n": int(values["n"]),
                "correct": int(values["correct"]),
            }
            for class_name, values in per_class.items()
        },
        bg_drop=float(fills["black"]["bg_drop"]),
        bg_drop_per_image_mean=float(fills["per_image_mean"]["bg_drop"]),
        ece=float(row["ece"]),
        extra={
            "run_id": row.get("run_id"),
            "seed": row.get("seed"),
            "clean_macro_f1": float(row["clean_macro_f1"]),
            "bg_drop_black": float(fills["black"]["bg_drop"]),
            "bg_drop_mid_grey": float(fills["mid_grey"]["bg_drop"]),
        },
    )


def _load_metrics(path: Path, *, run_id: str | None = None, name: str | None = None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared_split = str(payload.get("split", "VAL")).upper()
    if declared_split != "VAL":
        raise ValueError(
            f"Promotion selection requires VAL metrics; {path} declares {declared_split}"
        )
    row = _select_row(payload, run_id=run_id, name=name)
    row_split = str(row.get("split", "VAL")).upper()
    if row_split != "VAL":
        raise ValueError(
            f"Promotion selection requires VAL metrics; selected row declares {row_split}"
        )
    return _metrics_from_payload(row), row


def _require_observed_spreads(thresholds: dict[str, Any]) -> None:
    missing = [
        name for name, criterion in thresholds["criteria"].items()
        if not isinstance(criterion.get("observed_seed_spread"), (int, float))
    ]
    if missing:
        raise ValueError(
            "Observed seed spreads are not populated for: " + ", ".join(missing)
            + ". Run scripts/sync_gate_spreads.py from the saved WP1 report first."
        )


def _apply_promotion(
    *,
    candidate: GateMetrics,
    candidate_row: dict[str, Any],
    verdict: dict[str, Any],
    version: str,
    checkpoint_path: Path,
    client: Any,
    manifest_path: Path,
    project_root: Path,
    thresholds_hash: str,
    champion_run_id: str,
    champion_version: str,
) -> dict[str, Any]:
    """Move alias then atomically replace manifest, restoring alias on failure."""
    if not verdict["all_passed"]:
        return {"alias_moved": False, "refuse_reason": "candidate_failed_gate"}
    run_id = str(candidate_row.get("run_id") or "")
    if not run_id:
        return {"alias_moved": False, "refuse_reason": "candidate_run_id_missing"}
    cfg = load_config(require_data_root=False)
    try:
        version_obj = client.get_model_version(cfg.mlflow.registered_model_name, str(version))
    except Exception as exc:
        return {"alias_moved": False, "refuse_reason": f"registry_version_not_found: {exc}"}
    if str(version_obj.run_id) != run_id:
        return {"alias_moved": False, "refuse_reason": "candidate_run_id_mismatch"}

    dvc_ok, dvc_message = checkpoint_dvc_status(checkpoint_path, git_root=project_root)
    if not dvc_ok:
        return {
            "alias_moved": False,
            "refuse_reason": f"dvc_checkpoint_not_verified: {dvc_message}",
        }

    try:
        manifest = read_manifest(manifest_path)
        previous_alias = client.get_model_version_by_alias(
            cfg.mlflow.registered_model_name, cfg.mlflow.champion_alias
        )
        previous_version = str(previous_alias.version)
        if previous_version != str(champion_version):
            return {"alias_moved": False, "refuse_reason": "champion_version_changed_since_dry_run"}
        if str(previous_alias.run_id) != str(champion_run_id):
            return {"alias_moved": False, "refuse_reason": "champion_run_id_changed_since_dry_run"}
    except Exception as exc:
        return {
            "alias_moved": False,
            "refuse_reason": f"champion_manifest_or_alias_unavailable: {exc}",
        }

    if str(manifest.get("active_version")) != previous_version:
        return {"alias_moved": False, "refuse_reason": "manifest_active_version_differs_from_alias"}

    relative_checkpoint = checkpoint_path.resolve().relative_to(project_root.resolve()).as_posix()
    candidate_digest = file_md5(checkpoint_path)
    previous_manifest = json.loads(json.dumps(manifest))
    manifest.setdefault("versions", {})[str(version)] = {
        "run_id": run_id,
        "checkpoint_path": relative_checkpoint,
        "md5": candidate_digest,
        "gate_verdict": "PASS",
        "thresholds_sha256": thresholds_hash,
    }

    move = promote_alias(str(version), dry_run=False, client=client)
    if not move.get("moved"):
        return {"alias_moved": False, "refuse_reason": "registry_alias_did_not_move"}
    try:
        manifest["active_version"] = str(version)
        atomic_write_manifest(manifest_path, manifest)
    except Exception as exc:
        try:
            promote_alias(previous_version, dry_run=False, client=client)
            atomic_write_manifest(manifest_path, previous_manifest)
        except Exception as rollback_exc:
            return {
                "alias_moved": True,
                "refuse_reason": (
                    f"manifest update failed ({exc}); alias compensation failed ({rollback_exc})"
                ),
            }
        return {
            "alias_moved": False,
            "refuse_reason": f"manifest update failed; alias restored: {exc}",
        }

    return {
        "alias_moved": True,
        "previous_version": previous_version,
        "version": str(version),
        "checkpoint_path": relative_checkpoint,
        "checkpoint_md5": candidate_digest,
        "gate_verdict": "PASS",
        "dvc_status": dvc_message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", default="reports/candidate_gate_48ecf14d.json")
    parser.add_argument("--champion-metrics", default="reports/candidate_gate_comparison.json")
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--champion-run-id", required=True)
    parser.add_argument("--champion-version", default="2")
    parser.add_argument("--version", required=True, help="Existing MLflow registry version")
    parser.add_argument("--candidate-checkpoint", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Evaluate without registry writes (default)"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply only with committed threshold approval"
    )
    args = parser.parse_args()
    if args.dry_run and args.apply:
        parser.error("Choose either --dry-run or --apply")
    apply = bool(args.apply)

    candidate_path = PROJECT_ROOT / args.candidate_metrics
    champion_path = PROJECT_ROOT / args.champion_metrics
    thresholds_path = THRESHOLDS_PATH
    approval_path = APPROVAL_PATH
    manifest_path = PROJECT_ROOT / "models" / "champion_manifest.json"
    try:
        thresholds = load_gate_thresholds(thresholds_path)
        _require_observed_spreads(thresholds)
        candidate, candidate_row = _load_metrics(
            candidate_path, run_id=args.candidate_run_id, name=args.candidate_name
        )
        champion, _ = _load_metrics(champion_path, run_id=args.champion_run_id)
        approval_ok, approval_reason, thresholds_hash = verify_threshold_approval(
            approval_path,
            thresholds_path,
            require_committed=True,
            project_root=PROJECT_ROOT,
        )
        verdict = evaluate_gate(candidate, champion, thresholds, approved=approval_ok)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    record: dict[str, Any] = {
        "action": "promote",
        "dry_run": not apply,
        "model": "PlantDiseaseClassifier",
        "alias": "champion",
        "candidate_run_id": args.candidate_run_id,
        "version": str(args.version),
        "champion_run_id": args.champion_run_id,
        "champion_version": args.champion_version,
        "split": "VAL",
        "observed": candidate.extra,
        "thresholds_sha256": thresholds_hash,
        "approval_valid": approval_ok,
        "approval_message": approval_reason,
        "verdict": verdict,
        "alias_moved": False,
    }

    if apply and not approval_ok:
        record["refuse_reason"] = approval_reason
        append_promotion_log(record)
        print(json.dumps(record, indent=2))
        return 2

    if apply:
        raw_checkpoint = args.candidate_checkpoint or candidate_row.get("checkpoint")
        if not raw_checkpoint:
            raw_checkpoint = f"models/candidates/{args.candidate_run_id}.pth"
        checkpoint_path = (PROJECT_ROOT / raw_checkpoint).resolve()
        try:
            checkpoint_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            print(
                json.dumps({"error": "Candidate checkpoint must stay inside the project"}),
                file=sys.stderr,
            )
            return 2
        try:
            apply_result = _apply_promotion(
                candidate=candidate,
                candidate_row=candidate_row,
                verdict=verdict,
                version=str(args.version),
                checkpoint_path=checkpoint_path,
                client=get_client(),
                manifest_path=manifest_path,
                project_root=PROJECT_ROOT,
                thresholds_hash=thresholds_hash,
                champion_run_id=args.champion_run_id,
                champion_version=args.champion_version,
            )
        except Exception as exc:
            apply_result = {"alias_moved": False, "refuse_reason": str(exc)}
        record.update(apply_result)
        if not record.get("alias_moved"):
            record.setdefault("refuse_reason", "candidate_failed_gate")
    else:
        record["notes"] = "Dry-run performs no registry writes and never moves the alias."

    append_promotion_log(record)
    print(json.dumps(record, indent=2))
    if apply and not record.get("alias_moved"):
        return 1
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
