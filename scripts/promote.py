"""
scripts/promote.py
==================
Evaluate a candidate against the 4-criterion VAL trust gate and (optionally)
move the MLflow alias @champion.

Until gate thresholds are approved, every run is dry: the verdict is appended
to reports/promotion_log.jsonl and @champion is never moved.

Usage:
  python scripts/promote.py --dry-run
  python scripts/promote.py --candidate-metrics reports/gate_baselines_val.json
  python scripts/promote.py --candidate-metrics reports/candidate_gate_48ecf14d.json --champion-metrics reports/candidate_gate_comparison.json --champion-version 2
  python scripts/promote.py --apply   # refused while criteria are provisional
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.gates import GateMetrics, evaluate_gate, load_gate_thresholds
from src.registry import (
    append_promotion_log,
    default_tags,
    get_client,
    promote_alias,
    set_version_tags,
)
from src.utils import get_logger

log = get_logger("promote")


def _metrics_from_payload(payload: dict) -> GateMetrics:
    """Accept either a flat gate-baselines file or a nested metrics dump."""
    if "results" in payload:
        rows = payload["results"]
        row = next((item for item in rows if item.get("name") == "candidate" or
                    str(item.get("name", "")).startswith("candidate_seed_")), rows[-1])
        recall = row["min_recall"]
        fills = row["masked_background_by_fill"]
        return GateMetrics(
            macro_f1=float(row["clean_macro_f1"]),
            within_crop_macro_f1=float(row["within_crop_macro_f1"]),
            min_recall=float(recall["value"]),
            bg_drop=float(fills["black"]["bg_drop"]),
            ece=float(row["ece"]),
            min_recall_class=recall.get("class"),
            min_recall_n=recall.get("n"),
        )
    if "masked_background" in payload:
        mb = payload["masked_background"]
        rec = payload.get("min_recall", {})
        cal = payload.get("calibration", {})
        return GateMetrics(
            macro_f1=float(payload.get("clean_macro_f1", mb.get("clean_macro_f1"))),
            within_crop_macro_f1=payload.get("within_crop_macro_f1_excluding_single_class_crops"),
            min_recall=float(rec.get("value", rec) if isinstance(rec, dict) else rec),
            bg_drop=float(mb["bg_drop"]),
            ece=float(cal.get("ece_before", cal.get("ece", payload.get("ece")))),
            min_recall_class=rec.get("class") if isinstance(rec, dict) else None,
            min_recall_n=rec.get("n") if isinstance(rec, dict) else None,
            min_recall_wilson_lower=rec.get("wilson_lower") if isinstance(rec, dict) else None,
        )
    headline = payload.get("headline", payload)
    weakest = payload.get("weakest_5_classes", [])
    min_rec = min((c["recall"] for c in weakest), default=payload.get("min_recall"))
    min_cls = None
    min_n = None
    if weakest:
        worst = min(weakest, key=lambda c: c["recall"])
        min_cls = worst.get("class_name")
        min_n = worst.get("support")
    bg = payload.get("masked_background", {})
    cal = payload.get("calibration", {})
    return GateMetrics(
        macro_f1=float(headline.get("macro_f1", payload["macro_f1"])),
        within_crop_macro_f1=payload.get("within_crop_macro_f1_excluding_single_class_crops"),
        min_recall=float(min_rec),
        bg_drop=float(bg.get("bg_drop", payload["bg_drop"])),
        ece=float(cal.get("ece_before", payload.get("ece"))),
        min_recall_class=min_cls,
        min_recall_n=min_n,
    )


def _load_champion_metrics(path: Path | None) -> GateMetrics | None:
    if path is None or not path.exists():
        return None
    return _metrics_from_payload(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-metrics",
        default="reports/gate_baselines_val.json",
        help="JSON with VAL gate metrics for the candidate",
    )
    parser.add_argument(
        "--champion-metrics",
        default="reports/gate_baselines_val.json",
        help="JSON with VAL gate metrics for the current champion (omit/missing = first-model)",
    )
    parser.add_argument(
        "--candidate-name",
        default=None,
        help="Select a named candidate row when candidate-metrics contains multiple runs",
    )
    parser.add_argument(
        "--candidate-is-champion-baseline",
        action="store_true",
        help="Force first-model bootstrap even if champion-metrics exists",
    )
    parser.add_argument("--version", default=None, help="MLflow candidate version (avoid in dry-run)")
    parser.add_argument("--champion-version", default=None, help="Informational current champion version")
    parser.add_argument(
        "--champion-run-id",
        default=None,
        help="Select the champion row by run ID in a multi-run comparison JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the gate verdict without moving @champion (default behavior)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Attempt to move @champion (refused while thresholds are provisional)",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    cand_path = PROJECT_ROOT / args.candidate_metrics
    if not cand_path.exists():
        log.error(f"Candidate metrics not found: {cand_path}")
        sys.exit(1)

    candidate_payload = json.loads(cand_path.read_text(encoding="utf-8"))
    if "results" in candidate_payload and args.candidate_name:
        candidate_payload = next(
            row for row in candidate_payload["results"]
            if row.get("name") == args.candidate_name
        )
        candidate_payload = {"results": [candidate_payload]}
    candidate = _metrics_from_payload(candidate_payload)
    champ_path = PROJECT_ROOT / args.champion_metrics if args.champion_metrics else None
    champion = None
    if not args.candidate_is_champion_baseline:
        if champ_path is not None and champ_path.exists() and champ_path.resolve() != cand_path.resolve():
            champion_payload = json.loads(champ_path.read_text(encoding="utf-8"))
            if "results" in champion_payload:
                champion_row = next(
                    row for row in champion_payload["results"]
                    if row.get("run_id") == args.champion_run_id
                    or row.get("name") == args.champion_name
                )
                champion_payload = {"results": [champion_row]}
            champion = _metrics_from_payload(champion_payload)

    thresholds = load_gate_thresholds()
    verdict = evaluate_gate(candidate, champion, thresholds)

    cfg = load_config()
    tags = default_tags({
        "gate_verdict": verdict["verdict"],
        "val_macro_f1": f"{candidate.macro_f1:.4f}",
        "val_min_recall": f"{candidate.min_recall:.4f}",
        "val_bg_drop": f"{candidate.bg_drop:.4f}",
        "val_ece": f"{candidate.ece:.5f}",
        "robustness_metrics": (
            f"val_bg_drop={candidate.bg_drop:.4f};val_ece={candidate.ece:.5f};"
            f"val_min_recall={candidate.min_recall:.4f}"
        ),
        "gate_provisional": str(verdict["any_provisional"]).lower(),
    })

    record = {
        "action": "promote",
        "dry_run": dry_run,
        "model": cfg.mlflow.registered_model_name,
        "alias": cfg.mlflow.champion_alias,
        "version": args.version,
        "champion_version": args.champion_version,
        "tags": tags,
        "verdict": verdict,
        "alias_moved": False,
        "notes": (
            "Thresholds are provisional; @champion was not moved."
            if verdict["any_provisional"]
            else ""
        ),
    }

    if args.apply and verdict["any_provisional"]:
        log.error(
            "Refusing --apply: one or more gate thresholds are still provisional. "
            "Re-run without --apply (dry-run)."
        )
        record["alias_moved"] = False
        record["refuse_reason"] = "provisional_thresholds"
        append_promotion_log(record)
        print(json.dumps(record, indent=2))
        return 2

    if args.apply and verdict["all_passed"] and args.version:
        client = get_client()
        champion_path = PROJECT_ROOT / "models" / "best_model.pth"
        digest = hashlib.md5(usedforsecurity=False)
        with champion_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        try:
            version_obj = client.get_model_version(
                cfg.mlflow.registered_model_name, str(args.version)
            )
        except Exception as exc:
            log.error(f"Could not resolve candidate version before promotion: {exc}")
            record["alias_moved"] = False
            record["refuse_reason"] = "candidate_version_not_found"
            append_promotion_log(record)
            print(json.dumps(record, indent=2))
            return 2
        if str(version_obj.run_id) != str(candidate_payload.get("run_id")):
            log.error("Candidate registry version run ID does not match the evaluated report.")
            record["alias_moved"] = False
            record["refuse_reason"] = "candidate_run_id_mismatch"
            append_promotion_log(record)
            print(json.dumps(record, indent=2))
            return 2
        manifest = {
            "version": str(args.version),
            "run_id": candidate_payload.get("run_id"),
            "checkpoint_md5": digest.hexdigest(),
        }
        move = promote_alias(str(args.version), dry_run=False, client=client)
        record["alias_moved"] = move["moved"]
        if move["moved"]:
            set_version_tags(cfg.mlflow.registered_model_name, str(args.version), tags, client)
            (PROJECT_ROOT / "models" / "champion_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
    elif args.apply and verdict["all_passed"]:
        log.error("Refusing --apply without --version; registry version is required for manifest.")
        record["alias_moved"] = False
        record["refuse_reason"] = "missing_registry_version"
        append_promotion_log(record)
        print(json.dumps(record, indent=2))
        return 2
    else:
        log.info("[dry-run] gate verdict logged; alias not moved")
        if args.version:
            try:
                client = get_client()
                set_version_tags(cfg.mlflow.registered_model_name, str(args.version), tags, client)
            except Exception as exc:
                log.warning(f"Could not write MLflow tags in dry-run: {exc}")

    append_promotion_log(record)
    log.info(f"Decision: {verdict['verdict']} failures={verdict['failures']}")
    print(json.dumps(record, indent=2))
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
