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
  python scripts/promote.py --apply   # refused while criteria are provisional
"""

from __future__ import annotations

import argparse
import json
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
        "--candidate-is-champion-baseline",
        action="store_true",
        help="Force first-model bootstrap even if champion-metrics exists",
    )
    parser.add_argument("--version", default=None, help="MLflow model version to tag / alias")
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

    candidate = _metrics_from_payload(json.loads(cand_path.read_text(encoding="utf-8")))
    champ_path = PROJECT_ROOT / args.champion_metrics if args.champion_metrics else None
    champion = None
    if not args.candidate_is_champion_baseline:
        if champ_path is not None and champ_path.exists() and champ_path.resolve() != cand_path.resolve():
            champion = _load_champion_metrics(champ_path)

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
        set_version_tags(cfg.mlflow.registered_model_name, str(args.version), tags, client)
        move = promote_alias(str(args.version), dry_run=False, client=client)
        record["alias_moved"] = move["moved"]
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
