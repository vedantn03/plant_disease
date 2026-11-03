"""
src/gates.py
============
4-criterion promotion gate. Thresholds are loaded from
configs/gates/gate_thresholds.yaml and are marked provisional until approved.

Relative criteria compare a candidate to the current champion on VAL.
If there is no champion (first-model bootstrap), the candidate is compared
to itself, so relative rules pass by construction. Absolute floors still apply.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """
    Wilson score interval lower bound for a binomial proportion.

    Used as a proposed (not yet approved) floor for tiny-class recall.
    """
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    adj = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n) ** 0.5
    return float(max(0.0, (centre - adj) / denom))


def one_extra_error_recall(n: int, current_errors: int = 0) -> dict[str, float]:
    """Show the effect of one extra false negative on recall for support n."""
    if n <= 0:
        return {"n": 0.0, "recall_now": 0.0, "recall_one_extra_error": 0.0, "drop": 0.0}
    tp_now = max(0, n - current_errors)
    tp_extra = max(0, n - current_errors - 1)
    rec_now = tp_now / n
    rec_extra = tp_extra / n
    return {
        "n": n,
        "current_errors": current_errors,
        "recall_now": round(rec_now, 4),
        "recall_one_extra_error": round(rec_extra, 4),
        "drop": round(rec_now - rec_extra, 4),
        "wilson_lower_now": round(wilson_lower_bound(tp_now, n), 4),
        "wilson_lower_one_extra": round(wilson_lower_bound(tp_extra, n), 4),
    }


def load_gate_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = PROJECT_ROOT / "configs" / "gates" / "gate_thresholds.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def thresholds_sha256(path: str | Path | None = None) -> str:
    """Hash the exact gate YAML bytes used for approval."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "gates" / "gate_thresholds.yaml"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_threshold_approval(
    approval_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
    *,
    require_committed: bool = False,
    project_root: str | Path | None = None,
) -> tuple[bool, str, str]:
    """Require explicit approval metadata and a hash of the current threshold file."""
    if approval_path is None:
        approval_path = PROJECT_ROOT / "configs" / "gates" / "approval.yaml"
    approval = yaml.safe_load(Path(approval_path).read_text(encoding="utf-8")) or {}
    current_hash = thresholds_sha256(thresholds_path)
    if approval.get("approved") is not True:
        return False, "approval.yaml does not set approved: true", current_hash
    if not str(approval.get("approved_by", "")).strip():
        return False, "approval.yaml is missing approved_by", current_hash
    if not str(approval.get("approved_on", "")).strip():
        return False, "approval.yaml is missing approved_on", current_hash
    recorded_hash = str(approval.get("thresholds_sha256", "")).lower()
    if recorded_hash != current_hash:
        return (
            False,
            "thresholds_sha256 does not match the current gate_thresholds.yaml",
            current_hash,
        )
    if require_committed:
        root = Path(project_root) if project_root else PROJECT_ROOT
        for path in (Path(approval_path), Path(thresholds_path)):
            try:
                relative_path = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                return False, f"Approval input is outside the Git project: {path}", current_hash
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative_path],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if tracked.returncode != 0:
                return (
                    False,
                    f"Approval input must be tracked and committed: {relative_path}",
                    current_hash,
                )
            clean = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", relative_path],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if clean.returncode != 0:
                return (
                    False,
                    f"Approval input has uncommitted changes: {relative_path}",
                    current_hash,
                )
    return True, "approved threshold hash matches", current_hash


@dataclass
class GateMetrics:
    """VAL-split metrics used by the gate. None means 'not supplied'."""

    macro_f1: float
    min_recall: float
    bg_drop: float
    ece: float
    within_crop_macro_f1: float | None = None
    bg_drop_per_image_mean: float | None = None
    min_recall_class: str | None = None
    min_recall_n: int | None = None
    min_recall_wilson_lower: float | None = None
    min_recall_correct: int | None = None
    per_class_recall: dict[str, dict[str, float | int]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionResult:
    name: str
    passed: bool
    provisional: bool
    observed: float
    threshold: float
    reason: str


def evaluate_gate(
    candidate: GateMetrics,
    champion: GateMetrics | None,
    thresholds: dict[str, Any] | None = None,
    *,
    approved: bool = False,
) -> dict[str, Any]:
    """
    Evaluate the four criteria.

    First-model case: champion is None → relative macro-F1 uses the candidate
    as its own baseline (always passes the relative rule).
    """
    if thresholds is None:
        thresholds = load_gate_thresholds()

    criteria_cfg = thresholds["criteria"]
    first_model = champion is None
    candidate_f1 = candidate.within_crop_macro_f1 or candidate.macro_f1
    champion_f1 = None if champion is None else (
        champion.within_crop_macro_f1 or champion.macro_f1
    )
    champ_f1 = candidate_f1 if first_model else champion_f1

    results: list[CriterionResult] = []

    # 1. Within-crop macro-F1 vs champion - margin
    f1_cfg = criteria_cfg["within_crop_macro_f1"]
    f1_margin = float(f1_cfg["margin"])
    f1_floor = champ_f1 - f1_margin
    f1_pass = candidate_f1 >= f1_floor
    f1_reason = (
        f"VAL within-crop macro-F1 {candidate_f1:.4f} vs floor {f1_floor:.4f} "
        f"(champion {champ_f1:.4f} - margin {f1_margin:.4f}"
        f"{'; first-model bootstrap' if first_model else ''})"
    )
    results.append(CriterionResult(
        name="within_crop_macro_f1",
        passed=f1_pass,
        provisional=not approved,
        observed=candidate_f1,
        threshold=f1_floor,
        reason=f1_reason if f1_pass else f"FAIL: {f1_reason}",
    ))

    # 2. Minimum class recall, using Wilson lower bounds for small supports.
    rec_cfg = criteria_cfg["minimum_class_recall"]
    rec_floor = float(rec_cfg["floor"])
    support_cutoff = int(rec_cfg["support_cutoff"])
    wilson_z = float(rec_cfg["wilson_z"])
    recall_scores: list[tuple[str, float, int, bool]] = []
    for class_name, details in candidate.per_class_recall.items():
        n = int(details.get("n", 0))
        correct = int(details.get("correct", 0))
        recall = float(details.get("recall", correct / n if n else 0.0))
        tiny = n < support_cutoff
        score = wilson_lower_bound(correct, n, z=wilson_z) if tiny else recall
        recall_scores.append((class_name, score, n, tiny))
    if recall_scores:
        limiting_class, recall_score, limiting_n, used_wilson = min(
            recall_scores, key=lambda item: item[1]
        )
    else:
        limiting_class = candidate.min_recall_class or "unknown"
        limiting_n = int(candidate.min_recall_n or 0)
        used_wilson = limiting_n < support_cutoff
        if used_wilson and candidate.min_recall_correct is not None:
            recall_score = wilson_lower_bound(
                candidate.min_recall_correct, limiting_n, z=wilson_z
            )
        elif used_wilson and candidate.min_recall_wilson_lower is not None:
            recall_score = candidate.min_recall_wilson_lower
        else:
            recall_score = candidate.min_recall
    rec_pass = recall_score >= rec_floor
    rec_reason = (
        f"VAL recall floor score {recall_score:.4f} (class={limiting_class}, n={limiting_n}, "
        f"wilson_95_used={used_wilson}) vs floor {rec_floor:.4f}"
    )
    results.append(CriterionResult(
        name="minimum_class_recall",
        passed=rec_pass,
        provisional=not approved,
        observed=recall_score,
        threshold=rec_floor,
        reason=rec_reason if rec_pass else f"FAIL: {rec_reason}",
    ))

    # 3. Background sensitivity uses only per-image-mean fill.
    bg_cfg = criteria_cfg["bg_drop_per_image_mean"]
    bg_margin = float(bg_cfg["margin"])
    candidate_bg_drop = (
        candidate.bg_drop_per_image_mean
        if candidate.bg_drop_per_image_mean is not None else candidate.bg_drop
    )
    champion_bg_drop = (
        champion.bg_drop_per_image_mean
        if champion and champion.bg_drop_per_image_mean is not None
        else champion.bg_drop if champion else None
    )
    bg_ceiling = (
        candidate_bg_drop
        if first_model
        else round(champion_bg_drop + bg_margin, 10)
    )
    bg_pass = first_model or candidate_bg_drop <= bg_ceiling
    bg_reason = (
        f"VAL per-image-mean bg_drop {candidate_bg_drop:.4f} "
        f"vs champion-relative ceiling {bg_ceiling:.4f} (margin {bg_margin:.4f}); "
        "black and mid-grey fills are informational"
    )
    results.append(CriterionResult(
        name="bg_drop_per_image_mean",
        passed=bg_pass,
        provisional=not approved,
        observed=candidate_bg_drop,
        threshold=bg_ceiling,
        reason=bg_reason if bg_pass else f"FAIL: {bg_reason}",
    ))

    # 4. ECE ceiling
    ece_cfg = criteria_cfg["ece"]
    ece_max = float(ece_cfg["maximum"])
    ece_pass = candidate.ece <= ece_max
    ece_reason = f"VAL ECE {candidate.ece:.5f} vs max {ece_max:.4f}"
    results.append(CriterionResult(
        name="ece",
        passed=ece_pass,
        provisional=not approved,
        observed=candidate.ece,
        threshold=ece_max,
        reason=ece_reason if ece_pass else f"FAIL: {ece_reason}",
    ))

    any_provisional = any(r.provisional for r in results)
    all_pass = all(r.passed for r in results)
    failures = [r.name for r in results if not r.passed]

    return {
        "first_model": first_model,
        "all_passed": all_pass,
        "failures": failures,
        "any_provisional": any_provisional,
        "verdict": "PASS" if all_pass else "REJECT",
        "champion_macro_f1": champ_f1,
        "criteria": [
            {
                "name": r.name,
                "passed": r.passed,
                "provisional": r.provisional,
                "observed": r.observed,
                "threshold": r.threshold,
                "reason": r.reason,
            }
            for r in results
        ],
    }
