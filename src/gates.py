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


@dataclass
class GateMetrics:
    """VAL-split metrics used by the gate. None means 'not supplied'."""

    macro_f1: float
    min_recall: float
    bg_drop: float
    ece: float
    min_recall_class: str | None = None
    min_recall_n: int | None = None
    min_recall_wilson_lower: float | None = None
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
    champ_f1 = candidate.macro_f1 if first_model else champion.macro_f1

    results: list[CriterionResult] = []

    # 1. Macro-F1 vs champion - margin
    f1_cfg = criteria_cfg["macro_f1"]
    f1_margin = float(f1_cfg["margin"])
    f1_floor = champ_f1 - f1_margin
    f1_pass = candidate.macro_f1 >= f1_floor
    f1_reason = (
        f"VAL macro-F1 {candidate.macro_f1:.4f} vs floor {f1_floor:.4f} "
        f"(champion {champ_f1:.4f} - margin {f1_margin:.4f}"
        f"{'; first-model bootstrap' if first_model else ''})"
    )
    results.append(CriterionResult(
        name="macro_f1",
        passed=f1_pass,
        provisional=bool(f1_cfg.get("provisional", True)),
        observed=candidate.macro_f1,
        threshold=f1_floor,
        reason=f1_reason if f1_pass else f"FAIL: {f1_reason}",
    ))

    # 2. Per-class minimum recall (point estimate; tiny-class rule pending)
    rec_cfg = criteria_cfg["min_recall"]
    rec_floor = float(rec_cfg["floor"])
    rec_pass = candidate.min_recall >= rec_floor
    rec_reason = (
        f"VAL min-recall {candidate.min_recall:.4f} "
        f"(class={candidate.min_recall_class}, n={candidate.min_recall_n}) "
        f"vs floor {rec_floor:.4f}; tiny_class_rule={rec_cfg.get('tiny_class_rule')}"
    )
    results.append(CriterionResult(
        name="min_recall",
        passed=rec_pass,
        provisional=bool(rec_cfg.get("provisional", True)),
        observed=candidate.min_recall,
        threshold=rec_floor,
        reason=rec_reason if rec_pass else f"FAIL: {rec_reason}",
    ))

    # 3. Background-mask drop (upper bound on shortcut + masking artifacts)
    bg_cfg = criteria_cfg["bg_drop"]
    bg_max = float(bg_cfg["max_drop"])
    bg_pass = candidate.bg_drop <= bg_max
    bg_reason = (
        f"VAL bg_drop {candidate.bg_drop:.4f} vs max {bg_max:.4f} "
        "(drop mixes background reliance with masking artifacts)"
    )
    results.append(CriterionResult(
        name="bg_drop",
        passed=bg_pass,
        provisional=bool(bg_cfg.get("provisional", True)),
        observed=candidate.bg_drop,
        threshold=bg_max,
        reason=bg_reason if bg_pass else f"FAIL: {bg_reason}",
    ))

    # 4. ECE ceiling
    ece_cfg = criteria_cfg["ece"]
    ece_max = float(ece_cfg["max_ece"])
    ece_pass = candidate.ece <= ece_max
    ece_reason = f"VAL ECE {candidate.ece:.5f} vs max {ece_max:.4f}"
    results.append(CriterionResult(
        name="ece",
        passed=ece_pass,
        provisional=bool(ece_cfg.get("provisional", True)),
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
