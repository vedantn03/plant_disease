"""Summarize saved VAL candidate gate reports without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = {
    "champion": Path("reports/candidate_gate_comparison.json"),
    "seed_43": Path("reports/candidate_gate_48ecf14d.json"),
    "seed_44": Path("reports/candidate_gate_0e2c51ab.json"),
}


def _candidate(report: dict[str, Any], label: str) -> dict[str, Any]:
    rows = report["results"]
    if label == "champion":
        return next(
            row for row in rows
            if row.get("name") == "champion_v2"
            or row.get("run_id") == "fb731412d82544e0abc231b8d7c62f95"
        )
    expected = "48ecf14da2d84803a36d6fd108bd6b61" if label == "seed_43" else "0e2c51abf0834c248e03c5aef774e291"
    return next(row for row in rows if row.get("run_id") == expected)


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    fills = row["masked_background_by_fill"]
    return {
        "clean_macro_f1": float(row["clean_macro_f1"]),
        "within_crop_macro_f1": float(row["within_crop_macro_f1"]),
        "min_recall": float(row["min_recall"]["value"]),
        "ece": float(row["ece"]),
        **{f"bg_drop_{key}": float(value["bg_drop"]) for key, value in fills.items()},
    }


def summarize(paths: dict[str, Path], champion_run_id: str) -> tuple[dict[str, Any], str]:
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    champion_rows = reports["champion"]["results"]
    champion = next(
        (row for row in champion_rows if row.get("run_id") == champion_run_id),
        champion_rows[0],
    )
    rows = {
        "champion": champion,
        "seed_43": _candidate(reports["seed_43"], "seed_43"),
        "seed_44": _candidate(reports["seed_44"], "seed_44"),
    }
    metric_values = {name: _metrics(row) for name, row in rows.items()}
    metrics = {}
    for metric in metric_values["champion"]:
        vals = [metric_values[key][metric] for key in ("champion", "seed_43", "seed_44")]
        metrics[metric] = {
            "champion": vals[0], "seed_43": vals[1], "seed_44": vals[2],
            "mean": sum(vals) / len(vals), "range": max(vals) - min(vals),
        }
    candidate_classes = [
        row["min_recall"]["class"] for key, row in rows.items() if key != "champion"
    ]
    driving_class = max(set(candidate_classes), key=candidate_classes.count)
    if any(driving_class not in row["min_recall"].get("all", {}) for row in rows.values()):
        raise ValueError(
            "The common candidate minimum-recall class is missing from one model report."
        )
    counts = {}
    for name, row in rows.items():
        all_recall = row["min_recall"].get("all", {})
        detail = all_recall.get(driving_class)
        if detail is None:
            info = row["min_recall"]
            detail = info if info.get("class") == driving_class else {}
        total = int(detail.get("n", 0))
        correct = detail.get("correct")
        if correct is None:
            correct = round(float(detail.get("recall", detail.get("value", 0))) * total)
        counts[name] = {"correct": int(correct), "total": total}
    summary = {
        "split": "VAL",
        "models": {name: {"run_id": rows[name].get("run_id"), **metric_values[name]} for name in rows},
        "metrics": metrics,
        "minimum_recall_driving_class": driving_class,
        "driving_class_counts": counts,
        "notes": [
            "Three seeds are a small sample; summaries are descriptive, not a stable variance estimate.",
            "The earlier 0.8750 (70/80) uses a different denominator from supplied VAL n=77. Current counts are derived from each report; seed 43's supplied 0.883117 is 68/77. Reconcile 70/80 against its source before comparing.",
        ],
    }
    lines = ["# VAL Seed Variance Summary", "", "Descriptive only: three seeds are a small sample.", "",
             "| Metric | Champion v2 | Seed 43 | Seed 44 | Mean | Range |", "|---|---:|---:|---:|---:|---:|"]
    for metric, values in metrics.items():
        lines.append(f"| {metric} | {values['champion']:.6f} | {values['seed_43']:.6f} | {values['seed_44']:.6f} | {values['mean']:.6f} | {values['range']:.6f} |")
    lines += ["", f"Driving class for minimum recall: `{driving_class}`.", "",
              "| Model | Correct / total (VAL) |", "|---|---:|"]
    for name, count in counts.items():
        lines.append(f"| {name} | {count['correct']} / {count['total']} |")
    lines += ["", "The earlier 0.8750 (70/80) has a different denominator from the current VAL reports. "
              "Current champion and seed-43 VAL counts use n=77; seed 43 is 68/77. Use the source report "
              "for the historical 70/80 figure before comparing it; it does not match this VAL support.", ""]
    return summary, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name, default in DEFAULT_REPORTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-report", default=str(default))
    parser.add_argument("--champion-run-id", default="fb731412d82544e0abc231b8d7c62f95")
    parser.add_argument("--json-output", default="reports/seed_variance.json")
    parser.add_argument("--markdown-output", default="reports/seed_variance.md")
    args = parser.parse_args()
    paths = {name: PROJECT_ROOT / getattr(args, f"{name}_report") for name in DEFAULT_REPORTS}
    result, markdown = summarize(paths, args.champion_run_id)
    json_path, md_path = PROJECT_ROOT / args.json_output, PROJECT_ROOT / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {json_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
