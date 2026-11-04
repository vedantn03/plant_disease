"""Print empirical per-crop precision and coverage options from saved VAL predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402


def derive_table(path: Path, candidate_quantiles: tuple[float, ...] | list[float]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"VAL prediction file not found: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    required = {"predicted_class", "confidence", "correct"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    by_crop: dict[str, list[tuple[float, bool]]] = {}
    for row in rows:
        label = row["predicted_class"]
        crop = label.split("___", 1)[0]
        by_crop.setdefault(crop, []).append(
            (float(row["confidence"]), row["correct"].strip().lower() in {"1", "true", "yes"})
        )
    result = {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": {}}
    for crop, values in sorted(by_crop.items()):
        confidences = np.array([value[0] for value in values], dtype=float)
        correct = np.array([value[1] for value in values], dtype=bool)
        candidates = []
        for q in candidate_quantiles:
            cutoff = float(np.quantile(confidences, q))
            selected = confidences >= cutoff
            candidates.append({
                "candidate_quantile": q,
                "cutoff": cutoff,
                "precision": float(correct[selected].mean()) if selected.any() else None,
                "coverage": float(selected.mean()),
                "accepted": int(selected.sum()),
                "total": int(len(values)),
            })
        result["rows"][crop] = candidates
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/val_predictions.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(require_data_root=False)
    input_path = args.input if args.input != Path("reports/val_predictions.csv") else Path(
        cfg.advisory.val_predictions_path
    )
    output_path = args.output or Path(cfg.advisory.confidence_table_path)
    input_path = input_path if input_path.is_absolute() else PROJECT_ROOT / input_path
    output_path = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    try:
        result = derive_table(input_path, cfg.advisory.candidate_quantiles)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"input: {input_path}")
    print(f"input_sha256: {result['input_sha256']}")
    print(f"table_sha256: {table_hash}")
    print("crop | quantile | candidate cutoff | precision | coverage | accepted/total")
    for crop, values in result["rows"].items():
        for item in values:
            print(
                f"{crop} | {item['candidate_quantile']:.2f} | {item['cutoff']:.6f} | "
                f"{item['precision']:.6f} | {item['coverage']:.6f} | "
                f"{item['accepted']}/{item['total']}"
            )
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
