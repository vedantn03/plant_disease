"""Convert saved evaluation confusions into API hints without symptom claims."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402


def export_pairs(summary_path: Path, output_path: Path) -> int:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Saved evaluation summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs = summary.get("top_10_confused_pairs")
    if pairs is None:
        raise ValueError("Evaluation summary has no top_10_confused_pairs field")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("true_label", "predicted_label", "count"))
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "true_label": pair["true"],
                "predicted_label": pair["pred"],
                "count": pair["count"],
            })
    return len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports" / "eval_summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(require_data_root=False)
    output = args.output or ROOT / cfg.advisory.confused_pairs_path
    try:
        input_path = args.input if args.input.is_absolute() else ROOT / args.input
        count = export_pairs(input_path, output)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Wrote {count} saved confusion pairs to {output}; no differentiation text was generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
