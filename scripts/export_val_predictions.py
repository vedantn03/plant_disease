"""Convert saved VAL logits and labels to the confidence-tier input CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402


def export_predictions(
    logits_path: Path,
    labels_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> int:
    """Export argmax, T=1 confidence, and correctness from saved VAL arrays."""
    logits = np.load(logits_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False).astype(np.int64, copy=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("Saved VAL logits and labels have incompatible shapes")
    if logits.shape[1] != len(idx_to_class):
        raise ValueError("VAL logit class count does not match checkpoint class map")
    if any(int(label) not in idx_to_class for label in labels):
        raise ValueError("VAL labels contain an index absent from the checkpoint class map")

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("predicted_class", "confidence", "correct")
        )
        writer.writeheader()
        for true_index, predicted_index, probs in zip(labels, predictions, probabilities):
            writer.writerow({
                "predicted_class": idx_to_class[int(predicted_index)],
                "confidence": float(probs[int(predicted_index)]),
                "correct": int(true_index == predicted_index),
            })
    return len(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(require_data_root=False).advisory
    paths = {
        "logits": args.logits or Path(cfg.val_logits_path),
        "labels": args.labels or Path(cfg.val_labels_path),
        "checkpoint": args.checkpoint or Path(cfg.checkpoint_path),
        "output": args.output or Path(cfg.val_predictions_path),
    }
    paths = {
        key: path if path.is_absolute() else PROJECT_ROOT / path
        for key, path in paths.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file() and path != paths["output"]]
    if missing:
        parser.error("Required saved VAL/checkpoint files are missing: " + ", ".join(missing))
    count = export_predictions(paths["logits"], paths["labels"], paths["checkpoint"], paths["output"])
    print(f"Wrote {count} saved VAL rows to {paths['output']}; no inference or TEST rows were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
