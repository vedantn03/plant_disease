"""Compare a candidate checkpoint with the champion on the VAL promotion metrics.

This is a read-only evaluation. It never writes model weights, changes MLflow,
or moves an alias.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.dataset import PlantVillageDataset, get_transforms
from src.model import build_model
from src.robustness import evaluate_mask_fill_protocol
from src.utils import parse_class_name


def expected_calibration_error(
    logits: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute the same confidence-bin ECE used by the evaluation scripts."""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracy = predictions == labels
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            value += float(
                np.abs(accuracy[in_bin].mean() - confidence[in_bin].mean())
                * in_bin.mean()
            )
    return float(value)


def _infer(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def _within_crop_f1(labels: np.ndarray, predictions: np.ndarray, idx_to_class: dict[int, str]) -> float:
    single_class_crops = {"blueberry", "orange", "raspberry", "soybean", "squash"}
    class_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(len(idx_to_class))),
        average=None,
        zero_division=0,
    )
    values = []
    for class_index in range(len(idx_to_class)):
        crop, _ = parse_class_name(idx_to_class[class_index])
        if crop in single_class_crops:
            continue
        values.append(float(class_f1[class_index]))
    return float(np.mean(values)) if values else 0.0


def _class_metrics(labels: np.ndarray, predictions: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, Any]:
    recalls = []
    for class_index in range(len(idx_to_class)):
        selected = labels == class_index
        support = int(selected.sum())
        recall = float((predictions[selected] == class_index).mean()) if support else 0.0
        recalls.append((recall, idx_to_class[class_index], support))
    minimum = min(recalls, key=lambda item: (item[0], -item[2]))
    return {
        "value": round(minimum[0], 6),
        "class": minimum[1],
        "n": minimum[2],
        "all": {
            name: {"recall": round(recall, 6), "n": support}
            for recall, name, support in recalls
        },
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    name: str,
    cfg: Any,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    model = build_model(len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_path = PROJECT_ROOT / cfg.split.manifest_dir / "val.csv"
    dataset = PlantVillageDataset(
        split_path,
        transform=get_transforms("val", cfg),
        class_to_idx=class_to_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    logits, labels = _infer(model, loader, device)
    predictions = logits.argmax(axis=1)
    clean_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    masked = evaluate_mask_fill_protocol(
        model,
        dataset.samples,
        device,
        image_size=cfg.data.image_size,
        batch_size=32,
    )
    return {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "run_id": checkpoint.get("run_id"),
        "val_n": int(len(labels)),
        "clean_macro_f1": round(clean_f1, 6),
        "within_crop_macro_f1": round(_within_crop_f1(labels, predictions, idx_to_class), 6),
        "min_recall": _class_metrics(labels, predictions, idx_to_class),
        "ece": round(expected_calibration_error(logits, labels), 6),
        "masked_background_by_fill": masked,
        "inference_seconds": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", default="reports/candidate_gate_comparison.json", type=Path)
    args = parser.parse_args()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    champion_path = PROJECT_ROOT / "models" / "best_model.pth"
    started = time.perf_counter()
    results = [
        evaluate_checkpoint(champion_path, "champion_v2", cfg, device),
        evaluate_checkpoint(args.candidate, "candidate", cfg, device),
    ]
    elapsed = round(time.perf_counter() - started, 2)
    payload = {
        "read_only": True,
        "device": str(device),
        "same_preprocessing": "get_transforms('val', cfg): Resize(256)+CenterCrop(224)+ImageNet normalize",
        "shared_robustness_implementation": "src.robustness.py",
        "results": results,
        "elapsed_seconds": elapsed,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
