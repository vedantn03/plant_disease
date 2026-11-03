"""Read-only VAL comparison of a candidate checkpoint against the champion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import compute_ece  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.dataset import PlantVillageDataset, get_transforms  # noqa: E402
from src.model import build_model  # noqa: E402
from src.robustness import evaluate_mask_fill_protocol  # noqa: E402
from src.utils import parse_class_name  # noqa: E402


def _predict(model, dataset, device):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, target in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(target)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def _load_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(checkpoint["class_to_idx"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_checkpoint(name: str, path: Path, dataset, device) -> dict:
    model, checkpoint = _load_model(path, device)
    logits, labels = _predict(model, dataset, device)
    predictions = logits.argmax(axis=1)
    idx_to_class = {value: key for key, value in checkpoint["class_to_idx"].items()}
    single_class_crops = {"blueberry", "raspberry", "soybean", "orange", "squash"}
    per_class_f1 = f1_score(
        labels, predictions, labels=np.arange(logits.shape[1]),
        average=None, zero_division=0,
    )
    within_crop = [
        float(per_class_f1[index])
        for index, class_name in idx_to_class.items()
        if parse_class_name(class_name)[0] not in single_class_crops
    ]
    within_crop_macro_f1 = float(np.mean(within_crop))
    recalls = recall_score(labels, predictions, labels=np.arange(logits.shape[1]),
                          average=None, zero_division=0)
    weakest = int(np.argmin(recalls))
    class_names = {value: key for key, value in checkpoint["class_to_idx"].items()}
    per_fill = evaluate_mask_fill_protocol(
        model, dataset.samples, device, image_size=load_config().data.image_size, batch_size=32
    )
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    per_class_counts = {
        class_name: {
            "recall": round(float(recalls[index]), 6),
            "n": int((labels == index).sum()),
            "correct": int(((labels == index) & (predictions == index)).sum()),
        }
        for index, class_name in idx_to_class.items()
    }
    return {
        "name": name,
        "checkpoint": str(path),
        "run_id": checkpoint.get("run_id"),
        "seed": checkpoint.get("seed"),
        "val_n": int(len(labels)),
        "clean_macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 6),
        "within_crop_macro_f1": round(within_crop_macro_f1, 6),
        "min_recall": {
            "value": round(float(recalls[weakest]), 6),
            "class": class_names[weakest],
            "n": int((labels == weakest).sum()),
            "correct": int(((labels == weakest) & (predictions == weakest)).sum()),
            "all": per_class_counts,
        },
        "ece": round(float(compute_ece(probabilities, labels)), 6),
        "masked_background_by_fill": {
            fill: {"n": int(len(labels)), "fill": fill, **values}
            for fill, values in per_fill.items()
        },
        "inference_seconds": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=None, help="Candidate checkpoint path")
    parser.add_argument("--candidate-run-id", default=None, help="Candidate MLflow run ID")
    parser.add_argument("--seed", default=None, type=int, help="Candidate seed")
    parser.add_argument(
        "--champion-only",
        action="store_true",
        help="Evaluate only the configured champion checkpoint for the combined report",
    )
    parser.add_argument("--output", help="Optional per-candidate JSON output path")
    parser.add_argument(
        "--preserve-comparison",
        action="store_true",
        help="Write only a candidate record to reports/candidate_gate_<run8>.json",
    )
    args = parser.parse_args()
    if args.candidate is None and not args.champion_only:
        args.candidate = "models/candidates/48ecf14da2d84803a36d6fd108bd6b61.pth"
        args.candidate_run_id = "48ecf14da2d84803a36d6fd108bd6b61"
        args.seed = 43
    if not args.champion_only and (args.candidate_run_id is None or args.seed is None):
        parser.error("--candidate-run-id and --seed are required with --candidate")

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = PROJECT_ROOT / cfg.split.manifest_dir
    champion_path = PROJECT_ROOT / "models" / "best_model.pth"
    candidate_path = (PROJECT_ROOT / args.candidate).resolve() if args.candidate else None
    dataset = PlantVillageDataset(split_dir / "val.csv", transform=get_transforms("val", cfg))
    started = time.perf_counter()
    results = []
    if not args.champion_only:
        candidate_result = evaluate_checkpoint(
            f"candidate_seed_{args.seed}", candidate_path, dataset, device
        )
        candidate_result["run_id"] = args.candidate_run_id
        results.append(candidate_result)
    if args.champion_only or not args.preserve_comparison:
        champion_result = evaluate_checkpoint("champion_v2", champion_path, dataset, device)
        champion_result["run_id"] = "fb731412d82544e0abc231b8d7c62f95"
        results.insert(0, champion_result)
    report = {
        "read_only": True,
        "candidate_only": bool(args.preserve_comparison and not args.champion_only),
        "device": str(device),
        "same_preprocessing": "get_transforms('val', cfg): Resize(256)+CenterCrop(224)+ImageNet normalize",
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    default_name = (
        f"candidate_gate_{args.candidate_run_id[:8]}.json"
        if args.preserve_comparison and not args.champion_only
        else "candidate_gate_comparison.json"
    )
    out = PROJECT_ROOT / args.output if args.output else PROJECT_ROOT / "reports" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(PROJECT_ROOT)), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
