"""Read-only VAL/TEST three-fill robustness report.

This script reuses the masking implementation in ``src.robustness`` and never
writes model weights or changes configuration.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.model import _IMAGENET_MEAN, _IMAGENET_STD, build_model
from src.robustness import MASK_FILL_PROTOCOL, mask_background
from src.segmentation import segment_leaf
from src.utils import parse_class_name


def summarize_mask_areas(areas: list[float]) -> dict[str, float]:
    """Return descriptive foreground-area statistics for valid masks."""
    if not areas:
        return {"mean": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}
    values = np.asarray(areas, dtype=np.float64)
    return {
        "mean": round(float(np.mean(values)), 6),
        "median": round(float(np.median(values)), 6),
        "p5": round(float(np.percentile(values, 5)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
    }


def _read_samples(path: Path, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [(row["path"], class_to_idx[row["class_name"]]) for row in rows]


def _crop_for_label(label: int, idx_to_class: dict[int, str]) -> str:
    try:
        return parse_class_name(idx_to_class[label])[0]
    except (KeyError, ValueError):
        return "unknown"


def _transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _predict(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    labels: list[int],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    if not images:
        return [], []
    with torch.no_grad():
        predictions = model(torch.stack(images).to(device)).argmax(dim=1).cpu().tolist()
    return predictions, labels


def evaluate_split_fill(
    model: torch.nn.Module,
    samples: list[tuple[str, int]],
    idx_to_class: dict[int, str],
    device: torch.device,
    image_size: int,
    batch_size: int,
    fill_name: str,
) -> dict[str, Any]:
    """Evaluate one split/fill and collect crop and mask diagnostics."""
    transform = _transform(image_size)
    clean_preds: list[int] = []
    masked_preds: list[int] = []
    labels: list[int] = []
    clean_batches: list[torch.Tensor] = []
    masked_batches: list[torch.Tensor] = []
    batch_labels: list[int] = []
    areas: list[float] = []
    failures = 0
    failure_examples: list[str] = []

    def flush() -> None:
        nonlocal clean_batches, masked_batches, batch_labels
        pred, truth = _predict(model, clean_batches, batch_labels, device)
        clean_preds.extend(pred)
        pred, _ = _predict(model, masked_batches, batch_labels, device)
        masked_preds.extend(pred)
        labels.extend(truth)
        clean_batches = []
        masked_batches = []
        batch_labels = []

    for path, label in samples:
        try:
            image = Image.open(path).convert("RGB")
            mask = segment_leaf(image)
            area = float(mask.mean())
            is_failure = area < 0.02 or area > 0.98
            if is_failure:
                failures += 1
                if len(failure_examples) < 10:
                    failure_examples.append(str(path))
            else:
                areas.append(area)
            masked = mask_background(image, fill=fill_name)
            clean_batches.append(transform(image))
            masked_batches.append(transform(masked))
            batch_labels.append(int(label))
            if len(batch_labels) >= batch_size:
                flush()
        except Exception:
            failures += 1
            if len(failure_examples) < 10:
                failure_examples.append(str(path))
    flush()

    y_true = np.asarray(labels)
    clean = np.asarray(clean_preds)
    masked = np.asarray(masked_preds)
    clean_f1 = float(f1_score(y_true, clean, average="macro", zero_division=0))
    masked_f1 = float(f1_score(y_true, masked, average="macro", zero_division=0))
    per_crop: dict[str, dict[str, float | int]] = {}
    crops = defaultdict(list)
    for index, label in enumerate(y_true.tolist()):
        crops[_crop_for_label(label, idx_to_class)].append(index)
    for crop, indices in sorted(crops.items()):
        idx = np.asarray(indices)
        crop_clean = float(f1_score(y_true[idx], clean[idx], average="macro", zero_division=0))
        crop_masked = float(f1_score(y_true[idx], masked[idx], average="macro", zero_division=0))
        per_crop[crop] = {
            "n": int(len(idx)),
            "clean_macro_f1": round(crop_clean, 6),
            "masked_macro_f1": round(crop_masked, 6),
            "bg_drop": round(crop_clean - crop_masked, 6),
        }
    return {
        "n": int(len(y_true)),
        "fill": fill_name,
        "clean_macro_f1": round(clean_f1, 6),
        "masked_macro_f1": round(masked_f1, 6),
        "bg_drop": round(clean_f1 - masked_f1, 6),
        "retention_rate": round(masked_f1 / clean_f1, 6) if clean_f1 else 0.0,
        "per_crop": per_crop,
        "mask_area": summarize_mask_areas(areas),
        "mask_failure_count": failures,
        "mask_failure_rate": round(failures / len(samples), 6) if samples else 0.0,
        "mask_failure_examples": failure_examples,
        "valid_mask_count": len(areas),
    }


def _source_evidence() -> dict[str, Any]:
    """Record the current implementation path without asserting historical provenance."""
    diagnostics = (PROJECT_ROOT / "scripts" / "compute_val_diagnostics.py").read_text(
        encoding="utf-8"
    ).splitlines()
    robustness = (PROJECT_ROOT / "scripts" / "robustness_eval.py").read_text(
        encoding="utf-8"
    ).splitlines()
    shared = (PROJECT_ROOT / "src" / "robustness.py").read_text(encoding="utf-8").splitlines()

    def matches(lines: list[str], pattern: str) -> list[dict[str, Any]]:
        return [
            {"line": index, "text": line.strip()}
            for index, line in enumerate(lines, 1)
            if re.search(pattern, line)
        ]

    return {
        "shared_masking_module": "src/robustness.py",
        "shared_fill_protocol": matches(shared, r"MASK_FILL_PROTOCOL|mask_background"),
        "diagnostics_test_path": matches(diagnostics, r"test_masked|test_masked_by_fill|val_masked"),
        "diagnostics_black_selection": matches(diagnostics, r'\["black"\]'),
        "robustness_report_input": matches(robustness, r"val_by_fill|test_by_fill"),
        "historical_00663_note": (
            "The supplied historical TEST value 0.0663 is labeled black fill, but "
            "historical artifacts are required to prove the exact old code path."
        ),
    }


def _composition(samples: list[tuple[str, int]], idx_to_class: dict[int, str]) -> dict[str, Any]:
    classes = Counter(idx_to_class[label] for _, label in samples)
    crops = Counter(_crop_for_label(label, idx_to_class) for _, label in samples)
    return {"n": len(samples), "class_counts": dict(sorted(classes.items())), "crop_counts": dict(sorted(crops.items()))}


def main() -> int:
    cfg = load_config()
    split_dir = PROJECT_ROOT / cfg.split.manifest_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "models" / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    model = build_model(len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    splits = {
        "val": _read_samples(split_dir / "val.csv", class_to_idx),
        "test": _read_samples(split_dir / "test.csv", class_to_idx),
    }
    started = time.perf_counter()
    results: dict[str, Any] = {
        "checkpoint": "models/best_model.pth",
        "protocol": {
            "implementation": "src/robustness.py",
            "mask": "Woebbecke ExG segmentation",
            "fills": list(MASK_FILL_PROTOCOL),
            "preprocessing": "Resize(image_size*256/224), CenterCrop(image_size), ImageNet normalize",
        },
        "source_evidence": _source_evidence(),
        "split_composition": {
            name: _composition(samples, idx_to_class) for name, samples in splits.items()
        },
        "splits": {},
    }
    for split_name, samples in splits.items():
        results["splits"][split_name] = {}
        for fill_name in MASK_FILL_PROTOCOL:
            results["splits"][split_name][fill_name] = evaluate_split_fill(
                model, samples, idx_to_class, device, cfg.data.image_size, 32, fill_name
            )
    results["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "robustness_three_fill.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    lines = [
        "# Three-Fill Robustness Report",
        "",
        "Implementation: `src/robustness.py`; masking artifacts are mixed with background reliance.",
        "",
        "| Split | Fill | Clean F1 | Masked F1 | bg_drop | Mask area mean | median | p5 | p95 | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, fills in results["splits"].items():
        for fill_name, value in fills.items():
            area = value["mask_area"]
            lines.append(
                f"| {split_name.upper()} | {fill_name} | {value['clean_macro_f1']:.6f} | "
                f"{value['masked_macro_f1']:.6f} | {value['bg_drop']:.6f} | "
                f"{area['mean']:.6f} | {area['median']:.6f} | {area['p5']:.6f} | "
                f"{area['p95']:.6f} | {value['mask_failure_count']} |"
            )
    lines.extend(["", "## Gap investigation", "", "```json", json.dumps(results["source_evidence"], indent=2), "```"])
    (reports / "robustness_three_fill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "reports": ["reports/robustness_three_fill.json", "reports/robustness_three_fill.md"],
        "elapsed_seconds": results["elapsed_seconds"],
        "status": "complete",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
