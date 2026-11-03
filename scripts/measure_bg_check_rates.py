"""Measure background sensitivity outcomes on a seeded sample of VAL images."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.background_check import run_background_sensitivity
from src.calibration import probs_from_logits
from src.config_loader import load_config
from src.model import build_model, preprocess_image


def _read_val_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def measure_rates(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    cfg = load_config(project_root / "configs" / "config.yaml", require_data_root=False)
    bg_cfg = cfg.background_check
    requested_size = int(bg_cfg.val_sample_size)
    if requested_size <= 0:
        raise ValueError("background_check.val_sample_size must be positive")

    val_path = project_root / str(cfg.split.manifest_dir) / "val.csv"
    records = _read_val_records(val_path)
    if not records:
        raise ValueError(f"VAL manifest contains no records: {val_path}")
    sample = random.Random(int(cfg.seed)).sample(records, min(requested_size, len(records)))

    checkpoint_path = project_root / "models" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Champion checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    model = build_model(num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_root = Path(str(cfg.data.root)) if str(cfg.data.root).strip() else None
    counts: Counter[str] = Counter()
    added_latency_ms: list[float] = []
    started = time.perf_counter()
    for record in sample:
        image_path = Path(record["path"])
        if not image_path.is_absolute():
            image_path = (data_root / image_path) if data_root else (project_root / image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        tensor = preprocess_image(image, int(cfg.data.image_size)).to(device)
        device_type = device.type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            logits = model(tensor)
        probabilities = probs_from_logits(logits, temperature=1.0).squeeze()
        top_index = int(probabilities.argmax().item())
        result = run_background_sensitivity(
            image,
            model,
            device=device,
            image_size=int(cfg.data.image_size),
            idx_to_class=idx_to_class,
            original_label=idx_to_class[top_index],
            original_confidence=float(probabilities[top_index].item()),
            min_mask_area_fraction=float(bg_cfg.min_mask_area_fraction),
            max_mask_area_fraction=float(bg_cfg.max_mask_area_fraction),
            enabled=bool(bg_cfg.bg_check_enabled),
        )
        counts[result["result"]] += 1
        added_latency_ms.append(float(result["elapsed_ms"]))

    elapsed_seconds = time.perf_counter() - started
    sample_count = len(sample)
    shares = {
        result: counts[result] / sample_count
        for result in ("consistent", "inconsistent", "unknown")
    }
    report = {
        "split": "VAL",
        "seed": int(cfg.seed),
        "sample_size_requested": requested_size,
        "sample_size": sample_count,
        "shares": shares,
        "counts": {result: counts[result] for result in shares},
        "mean_added_latency_ms": (
            sum(added_latency_ms) / len(added_latency_ms) if added_latency_ms else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "background_check_enabled": bool(bg_cfg.bg_check_enabled),
        "mask_area_bounds": {
            "min": float(bg_cfg.min_mask_area_fraction),
            "max": float(bg_cfg.max_mask_area_fraction),
        },
        "thresholds_provisional": True,
    }
    output_path = project_root / str(bg_cfg.rates_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output": str(output_path), **report}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(json.dumps(measure_rates(), indent=2))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
