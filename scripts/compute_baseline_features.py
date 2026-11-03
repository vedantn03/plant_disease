"""Create a seeded, TRAIN-split image-quality baseline for drift monitoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.image_features import blur_score, image_brightness
from src.calibration import probs_from_logits
from src.model import build_model, preprocess_image
from src.model_loader import file_md5


def _data_hash(manifest_path: Path) -> str:
    if not manifest_path.exists():
        raise FileNotFoundError(f"DVC split hash manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def _feature_distribution(values: list[float], bins: int, epsilon: float) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0 or not np.isfinite(values_array).all():
        raise ValueError("Baseline feature values must be non-empty and finite")
    quantile_levels = np.linspace(0.0, 1.0, bins + 1)
    cut_points = np.unique(np.quantile(values_array, quantile_levels[1:-1])).tolist()
    histogram_edges = [-np.inf, *cut_points, np.inf]
    counts, _ = np.histogram(values_array, bins=histogram_edges)
    quantiles = np.quantile(values_array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "summary": {
            "min": float(values_array.min()),
            "max": float(values_array.max()),
            "mean": float(values_array.mean()),
            "std": float(values_array.std()),
            "quantiles": {
                "0": float(quantiles[0]),
                "0.25": float(quantiles[1]),
                "0.5": float(quantiles[2]),
                "0.75": float(quantiles[3]),
                "1": float(quantiles[4]),
            },
        },
        "quantile_bin_edges": cut_points,
        "bin_counts": [int(value) for value in counts],
        "reference_values": [float(value) for value in values_array],
        "psi_epsilon": epsilon,
    }


def build_baseline_payload(
    feature_values: dict[str, list[float]],
    *,
    requested_sample_size: int,
    seed: int,
    dvc_data_hash: str,
    psi_bins: int,
    psi_epsilon: float,
    checkpoint_md5: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    sizes = {len(values) for values in feature_values.values()}
    if not sizes or len(sizes) != 1 or next(iter(sizes)) == 0:
        raise ValueError("Brightness and blur baseline arrays must have one equal non-zero size")
    sample_size = next(iter(sizes))
    return {
        "kind": "train_split_feature_baseline",
        "sample_size_requested": requested_sample_size,
        "sample_size": sample_size,
        "seed": seed,
        "dvc_data_hash": dvc_data_hash,
        "checkpoint_md5": checkpoint_md5,
        "psi_bins_requested": psi_bins,
        "elapsed_seconds": elapsed_seconds,
        "features": {
            name: _feature_distribution(values, psi_bins, psi_epsilon)
            for name, values in feature_values.items()
        },
    }


def create_baseline(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    cfg = load_config(project_root / "configs" / "config.yaml", require_data_root=False)
    drift_cfg = cfg.drift
    requested_sample_size = int(drift_cfg.baseline_sample_size)
    seed = int(cfg.seed)
    split_path = project_root / "data" / "splits" / "train.csv"
    hash_path = project_root / "data" / "splits" / "manifest_hashes.json"
    if not split_path.exists():
        raise FileNotFoundError(f"TRAIN split manifest not found: {split_path}")
    if requested_sample_size <= 0:
        raise ValueError("drift.baseline_sample_size must be positive")

    with split_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("TRAIN split manifest contains no records")
    selected = random.Random(seed).sample(records, min(requested_sample_size, len(records)))
    data_root = Path(str(cfg.data.root)) if str(cfg.data.root).strip() else None
    checkpoint_path = project_root / "models" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Champion checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    model = build_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    brightness_values: list[float] = []
    blur_values: list[float] = []
    confidence_values: list[float] = []
    started = time.perf_counter()
    for record in selected:
        image_path = Path(record["path"])
        if not image_path.is_absolute():
            if data_root is None:
                raise ValueError("Relative TRAIN image paths require DATA_ROOT to be configured")
            image_path = data_root / image_path
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            brightness_values.append(image_brightness(rgb_image))
            blur_values.append(blur_score(rgb_image))
            input_tensor = preprocess_image(rgb_image, int(cfg.data.image_size)).to(device)
            with torch.inference_mode():
                logits = model(input_tensor)
            probabilities = probs_from_logits(logits, temperature=1.0).squeeze()
            confidence_values.append(float(probabilities.max().item()))
    elapsed_seconds = time.perf_counter() - started

    payload = build_baseline_payload(
        {
            "brightness": brightness_values,
            "blur": blur_values,
            "confidence": confidence_values,
        },
        requested_sample_size=requested_sample_size,
        seed=seed,
        dvc_data_hash=_data_hash(hash_path),
        psi_bins=int(drift_cfg.psi_bins),
        psi_epsilon=float(drift_cfg.psi_epsilon),
        checkpoint_md5=file_md5(checkpoint_path),
        elapsed_seconds=elapsed_seconds,
    )
    output_path = project_root / str(drift_cfg.baseline_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"output": str(output_path), "sample_size": payload["sample_size"],
            "elapsed_seconds": elapsed_seconds}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(json.dumps(create_baseline(), indent=2))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
