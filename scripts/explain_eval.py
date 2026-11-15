"""Measure current-image attribution deletion against equal-area random masking.

This command requires the local champion checkpoint and real VAL/ TRAIN manifests. It never
trains or writes model weights. It writes only the configured explanation report when run.
"""

from __future__ import annotations

import base64
import csv
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.api import _build_model_bundle
from src.config_loader import load_config
from src.explainability import ExplanationUnavailable, _prediction_fn, generate_explanation


def _resolve_manifest_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    return root / path


def _mask_pair(
    image: Image.Image, mask: Image.Image, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if selected.shape != rgb.shape[:2] or not selected.any():
        raise ValueError("Explanation mask is empty or does not match the model input crop.")
    random_mask = np.zeros(selected.size, dtype=bool)
    random_mask[rng.choice(selected.size, size=int(selected.sum()), replace=False)] = True
    random_mask = random_mask.reshape(selected.shape)
    fill = np.mean(rgb, axis=(0, 1)).round().astype(np.uint8)
    top_masked = rgb.copy()
    random_masked = rgb.copy()
    top_masked[selected] = fill
    random_masked[random_mask] = fill
    return top_masked, random_masked


def _confidence_drop(
    predict, original: np.ndarray, top_masked: np.ndarray, random_masked: np.ndarray, class_idx: int
) -> tuple[float, float]:
    batch = predict(np.stack([original, top_masked, random_masked]))
    original_confidence = float(batch[0, class_idx])
    return original_confidence - float(batch[1, class_idx]), original_confidence - float(batch[2, class_idx])


def _rows(manifest: Path, root: Path) -> list[tuple[Path, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"VAL manifest is missing: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        result = []
        for row in csv.DictReader(handle):
            raw_path = row.get("path", "")
            class_name = row.get("class_name", "")
            if raw_path and class_name:
                result.append((_resolve_manifest_path(raw_path, root), class_name))
    result = [(path, label) for path, label in result if path.is_file()]
    if not result:
        raise FileNotFoundError("VAL manifest contains no readable image paths.")
    return result


def run() -> str:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "config.yaml", require_data_root=False)
    val_manifest = root / "data" / "splits" / "val.csv"
    train_manifest = root / cfg.explainability.train_manifest
    images = _rows(val_manifest, root)
    count = min(len(images), int(cfg.explainability.evaluation_max_images))
    rng = np.random.default_rng(int(cfg.explainability.seed))
    selected_indices = rng.choice(len(images), size=count, replace=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = _build_model_bundle(cfg, root, device)
    model = bundle["model"]
    model_version = bundle["model_version"]
    index_by_class = bundle["class_to_idx"]
    class_by_index = bundle["idx_to_class"]
    settings = cfg.explainability
    params = {
        "seed": int(settings.seed),
        "image_size": int(cfg.data.image_size),
        "train_manifest": str(train_manifest),
        "data_root": str(cfg.data.root),
        "lime_num_samples": int(settings.lime.num_samples),
        "lime_num_features": int(settings.lime.num_features),
        "lime_kernel_size": int(settings.lime.kernel_size),
        "lime_max_dist": int(settings.lime.max_dist),
        "lime_ratio": float(settings.lime.ratio),
        "shap_n_background": int(settings.shap.n_background),
        "shap_n_samples": int(settings.shap.n_samples),
        "deletion_region_fraction": float(settings.deletion_region_fraction),
    }
    predict = _prediction_fn(model, device, int(cfg.data.image_size), None, prepared_rgb=True)
    totals = {"lime": {"top": [], "random": [], "elapsed": 0.0}, "shap": {"top": [], "random": [], "elapsed": 0.0}}
    errors: dict[str, str] = {}
    try:
        for sample_index in selected_indices:
            image_path, _label = images[int(sample_index)]
            with Image.open(image_path) as opened:
                original_image = opened.convert("RGB")
            for method in ("lime", "shap"):
                started = time.perf_counter()
                try:
                    output = generate_explanation(
                        method=method,
                        image=original_image,
                        model=model,
                        device=device,
                        idx_to_class=class_by_index,
                        model_version=model_version,
                        params=params,
                    )
                    model_input = Image.open(io.BytesIO(base64.b64decode(
                        output["model_input_png_base64"]
                    ))).convert("RGB")
                    mask = Image.open(io.BytesIO(base64.b64decode(
                        output["attribution_mask_png_base64"]
                    ))).convert("L")
                    top_masked, random_masked = _mask_pair(model_input, mask, rng)
                    original_pixels = np.asarray(model_input, dtype=np.uint8)
                    class_idx = index_by_class[output["predicted_class"]]
                    top_drop, random_drop = _confidence_drop(
                        predict, original_pixels, top_masked, random_masked, class_idx
                    )
                    totals[method]["top"].append(top_drop)
                    totals[method]["random"].append(random_drop)
                    totals[method]["elapsed"] += time.perf_counter() - started
                except Exception as exc:
                    errors.setdefault(method, f"{type(exc).__name__}: {exc}")
    finally:
        if bundle.get("grad_cam"):
            bundle["grad_cam"].remove_hooks()

    lines = [
        "# Explainability deletion comparison",
        "",
        "This report is computed from the configured serving checkpoint and VAL manifest. It measures",
        "the mean change in the original predicted class confidence after replacing the highest",
        "attribution pixels with that image's mean colour, compared with an equal-area random mask.",
        "It is a perturbation diagnostic; it does not prove a disease or explanation correctness.",
        "",
        f"- Model version: {model_version}",
        f"- Split: VAL ({val_manifest.relative_to(root).as_posix()})",
        f"- Images selected: {count}",
        f"- Attribution-region fraction: {params['deletion_region_fraction']}",
        "",
        "| Method | Completed images | Mean confidence drop, attributed region | Mean confidence drop, random region | Paired difference (attributed - random) | Runtime seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in totals.items():
        if result["top"]:
            top_mean = float(np.mean(result["top"]))
            random_mean = float(np.mean(result["random"]))
            lines.append(
                f"| {method.upper()} | {len(result['top'])} | {top_mean:.6f} | "
                f"{random_mean:.6f} | {top_mean - random_mean:.6f} | {result['elapsed']:.3f} |"
            )
        else:
            lines.append(f"| {method.upper()} | 0 | not available | not available | not available | {result['elapsed']:.3f} |")
    if errors:
        lines.extend(["", "## Method errors"])
        lines.extend(f"- {method.upper()}: {message}" for method, message in errors.items())
    lines.extend(["", "The sample is limited to the configured evaluation image count; read the counts above before interpreting the comparison."])
    report = "\n".join(lines) + "\n"
    output_path = root / settings.report_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote explanation evaluation to {output_path}")
    return report


def main() -> int:
    try:
        run()
    except (FileNotFoundError, ExplanationUnavailable, RuntimeError, KeyError, ValueError) as exc:
        print(f"Explanation evaluation not run: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
