"""Background sensitivity check using the project's ExG leaf segmentation."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.calibration import probs_from_logits
from src.model import preprocess_image
from src.segmentation import segment_leaf_with_status


def unknown_background_check(
    reason: str,
    *,
    original_label: str | None = None,
    original_confidence: float | None = None,
    elapsed_ms: float = 0.0,
    mask_area_fraction: float | None = None,
) -> dict[str, Any]:
    """Build an explicit unknown result without implying a successful comparison."""
    return {
        "result": "unknown",
        "reason": reason,
        "mask_area_fraction": mask_area_fraction,
        "original_label": original_label,
        "neutralized_label": None,
        "original_confidence": original_confidence,
        "neutralized_confidence": None,
        "confidence_change": None,
        "neutralized_image_base64": None,
        "elapsed_ms": elapsed_ms,
    }


def build_trust_response_fields(
    *,
    brightness: float | None,
    blur_score: float | None,
    ood_status: str,
    bg_result: dict[str, Any],
) -> dict[str, Any]:
    """Return additive API trust fields from observed values only."""
    result = bg_result.get("result")
    if result not in {"consistent", "inconsistent", "unknown"}:
        result = "unknown"
    return {
        "quality": {
            "brightness": brightness,
            "blur_score": blur_score,
        },
        "ood_status": ood_status,
        "bg_check": result,
        "bg_check_details": bg_result,
    }


def run_background_sensitivity(
    image: Image.Image,
    model: Any,
    *,
    device: str | torch.device,
    image_size: int,
    idx_to_class: dict[int, str],
    original_label: str,
    original_confidence: float,
    min_mask_area_fraction: float,
    max_mask_area_fraction: float,
    allowed_class_names: list[str] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Re-run the model after replacing the segmented background with image mean RGB."""
    started = time.perf_counter()
    if not enabled:
        return unknown_background_check(
            "disabled",
            original_label=original_label,
            original_confidence=original_confidence,
        )

    if not 0.0 <= min_mask_area_fraction <= max_mask_area_fraction <= 1.0:
        return unknown_background_check(
            "invalid_mask_area_bounds",
            original_label=original_label,
            original_confidence=original_confidence,
        )

    try:
        mask, segmentation_succeeded = segment_leaf_with_status(image)
        if not segmentation_succeeded:
            return unknown_background_check(
                "mask_failed",
                original_label=original_label,
                original_confidence=original_confidence,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        if mask.shape != (image.height, image.width) or mask.dtype != np.bool_:
            return unknown_background_check(
                "invalid_mask",
                original_label=original_label,
                original_confidence=original_confidence,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        mask_area_fraction = float(mask.mean())
        if not min_mask_area_fraction <= mask_area_fraction <= max_mask_area_fraction:
            return unknown_background_check(
                "mask_area_out_of_bounds",
                original_label=original_label,
                original_confidence=original_confidence,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                mask_area_fraction=mask_area_fraction,
            )

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mean_rgb = rgb.mean(axis=(0, 1)).round().astype(np.uint8)
        neutralized = np.where(mask[..., None], rgb, mean_rgb[None, None, :]).astype(np.uint8)
        neutralized_image = Image.fromarray(neutralized, mode="RGB")
        tensor = preprocess_image(neutralized_image, image_size).to(device)

        device_type = torch.device(device).type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            logits = model(tensor)
        logits = logits.clone()
        if allowed_class_names:
            allowed_indices = {
                index for index, name in idx_to_class.items() if name in allowed_class_names
            }
            if allowed_indices:
                for index in range(logits.shape[1]):
                    if index not in allowed_indices:
                        logits[0, index] = -1e9
        probabilities = probs_from_logits(logits, temperature=1.0).squeeze()
        top_index = int(probabilities.argmax().item())
        neutralized_label = idx_to_class[top_index]
        neutralized_confidence = float(probabilities[top_index].item())

        image_buffer = io.BytesIO()
        neutralized_image.save(image_buffer, format="PNG")
        image_encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        return {
            "result": "consistent" if neutralized_label == original_label else "inconsistent",
            "reason": None,
            "mask_area_fraction": mask_area_fraction,
            "original_label": original_label,
            "neutralized_label": neutralized_label,
            "original_confidence": float(original_confidence),
            "neutralized_confidence": neutralized_confidence,
            "confidence_change": neutralized_confidence - float(original_confidence),
            "neutralized_image_base64": image_encoded,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    except Exception:
        return unknown_background_check(
            "check_failed",
            original_label=original_label,
            original_confidence=original_confidence,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
