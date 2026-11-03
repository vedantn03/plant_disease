"""Background sensitivity tests use synthetic images and a fixed test model."""

import torch
from PIL import Image, ImageDraw

from src.background_check import build_trust_response_fields, run_background_sensitivity


class _FixedLogitModel:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, tensor):
        return torch.tensor([self.logits], dtype=torch.float32, device=tensor.device)


def _synthetic_leaf():
    image = Image.new("RGB", (64, 64), (110, 110, 110))
    ImageDraw.Draw(image).ellipse((12, 8, 52, 56), fill=(35, 145, 45))
    return image


def _check(image, logits, *, minimum=0.05, maximum=0.95):
    return run_background_sensitivity(
        image,
        _FixedLogitModel(logits),
        device="cpu",
        image_size=32,
        idx_to_class={0: "Apple___healthy", 1: "Corn___rust"},
        original_label="Apple___healthy",
        original_confidence=0.8,
        min_mask_area_fraction=minimum,
        max_mask_area_fraction=maximum,
    )


def test_background_sensitivity_can_be_consistent():
    result = _check(_synthetic_leaf(), [3.0, 0.0])
    assert result["result"] == "consistent"
    assert result["neutralized_label"] == result["original_label"]
    assert result["confidence_change"] is not None
    assert result["neutralized_image_base64"]


def test_background_sensitivity_can_be_inconsistent():
    result = _check(_synthetic_leaf(), [0.0, 3.0])
    assert result["result"] == "inconsistent"
    assert result["neutralized_label"] != result["original_label"]


def test_background_sensitivity_is_unknown_when_segmentation_falls_back():
    result = _check(Image.new("RGB", (64, 64), (120, 120, 120)), [3.0, 0.0])
    assert result["result"] == "unknown"
    assert result["reason"] == "mask_failed"
    assert result["neutralized_label"] is None


def test_background_sensitivity_is_unknown_outside_mask_bounds():
    result = _check(_synthetic_leaf(), [3.0, 0.0], minimum=0.9, maximum=0.95)
    assert result["result"] == "unknown"
    assert result["reason"] == "mask_area_out_of_bounds"


def test_trust_response_fields_are_additive_and_invalid_result_is_unknown():
    original = {"status": "ok", "crop": "apple", "confidence": 0.8}
    fields = build_trust_response_fields(
        brightness=120.0,
        blur_score=3.0,
        ood_status="accepted",
        bg_result={"result": "not-a-result"},
    )
    response = {**original, **fields}

    assert {key: response[key] for key in original} == original
    assert response["bg_check"] == "unknown"
    assert response["quality"] == {"brightness": 120.0, "blur_score": 3.0}
    assert response["ood_status"] == "accepted"
