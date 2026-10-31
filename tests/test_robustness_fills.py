import numpy as np
from PIL import Image

from src.robustness import mask_background


def test_mask_background_supports_fixed_protocol_fills(monkeypatch):
    image = Image.fromarray(np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))
    monkeypatch.setattr(
        "src.robustness.segment_leaf",
        lambda _image: np.array([[True, False]], dtype=bool),
    )

    black = np.asarray(mask_background(image, "black"))
    grey = np.asarray(mask_background(image, "mid_grey"))
    assert tuple(black[0, 1]) == (0, 0, 0)
    assert tuple(grey[0, 1]) == (128, 128, 128)


def test_mask_background_mean_fill_uses_background_pixels(monkeypatch):
    image = Image.fromarray(
        np.array([[[10, 20, 30], [40, 50, 60], [80, 90, 100]]], dtype=np.uint8)
    )
    monkeypatch.setattr(
        "src.robustness.segment_leaf",
        lambda _image: np.array([[True, False, False]], dtype=bool),
    )

    result = np.asarray(mask_background(image, "per_image_mean"))
    assert tuple(result[0, 1]) == (60, 70, 80)
    assert tuple(result[0, 2]) == (60, 70, 80)
