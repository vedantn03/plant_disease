"""Image quality features shared by serving and drift-baseline computation."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import laplace


def image_brightness(img: Image.Image) -> float:
    """Mean grayscale pixel brightness on the API's 0–255 scale."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return float(gray.mean())


def blur_score(img: Image.Image) -> float:
    """Variance of the grayscale Laplacian, matching the API feature."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return float(laplace(gray).var())
