"""
src/segmentation.py
===================
Agronomic foreground-background leaf segmentation for PlantVillage images.

Defensible methodology:
- Uses the standard Woebbecke Excess Green Index (ExG = 2G - R - B, Woebbecke et al., 1995)
  for identifying healthy vegetative tissue.
- Uses Saturation + Red/Blue ratio in HSV/RGB to capture chlorotic (yellow) and necrotic (brown) disease lesions.
- Filters neutral laboratory background boards (gray, off-white, dark matte).
- Applies morphological closing and hole-filling (via scipy.ndimage) to recover internal veins/spots.
- Isolates the primary leaf blade via largest connected component filtering.
- Supports background randomization (compositing onto noise, gradients, soils, textures) for bias ablation.
"""

from typing import Literal, Optional, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage as ndi


def segment_leaf(image: Image.Image) -> np.ndarray:
    """
    Produce a binary boolean mask of the leaf foreground.
    
    Args:
        image: PIL RGB image.
        
    Returns:
        mask: 2D boolean numpy array of shape (H, W), True = leaf, False = background.
    """
    img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    
    # 1. Color indices
    exg = 2.0 * g - r - b  # Excess Green Index
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 0.05, (max_c - min_c) / (max_c + 1e-6), 0.0)
    val = max_c
    
    # Tissue criteria:
    is_green = (exg > 0.02)
    is_lesion = (sat > 0.16) & (r > b) & (val > 0.12)
    is_colored_tissue = (sat > 0.14) & (val > 0.10) & (val < 0.96)
    
    initial_mask = is_green | is_lesion | is_colored_tissue
    
    # Remove neutral gray, black, or bleached white backgrounds
    is_neutral_bg = (sat < 0.10) | (val < 0.08) | ((val > 0.94) & (sat < 0.12))
    initial_mask = initial_mask & (~is_neutral_bg)
    
    # 2. Morphological cleanup
    struct5 = np.ones((5, 5), dtype=bool)
    cleaned = ndi.binary_opening(initial_mask, structure=struct5)
    cleaned = ndi.binary_closing(cleaned, structure=np.ones((7, 7), dtype=bool))
    filled = ndi.binary_fill_holes(cleaned)
    
    # 3. Keep largest connected component (leaf lamina)
    labeled, num_features = ndi.label(filled)
    if num_features == 0:
        # Fallback if entirely empty: central ellipse
        h, w = r.shape
        y, x = np.ogrid[:h, :w]
        return ((x - w / 2.0) ** 2 / (w * 0.4) ** 2 + (y - h / 2.0) ** 2 / (h * 0.4) ** 2) <= 1.0
        
    counts = np.bincount(labeled.flat)
    counts[0] = 0  # ignore 0 (background)
    largest_label = counts.argmax()
    final_mask = (labeled == largest_label)
    
    # Fill remaining internal holes within leaf lamina
    final_mask = ndi.binary_fill_holes(final_mask)
    return final_mask


def generate_synthetic_background(
    h: int,
    w: int,
    bg_type: Literal["soil", "noise", "gradient", "solid", "random"] = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a synthetic background RGB image array (H, W, 3) in uint8 [0, 255].
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if bg_type == "random":
        bg_type = rng.choice(["soil", "noise", "gradient", "solid"])
        
    if bg_type == "soil":
        # Simulates soil / earth tones (brown, tan, dark loam) with Perlin-like smoothed noise
        base_color = rng.uniform([0.25, 0.15, 0.08], [0.55, 0.40, 0.25], size=(1, 1, 3))
        noise = rng.normal(0.0, 0.08, size=(h // 4 + 1, w // 4 + 1, 3))
        # Zoom noise to (h, w)
        zoom_factors = (h / noise.shape[0], w / noise.shape[1], 1)
        noise_resized = ndi.zoom(noise, zoom_factors, order=1)[:h, :w]
        bg = np.clip(base_color + noise_resized, 0.0, 1.0)
        return (bg * 255).astype(np.uint8)
        
    elif bg_type == "noise":
        # Uniform or Gaussian color noise
        bg = rng.uniform(0.1, 0.9, size=(h, w, 3))
        return (bg * 255).astype(np.uint8)
        
    elif bg_type == "gradient":
        # Linear directional color gradient
        c1 = rng.uniform(0.1, 0.9, size=(3,))
        c2 = rng.uniform(0.1, 0.9, size=(3,))
        axis = rng.choice([0, 1])
        t = np.linspace(0, 1, h if axis == 0 else w).reshape((-1, 1) if axis == 0 else (1, -1))
        t = np.repeat(t, w if axis == 0 else h, axis=1 if axis == 0 else 0)
        bg = (1 - t)[..., None] * c1 + t[..., None] * c2
        return np.clip(bg * 255, 0, 255).astype(np.uint8)
        
    else:  # solid
        color = rng.uniform(0.05, 0.95, size=(1, 1, 3))
        bg = np.ones((h, w, 3), dtype=np.float32) * color
        return (bg * 255).astype(np.uint8)


def composite_leaf_on_background(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    bg_type: Literal["soil", "noise", "gradient", "solid", "random"] = "random",
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """
    Composite a segmented leaf image onto a randomized synthetic background.
    """
    img_rgb = image.convert("RGB")
    if mask is None:
        mask = segment_leaf(img_rgb)
        
    img_arr = np.array(img_rgb, dtype=np.uint8)
    h, w = img_arr.shape[:2]
    
    bg_arr = generate_synthetic_background(h, w, bg_type=bg_type, rng=rng)
    
    # 3-channel boolean mask
    mask_3c = mask[..., None]
    composite_arr = np.where(mask_3c, img_arr, bg_arr)
    return Image.fromarray(composite_arr)
