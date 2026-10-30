"""
src/ood_detector.py
===================
Layered input-validation and Out-Of-Distribution (OOD) detection.

Architecture:
1. Layer 1: Heuristic pre-checks (fast, deterministic, zero-compute):
   - Decodability & minimum dimension verification (>= 64x64).
   - Non-flatness / standard deviation thresholding (catches blank/monochrome images).
   - Vegetation/tissue color check: verifies plant tissue signatures
     (green chlorophyll: G > B, or necrotic/chlorotic yellow/brown lesion: R > B).
     Rejects non-leaf surfaces (plain grey tiles, blue screens, flat noise).
2. Layer 2: Penultimate-layer feature extraction:
   - Energy Score (Liu et al., NeurIPS 2020: "Energy-based Out-of-distribution Detection").
   - Mahalanobis Distance (Lee et al., NeurIPS 2018).
3. Status mapping:
   - `ok`: Valid leaf, high confidence.
   - `uncertain`: Valid leaf, but ambiguous/low confidence diagnosis.
   - `rejected_not_a_leaf`: Non-leaf, grey tile, blank, or unstructured OOD input.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image


def check_image_heuristics(image: Image.Image) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Run fast pre-inference heuristic validation on an incoming PIL image.

    Returns:
        (is_valid, rejection_reason_code, human_message)
    """
    w, h = image.size

    # Check 1: Minimum dimension
    if w < 64 or h < 64:
        return (
            False,
            "IMAGE_TOO_SMALL",
            f"Image dimensions ({w}x{h}) are too small for diagnostic analysis (minimum 64x64 required)."
        )

    arr = np.array(image.convert("RGB"), dtype=np.float32)

    # Check 2: Flat / monochrome canvas (e.g. solid grey, blank frame)
    pixel_std = arr.std()
    if pixel_std < 8.0:
        return (
            False,
            "FLAT_IMAGE",
            "Image lacks sufficient texture or contrast (near-constant uniform color detected)."
        )

    # Check 3: Vegetation / leaf color plausibility
    # Normalize RGB to [0, 1]
    norm = arr / 255.0
    r, g, b = norm[..., 0], norm[..., 1], norm[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 0.05, (max_c - min_c) / (max_c + 1e-6), 0.0)
    exg = 2.0 * g - r - b  # Excess Green Index

    # Real crop foliage contains chlorophyll (G > B and exg > 0.01)
    # or disease lesion tissue (chlorotic yellow or necrotic brown: R > B and sat > 0.14)
    # Non-leaf objects (blue sky, gray pavement, synthetic blue/purple cards) have B >= G or sat near zero.
    is_green_tissue = (exg > 0.02) & (g > b)
    is_lesion_tissue = (sat > 0.14) & (r > b) & (r > 0.20)

    leaf_pixel_ratio = (is_green_tissue | is_lesion_tissue).mean()

    if leaf_pixel_ratio < 0.03:
        return (
            False,
            "NOT_A_LEAF_COLOR",
            "Image lacks optical characteristics of crop foliage (no vegetative or lesion color signatures detected)."
        )

    return (True, None, None)


def compute_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute Helmholtz free energy score from class logits.
    E(x) = -T * logsumexp(logits / T)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


class MahalanobisDetector:
    """
    Class-conditional Mahalanobis distance detector fit on training split features.
    """
    def __init__(self, num_classes: int = 38, feat_dim: int = 512):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.class_means: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.class_means = np.zeros((self.num_classes, self.feat_dim), dtype=np.float32)
        diffs = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                c_feats = features[mask]
                self.class_means[c] = c_feats.mean(axis=0)
                diffs.append(c_feats - self.class_means[c])
            else:
                diffs.append(np.zeros((1, self.feat_dim)))

        all_diffs = np.vstack(diffs)
        cov = np.cov(all_diffs, rowvar=False) + 1e-4 * np.eye(self.feat_dim)
        self.precision = np.linalg.pinv(cov)

    def score(self, feat: np.ndarray) -> float:
        if self.class_means is None or self.precision is None:
            return 0.0

        diff = feat[None, :] - self.class_means
        left = diff @ self.precision
        distances_sq = (left * diff).sum(axis=-1)
        return float(np.sqrt(np.maximum(distances_sq.min(), 0.0)))
