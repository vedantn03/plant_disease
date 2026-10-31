"""
src/robustness.py
=================
Novelty & Governance Module: Shortcut-Robustness & Capture Bias Suite.

Evaluates how much the classifier relies on biological pathology (leaf lamina / lesions)
versus non-biological shortcuts (laboratory background posterboards, capture lighting, corner artifacts).

Methodology:
  1. Masked-Background Evaluation:
     Evaluates model accuracy / macro-F1 when laboratory backgrounds are removed/masked
     via Woebbecke Excess-Green segmentation (Woebbecke et al., 1995).
     Computes bg_drop = clean_f1 - masked_f1.
  2. Background-Only Probe (Noyan 2022 replication):
     Random Forest trained purely on 4 corner pixels (12 values, zero leaf tissue)
     compared to random chance (1/38 = 2.63%) and majority-class baseline.
  3. Attribution Concentration (Grad-CAM in Leaf):
     Quantifies the fraction of attribution mass concentrated on leaf tissue
     vs background paper: mass_leaf = sum(heatmap * mask) / sum(heatmap).
  4. Segmentation Quality Audit:
     Validates mask fidelity, corner activity, and failure rate across benchmark samples.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
from torchvision import transforms

from src.model import GradCAM, preprocess_image, _IMAGENET_MEAN, _IMAGENET_STD
from src.segmentation import segment_leaf


def mask_background(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Zero (or fill) pixels classified as laboratory background by ExG segmentation."""
    mask = segment_leaf(image)
    arr = np.array(image.convert("RGB"))
    arr[~mask] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(arr)


def evaluate_masked_background(
    model: torch.nn.Module,
    samples: List[Tuple[str, int]],
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 32,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Dict[str, Any]:
    """
    Clean vs background-masked macro-F1 on a provided split.

    Fill is black (0,0,0) then the standard Resize(256)+CenterCrop(224)+ImageNet
    normalise pipeline. bg_drop mixes true background reliance with the
    distribution shift induced by the mask itself.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    def _run(masked: bool) -> tuple[np.ndarray, np.ndarray]:
        preds: list[int] = []
        labels: list[int] = []
        batch_tensors: list[torch.Tensor] = []
        batch_labels: list[int] = []

        def flush() -> None:
            if not batch_tensors:
                return
            x = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(batch_labels)
            batch_tensors.clear()
            batch_labels.clear()

        for path, label in samples:
            im = Image.open(path).convert("RGB")
            if masked:
                im = mask_background(im, fill=fill)
            batch_tensors.append(transform(im))
            batch_labels.append(int(label))
            if len(batch_tensors) >= batch_size:
                flush()
        flush()
        return np.array(preds), np.array(labels)

    clean_pred, clean_y = _run(masked=False)
    masked_pred, masked_y = _run(masked=True)
    clean_f1 = float(f1_score(clean_y, clean_pred, average="macro", zero_division=0))
    masked_f1 = float(f1_score(masked_y, masked_pred, average="macro", zero_division=0))
    bg_drop = clean_f1 - masked_f1
    retention = (masked_f1 / clean_f1) if clean_f1 > 0 else 0.0
    return {
        "n": int(len(samples)),
        "fill": list(fill),
        "clean_macro_f1": round(clean_f1, 4),
        "masked_macro_f1": round(masked_f1, 4),
        "bg_drop": round(bg_drop, 4),
        "retention_rate": round(retention, 4),
    }


def evaluate_mask_quality(
    samples: List[Tuple[str, int]],
    n_samples: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Audit leaf foreground segmentation quality across a fixed seeded subset.
    """
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)

    fg_ratios = []
    corner_actives = []
    failures = 0
    records = []

    for idx in chosen_idx:
        img_path, lbl = samples[idx]
        try:
            im = Image.open(img_path).convert("RGB")
            mask = segment_leaf(im)
            fg_ratio = float(mask.mean())
            fg_ratios.append(fg_ratio)

            # Check if any 4 corners are active (should be false for clean lab photos)
            h, w = mask.shape
            c_active = bool(mask[0, 0] or mask[0, w - 1] or mask[h - 1, 0] or mask[h - 1, w - 1])
            corner_actives.append(c_active)

            # An empty mask (< 2%) or fully filled mask (> 98%) counts as failure
            is_failure = (fg_ratio < 0.02) or (fg_ratio > 0.98)
            if is_failure:
                failures += 1

            records.append({
                "path": str(img_path),
                "label": int(lbl),
                "fg_ratio": round(fg_ratio, 4),
                "corners_active": c_active,
                "status": "FAIL" if is_failure else "CLEAN",
            })
        except Exception:
            failures += 1

    clean_count = len(chosen_idx) - failures
    return {
        "n_samples": len(chosen_idx),
        "failure_count": failures,
        "failure_rate": round(failures / len(chosen_idx), 4) if chosen_idx.size > 0 else 0.0,
        "clean_rate": round(clean_count / len(chosen_idx), 4) if chosen_idx.size > 0 else 0.0,
        "mean_fg_ratio": round(float(np.mean(fg_ratios)), 4) if fg_ratios else 0.0,
        "std_fg_ratio": round(float(np.std(fg_ratios)), 4) if fg_ratios else 0.0,
        "corner_activity_rate": round(float(np.mean(corner_actives)), 4) if corner_actives else 0.0,
        "samples": records,
    }


def evaluate_background_probe_from_csv(
    train_records: List[Dict[str, str]],
    test_records: List[Dict[str, str]],
    class_to_idx: Dict[str, int],
    seed: int = 42,
    max_train: int = 2500,
    max_test: int = 1000,
) -> Dict[str, float]:
    """
    Train a Random Forest classifier solely on 4 corner pixels (12 features)
    to test how much spurious label signal is memorized from paper backgrounds.
    """
    def extract_corner_features(records, limit):
        subset = records[:limit]
        X, y = [], []
        for r in subset:
            try:
                im = Image.open(r["path"]).convert("RGB").resize((64, 64))
                arr = np.array(im)
                feat = np.concatenate([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]) / 255.0
                X.append(feat)
                y.append(class_to_idx[r["class_name"]])
            except Exception:
                continue
        return np.array(X), np.array(y)

    X_train, y_train = extract_corner_features(train_records, max_train)
    X_test, y_test = extract_corner_features(test_records, max_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    num_classes = len(class_to_idx)
    chance = 1.0 / num_classes

    return {
        "corner_rf_test_acc": round(acc, 4),
        "chance_level": round(chance, 4),
        "ratio_over_chance": round(acc / chance, 2),
    }


def compute_gradcam_leaf_ratio(
    model: torch.nn.Module,
    samples: List[Tuple[str, int]],
    device: torch.device,
    n_samples: int = 100,
    seed: int = 42,
    image_size: int = 224,
) -> Dict[str, float]:
    """
    Measure what percentage of Grad-CAM attribution mass falls inside the leaf
    versus the laboratory background.
    """
    model.eval()
    gcam = GradCAM(model)
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)

    leaf_ratios = []

    for idx in chosen_idx:
        img_path, lbl = samples[idx]
        try:
            im = Image.open(img_path).convert("RGB")
            mask = segment_leaf(im)
            inp = preprocess_image(im, image_size).to(device)

            with torch.no_grad():
                logits = model(inp)
                pred_cls = int(torch.argmax(logits, dim=1).item())

            heatmap = gcam.generate(inp, pred_cls)
            h, w = mask.shape
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

            total_mass = float(heatmap_resized.sum())
            if total_mass > 0:
                leaf_mass = float((heatmap_resized * mask).sum())
                ratio = leaf_mass / total_mass
                leaf_ratios.append(ratio)
        except Exception:
            continue

    return {
        "n_evaluated": len(leaf_ratios),
        "gradcam_leaf_mass_mean": round(float(np.mean(leaf_ratios)), 4) if leaf_ratios else 0.0,
        "gradcam_leaf_mass_std": round(float(np.std(leaf_ratios)), 4) if leaf_ratios else 0.0,
        "gradcam_bg_mass_mean": round(1.0 - float(np.mean(leaf_ratios)), 4) if leaf_ratios else 0.0,
    }
