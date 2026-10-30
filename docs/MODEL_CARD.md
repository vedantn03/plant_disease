# Model Card: PlantVillage Leaf Disease Classifier

> All numbers in this card come from `reports/metrics.json` and `models/best_metrics.json`.
> Run `make evaluate` to regenerate. Do not use numbers from older session summaries
> or conversation logs — only the artifact files are authoritative.

---

## Model Details

| Field | Value |
|-------|-------|
| **Name** | PlantDiseaseClassifier |
| **Architecture** | ResNet18 (pretrained ImageNet, fine-tuned on PlantVillage full split) |
| **Framework** | PyTorch 2.11.0+cu128 |
| **Task** | Multi-class image classification (38 classes, 14 crops) |
| **Input** | RGB image → Resize(256) → CenterCrop(224) → ImageNet normalise |
| **Output** | Class probabilities over 38 PlantVillage classes |
| **Version** | Champion v2 (full-data run, 8 epochs) |
| **MLflow Run ID** | `fb731412d82544e0abc231b8d7c62f95` |
| **Checkpoint** | `models/best_model.pth` |
| **Registry alias** | `champion` → version 2 |

---

## Intended Use

**This model is intended ONLY for:**
- Demonstrating an MLOps pipeline for a university course project.
- Educational illustration of how a leaf-image classification system could be structured.
- Providing a prototype advisory suggestion from a cited, auditable knowledge base — NOT from the model itself.

**This model is NOT intended for:**
- Field-ready crop disease diagnosis.
- Replacing professional agronomic or regulatory advice.
- Any commercial or safety-critical application.

---

## Training Data

| Item | Value |
|------|-------|
| **Dataset** | PlantVillage (color subset) |
| **Total images audited** | 54,305 |
| **Exact duplicates removed** | 21 (SHA-256 deduplication) |
| **Images after deduplication** | 54,284 |
| **Split strategy** | Group-aware stratified 70 / 15 / 15 — `<crop>::<capture_id>` scoping, zero group overlap across splits |
| **Training images** | **38,002 (full split, no subsampling)** |
| **Validation images** | 8,148 |
| **Test images** | 8,134 |
| **Training epochs** | 8 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 (CosineAnnealingLR) |
| **Loss** | Weighted cross-entropy (inverse-frequency class weights) |
| **Augmentations (train)** | RandomResizedCrop(224, scale=(0.7,1.0)), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.3), RandomRotation(±15°), RandomAffine(translate 8%, scale ±5%), ColorJitter |
| **Preprocessing (val/test/inference)** | Resize(256) → CenterCrop(224) → ImageNet normalise |
| **AMP** | Enabled (FP16, RTX 3050 6 GB) |
| **Imbalance handling** | Inverse-frequency class-weighted CE loss |

---

## Evaluation Metrics (Held-Out Test Split, N = 8,134)

All from `reports/metrics.json`.

| Metric | Point Estimate | 95% Bootstrap CI |
| :--- | :---: | :---: |
| **Macro-F1** | **99.03%** | [98.71%, 99.31%] |
| **Weighted-F1** | **99.30%** | [99.12%, 99.47%] |
| **Overall Accuracy** | **99.30%** | [99.13%, 99.47%] |

### Inference Cost

| Item | Value |
|------|-------|
| Parameters | 11.2 M |
| Model file size | 128.26 MB |
| GPU latency (RTX 3050) | 2.54 ms median, 4.19 ms p95 |
| CPU latency (host) | 61.98 ms median, 64.74 ms p95 |

### Weakest 5 Classes (from `reports/metrics.json`)

| Class | F1 | Recall | Support |
|---|---|---|---|
| Corn Cercospora leaf spot / Gray leaf spot | 0.8993 | 0.8701 | 77 |
| Corn Northern Leaf Blight | 0.9435 | 0.9595 | 148 |
| Potato healthy | 0.9583 | 1.0000 | 23 |
| Tomato Early Blight | 0.9592 | 0.9400 | 150 |
| Tomato Late Blight | see metrics.json | see metrics.json | — |

### Per-Crop Summary

| Crop | Macro-F1 | Accuracy | N |
|---|---|---|---|
| Apple | 1.0000 | 1.0000 | 474 |
| Blueberry | 1.0000 | 1.0000 | 225 |
| Cherry (incl. sour) | 1.0000 | 1.0000 | 286 |
| Corn (maize) | 0.9593 | 0.9706 | 578 |
| Grape | 1.0000 | 1.0000 | 608 |
| Orange | 1.0000 | 1.0000 | 826 |
| Peach | 1.0000 | 1.0000 | 399 |
| Pepper (bell) | 1.0000 | 1.0000 | 372 |
| Potato | 0.7371 | 0.9907 | 323 |
| Raspberry | 1.0000 | 1.0000 | 56 |
| Soybean | 1.0000 | 1.0000 | 764 |
| Squash | 1.0000 | 1.0000 | 275 |
| Strawberry | 1.0000 | 1.0000 | 234 |
| Tomato | 0.8210 | 0.9864 | 2,714 |

> Note: Potato (F1=0.7371) and Tomato (F1=0.8210) crop-level macro-F1 are lower because
> (a) they have more disease sub-classes with small support (e.g. Potato healthy: n=23),
> and (b) Solanaceae leaves share morphological features across crops.

---

## Calibration

| Item | Value | Source |
|------|-------|--------|
| Temperature scaling factor T | 1.32 | `reports/metrics.json` → temperature_scale |
| Pre-scaling ECE | see calibration section of metrics.json | measured |
| Post-scaling ECE | see calibration section of metrics.json | measured |

---

## Empirical Bias Probe (Capture-Bias Replication, Noyan 2022)

> These numbers are **unflattering and intentionally disclosed**. They document that
> PlantVillage is a biased benchmark, NOT that our model is poor at leaf diagnosis.

| Probe | Result | Chance | Meaning |
|---|---|---|---|
| Corner-pixel RF (4 corner pixels, 12 values, no leaf) | **46.1% accuracy** | 2.6% | Nearly half the class identity is in the background paper/lighting |
| Centre-masked CNN (central 60% of leaf hidden) | **68.67% accuracy** | 2.6% | Model can classify many images from background alone |

---

## Out-of-Distribution (OOD) Rejection at 0.50 Softmax Threshold

| Item | Value | Source |
|------|-------|--------|
| N OOD samples tested (synthetic noise + gradients) | 200 | `reports/metrics.json` → ood_rejection |
| Rejection rate at 0.50 threshold | **0.0% (0 of 200 rejected)** | measured |
| Mean OOD confidence | 0.9362 (93.62%) | measured |
| p95 OOD confidence | 0.9997 | measured |

> **Critical limitation**: Standard Softmax thresholding does NOT adequately detect
> out-of-distribution inputs. The model accepts ALL synthetic noise/non-leaf images
> with very high confidence. An upstream leaf detection or dedicated OOD detector
> (Energy score, Mahalanobis distance) is required before production use.

---

## Robustness Under Corruption (from `reports/metrics.json`)

Accuracy at mild / moderate / severe corruption levels:

| Corruption | Mild | Moderate | Severe |
|---|---|---|---|
| Gaussian Blur | 0.9667 | 0.6033 | 0.3000 |
| Darken | 0.9967 | 0.9800 | 0.4300 |
| Brighten | 0.9933 | 0.9733 | 0.9167 |
| Gaussian Noise | 0.9867 | 0.5867 | 0.0933 |

---

## Dataset Structural Gaps

| Crop | Classes Available | Missing |
|---|---|---|
| Blueberry | healthy only | no disease class |
| Raspberry | healthy only | no disease class |
| Soybean | healthy only | no disease class |
| Orange | disease only (Citrus Greening) | no healthy class |
| Squash | disease only (Powdery Mildew) | no healthy class |

These 5 crops represent ~26.4% of the test set (2,146 / 8,134 images). For these
crops, crop identity alone determines the predicted label — the model is forced to
diagnose Orange leaves as diseased regardless of actual health status.

---

## Advisory Knowledge Base Coverage

| Status | Count | Notes |
|---|---|---|
| VERIFIED | 8 | Cited university extension sources (US-based); regions noted |
| PLACEHOLDER | 30 | No verified dosage; returns "No verified recommendation available" |
| Total | 38 | One entry per class |

---

## Limitations

1. **Lab vs. field gap**: Trained exclusively on PlantVillage single-leaf lab images
   (uniform backgrounds). Behaviour on field photos with soil, shadows, and complex
   backgrounds is unknown and likely poor.
2. **Capture bias**: Corner-pixel probe accuracy 46.1% (vs. 2.6% chance) proves
   substantial label information is in the background, not the leaf.
3. **OOD overconfidence**: 0% rejection of synthetic noise at the 0.50 threshold;
   model is severely overconfident on unknown inputs.
4. **Structural class gaps**: Orange/Squash have no healthy class; Blueberry/Raspberry/
   Soybean have no disease class. Predictions for these crops cannot be "wrong or right"
   in the traditional sense.
5. **Advisory coverage**: Only 8/38 classes have university-verified treatment notes.
6. **Single-crop image assumption**: Model assumes the uploaded image contains a single
   crop type. Mixed-crop or whole-plant images are not supported.
7. **No severity rating**: Outputs a class label only; does not rate disease severity
   (% leaf area affected).
8. **EXIF metadata**: Uploaded images are processed in-memory and not stored. GPS/EXIF
   data is never logged. (Phase 3 privacy hardening adds explicit EXIF stripping.)
