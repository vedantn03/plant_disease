"""
scripts/benchmark_xai.py
========================
Comparative XAI Benchmark: Grad-CAM vs SHAP vs LIME (Unit-V Syllabus Requirement).
Measures inference latency, attribution sparsity, and foreground leaf concentration.
Outputs:
  - reports/figures/xai_trio_comparison.png
  - reports/xai_benchmark.json
"""

from pathlib import Path
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model, GradCAM, preprocess_image
from src.segmentation import segment_leaf
from src.utils import get_logger

log = get_logger("benchmark_xai")


def main():
    log.info("Starting XAI Benchmark: Grad-CAM vs SHAP vs LIME...")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json_path = PROJECT_ROOT / "reports" / "xai_benchmark.json"

    ckpt_path = PROJECT_ROOT / "models" / "best_model.pth"
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found at {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading champion model onto {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Pick a representative leaf sample (Tomato Early Blight)
    demo_img_path = PROJECT_ROOT / "demo_samples" / "Tomato___Early_blight.jpg"
    if not demo_img_path.exists():
        log.error(f"Demo image not found at {demo_img_path}")
        return

    raw_img = Image.open(demo_img_path).convert("RGB")
    raw_np = np.array(raw_img.resize((224, 224)))
    input_tensor = preprocess_image(raw_img).to(device)

    # 1. Leaf foreground mask
    seg_mask = segment_leaf(raw_img)
    seg_mask_224 = np.array(Image.fromarray(seg_mask).resize((224, 224), resample=Image.NEAREST)) > 0
    leaf_pixel_ratio = float(seg_mask_224.sum() / seg_mask_224.size)

    # --------------------------------------------------------------------------
    # A. Grad-CAM
    # --------------------------------------------------------------------------
    log.info("Benchmarking Grad-CAM...")
    gradcam = GradCAM(model)
    t0 = time.perf_counter()
    with torch.enable_grad():
        cam_map = gradcam.generate(input_tensor)
    gradcam_latency = (time.perf_counter() - t0) * 1000

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        conf = probs[0, pred_idx].item()

    cam_tensor = torch.from_numpy(cam_map).unsqueeze(0).unsqueeze(0)
    cam_interp = torch.nn.functional.interpolate(
        cam_tensor, size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()
    cam_norm = cam_interp / (cam_interp.max() + 1e-8)
    cam_leaf_mass = float(cam_norm[seg_mask_224].sum() / (cam_norm.sum() + 1e-8))

    # --------------------------------------------------------------------------
    # B. SHAP (GradientExplainer)
    # --------------------------------------------------------------------------
    log.info("Benchmarking SHAP (GradientExplainer)...")
    import shap
    bg_tensors = torch.zeros(10, 3, 224, 224).to(device)
    explainer_shap = shap.GradientExplainer(model, bg_tensors)
    t0 = time.perf_counter()
    shap_vals, _ = explainer_shap.shap_values(input_tensor, ranked_outputs=1)
    shap_latency = (time.perf_counter() - t0) * 1000

    if isinstance(shap_vals, list):
        sv = shap_vals[0]
    else:
        sv = shap_vals
    sv = np.squeeze(sv)
    shap_map = np.abs(sv).mean(axis=0)
    shap_norm = shap_map / (shap_map.max() + 1e-8)
    shap_leaf_mass = float(shap_norm[seg_mask_224].sum() / (shap_norm.sum() + 1e-8))

    # --------------------------------------------------------------------------
    # C. LIME (Superpixel Image Explainer)
    # --------------------------------------------------------------------------
    log.info("Benchmarking LIME (Superpixel Explainer)...")
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    lime_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def batch_predict(images):
        model.eval()
        batch = torch.stack([lime_transform(img) for img in images], dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    lime_explainer = lime_image.LimeImageExplainer()
    t0 = time.perf_counter()
    explanation = lime_explainer.explain_instance(
        raw_np,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=100,
    )
    lime_latency = (time.perf_counter() - t0) * 1000

    temp, lime_mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )
    lime_leaf_mass = float(lime_mask[seg_mask_224].sum() / (lime_mask.sum() + 1e-8))

    # --------------------------------------------------------------------------
    # Save Side-by-Side Comparison Figure
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(raw_np)
    axes[0].set_title(f"Input Leaf ({idx_to_class[pred_idx]})\nConfidence: {conf:.1%}")
    axes[0].axis("off")

    axes[1].imshow(raw_np)
    axes[1].imshow(cam_norm, cmap="jet", alpha=0.5)
    axes[1].set_title(f"1. Grad-CAM (CNN Hooks)\nLatency: {gradcam_latency:.1f}ms | Leaf Mass: {cam_leaf_mass:.1%}")
    axes[1].axis("off")

    axes[2].imshow(raw_np)
    axes[2].imshow(shap_norm, cmap="hot", alpha=0.55)
    axes[2].set_title(f"2. SHAP (Shapley Values)\nLatency: {shap_latency:.1f}ms | Leaf Mass: {shap_leaf_mass:.1%}")
    axes[2].axis("off")

    axes[3].imshow(mark_boundaries(temp, lime_mask))
    axes[3].set_title(f"3. LIME (Superpixels)\nLatency: {lime_latency:.1f}ms | Leaf Mass: {lime_leaf_mass:.1%}")
    axes[3].axis("off")

    plt.tight_layout()
    fig_path = figures_dir / "xai_trio_comparison.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved trio XAI figure to {fig_path}")

    # --------------------------------------------------------------------------
    # Save Quantitative JSON Benchmark
    # --------------------------------------------------------------------------
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": str(demo_img_path.name),
        "predicted_class": idx_to_class[pred_idx],
        "confidence": float(conf),
        "leaf_foreground_area_ratio": round(leaf_pixel_ratio, 4),
        "frameworks": {
            "gradcam": {
                "name": "Grad-CAM (Layer 4 Activation Gradients)",
                "latency_ms": round(gradcam_latency, 2),
                "foreground_leaf_attribution_mass": round(cam_leaf_mass, 4),
                "granularity": "Coarse 7x7 spatial feature map",
                "syllabus_topic": "Unit-V: Deep Neural Activation Mapping",
            },
            "shap": {
                "name": "SHAP (GradientExplainer Shapley Values)",
                "latency_ms": round(shap_latency, 2),
                "foreground_leaf_attribution_mass": round(shap_leaf_mass, 4),
                "granularity": "Pixel-level additive Shapley attributions",
                "syllabus_topic": "Unit-V: Game-theoretic Feature Attribution",
            },
            "lime": {
                "name": "LIME (Local Interpretable Model-agnostic Explanations)",
                "latency_ms": round(lime_latency, 2),
                "foreground_leaf_attribution_mass": round(lime_leaf_mass, 4),
                "granularity": "Perturbation superpixel segmentation",
                "syllabus_topic": "Unit-V: Local Surrogate Perturbation Modeling",
            },
        },
    }

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved XAI benchmark metrics to {benchmark_json_path}")
    print("\n" + "=" * 60)
    print("XAI COMPARATIVE BENCHMARK COMPLETED")
    print("=" * 60)
    print(f"Grad-CAM:  Latency = {gradcam_latency:6.1f} ms  |  Leaf Mass = {cam_leaf_mass:.1%}")
    print(f"SHAP:      Latency = {shap_latency:6.1f} ms  |  Leaf Mass = {shap_leaf_mass:.1%}")
    print(f"LIME:      Latency = {lime_latency:6.1f} ms  |  Leaf Mass = {lime_leaf_mass:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
