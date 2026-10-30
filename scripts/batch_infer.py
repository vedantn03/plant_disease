"""
scripts/batch_infer.py
======================
Batch inference pipeline for processing entire directories of crop leaf images.
Demonstrates:
  - Throughput comparison: Single-image synchronous (/predict) vs. Batch processing (16 vs 64).
  - Production batch outputs to CSV with status, crop, condition, confidence, and latency.

Usage:
  python scripts/batch_infer.py --input-dir demo_samples/ --output-csv reports/batch_predictions.csv
  python scripts/batch_infer.py --benchmark
"""

import argparse
import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model, preprocess_image
from src.ood_detector import check_image_heuristics
from src.utils import parse_class_name


def run_batch_inference(image_paths, model, idx_to_class, batch_size=32, device="cuda"):
    results = []
    t_start = time.perf_counter()

    for i in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[i : i + batch_size]
        tensors = []
        valid_indices = []

        for idx_in_chunk, p in enumerate(chunk_paths):
            try:
                im = Image.open(p).convert("RGB")
                is_valid, code, msg = check_image_heuristics(im)
                if not is_valid:
                    results.append({
                        "filename": p.name,
                        "status": "rejected_not_a_leaf",
                        "crop": "unknown",
                        "condition": "unknown",
                        "confidence": 0.0,
                        "reason": code,
                    })
                else:
                    tensors.append(preprocess_image(im))
                    valid_indices.append((idx_in_chunk, p.name))
            except Exception as e:
                results.append({
                    "filename": p.name,
                    "status": "error",
                    "crop": "unknown",
                    "condition": "unknown",
                    "confidence": 0.0,
                    "reason": str(e),
                })

        if tensors:
            batch_t = torch.cat(tensors, dim=0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = model(batch_t)
            probs = F.softmax(logits, dim=1)
            top_probs, top_idxs = probs.max(dim=1)

            for j, (_, fname) in enumerate(valid_indices):
                prob = top_probs[j].item()
                cls_name = idx_to_class[top_idxs[j].item()]
                try:
                    c, cond = parse_class_name(cls_name)
                except Exception:
                    c, cond = cls_name, "unknown"
                results.append({
                    "filename": fname,
                    "status": "ok" if prob >= 0.50 else "uncertain",
                    "crop": c,
                    "condition": cond,
                    "confidence": round(prob, 4),
                    "reason": "high_confidence" if prob >= 0.50 else "low_confidence",
                })

    elapsed = time.perf_counter() - t_start
    return results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="demo_samples", help="Input image directory")
    parser.add_argument("--output-csv", default="reports/batch_predictions.csv", help="Output CSV path")
    parser.add_argument("--benchmark", action="store_true", help="Run latency/throughput benchmark across batch sizes")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = root / "models" / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    num_classes = len(idx_to_class)

    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    in_dir = root / args.input_dir
    img_paths = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.JPG")) + list(in_dir.glob("*.png"))

    if not img_paths:
        print(f"No images found in {in_dir}")
        return

    print(f"Found {len(img_paths)} images in {in_dir}.")

    if args.benchmark:
        print("\n=== Throughput Benchmark (Single vs. Batches of 16 and 64) ===")
        # Replicate sample paths to test 128 images
        bench_paths = (img_paths * (128 // len(img_paths) + 1))[:128]

        # Batch size 1
        _, t1 = run_batch_inference(bench_paths, model, idx_to_class, batch_size=1, device=device)
        fps1 = len(bench_paths) / t1

        # Batch size 16
        _, t16 = run_batch_inference(bench_paths, model, idx_to_class, batch_size=16, device=device)
        fps16 = len(bench_paths) / t16

        # Batch size 64
        _, t64 = run_batch_inference(bench_paths, model, idx_to_class, batch_size=64, device=device)
        fps64 = len(bench_paths) / t64

        bench_md = f"""# Batch vs. Real-Time Inference Architectural Benchmark

| Mode | Batch Size | Latency per 100 images | Throughput (images/sec) | Typical Deployment Use Case |
|---|:---:|:---:|:---:|---|
| **Real-Time Endpoint (`/predict`)** | 1 | {t1/len(bench_paths)*100:.2f} s | **{fps1:.1f} fps** | Smartphone app synchronous user scan in the field |
| **Micro-Batch (`/predict_batch`)** | 16 | {t16/len(bench_paths)*100:.2f} s | **{fps16:.1f} fps** | Drone scout camera burst uploads |
| **High-Throughput Offline (`batch_infer.py`)** | 64 | {t64/len(bench_paths)*100:.2f} s | **{fps64:.1f} fps** | Nightly warehouse / drone flight batch processing |
"""
        print(bench_md)
        with open(root / "reports" / "INFERENCE_BENCHMARK.md", "w", encoding="utf-8") as f:
            f.write(bench_md)

    # Standard run
    results, elapsed = run_batch_inference(img_paths, model, idx_to_class, batch_size=32, device=device)
    out_path = root / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "status", "crop", "condition", "confidence", "reason"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBatch inference completed in {elapsed:.2f}s ({len(img_paths)/elapsed:.1f} img/s).")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
