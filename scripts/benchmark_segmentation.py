"""
scripts/benchmark_segmentation.py
=================================
Evaluates the agronomic leaf segmentation pipeline on a stratified sample of 40 test images.
Saves a visual grid artifact and computes the empirical mask failure rate.
"""

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.segmentation import segment_leaf, composite_leaf_on_background

def main():
    root = Path(__file__).resolve().parent.parent
    test_csv = root / "data" / "splits" / "test.csv"
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    with open(test_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
        
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(records), size=40, replace=False)
    sampled = [records[i] for i in sample_indices]
    
    results = []
    failure_count = 0
    
    # Setup 8 rows x 5 columns grid (40 pairs: original + segmented boundary)
    fig, axes = plt.subplots(8, 5, figsize=(20, 32))
    axes = axes.flatten()
    
    for idx, (rec, ax) in enumerate(zip(sampled, axes)):
        path = Path(rec["path"])
        cls_name = rec["class_name"]
        
        try:
            img = Image.open(path).convert("RGB")
            mask = segment_leaf(img)
            fg_ratio = float(mask.mean())
            
            # Heuristic failure check:
            # 1. Extreme ratio: < 10% (leaf dropped) or > 98% (background captured entirely)
            # 2. All 4 image corners active (border leak)
            corners_active = (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1])
            is_suspicious = (fg_ratio < 0.10) or (fg_ratio > 0.98) or corners_active
            
            if is_suspicious:
                failure_count += 1
                
            results.append({
                "class_name": cls_name,
                "fg_ratio": round(fg_ratio, 4),
                "corners_active": bool(corners_active),
                "status": "FLAGGED_POOR" if is_suspicious else "CLEAN",
                "path": str(path)
            })
            
            # Render composite overlay: masked leaf with red contour or dim background
            img_arr = np.array(img).copy()
            # Dim background by 70% to clearly visualize extracted lamina
            bg_dim = (img_arr * 0.3).astype(np.uint8)
            overlay = np.where(mask[..., None], img_arr, bg_dim)
            
            ax.imshow(overlay)
            status_tag = "[FAIL]" if is_suspicious else "[OK]"
            color = "red" if is_suspicious else "green"
            short_cls = cls_name.split("___")[-1][:22]
            ax.set_title(f"#{idx+1} {status_tag} {short_cls} ({fg_ratio:.1%})", fontsize=8, color=color)
            ax.axis("off")
            
        except Exception as e:
            failure_count += 1
            results.append({"class_name": cls_name, "error": str(e), "status": "ERROR"})
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            ax.axis("off")
            
    plt.tight_layout()
    grid_path = figures_dir / "segmentation_40_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close()
    
    failure_rate = failure_count / 40.0
    summary = {
        "n_samples": 40,
        "failure_count": failure_count,
        "failure_rate": round(failure_rate, 4),
        "clean_rate": round(1.0 - failure_rate, 4),
        "grid_figure": str(grid_path.relative_to(root)),
        "samples": results
    }
    
    out_json = root / "reports" / "segmentation_quality.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Segmentation Benchmark Complete:")
    print(f"  Samples: 40")
    print(f"  Flagged Poor/Failed: {failure_count} / 40 ({failure_rate:.1%})")
    print(f"  Clean Extraction: {40 - failure_count} / 40 ({1.0 - failure_rate:.1%})")
    print(f"  Saved figure: {grid_path}")
    print(f"  Saved metrics: {out_json}")

if __name__ == "__main__":
    main()
