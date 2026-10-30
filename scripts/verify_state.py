"""Quick state verification — run before Phase 2 work begins."""
import sys, json, pathlib, csv, torch
from collections import Counter
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

root = pathlib.Path(__file__).resolve().parent.parent

# ── Checkpoint ────────────────────────────────────────────────────────────────
ckpt = torch.load(str(root / "models" / "best_model.pth"), map_location="cpu", weights_only=False)
cfg_d = ckpt.get("cfg", {})
data_cfg = cfg_d.get("data", {})
train_cfg = cfg_d.get("training", {})

print("=== CHECKPOINT ===")
print(f"  Epoch saved:             {ckpt.get('epoch')}")
print(f"  Val macro-F1 (ckpt):     {ckpt.get('val_macro_f1'):.4f}")
print(f"  subsample_per_class:     {data_cfg.get('subsample_per_class')}")
print(f"  epochs configured:       {train_cfg.get('epochs')}")
print(f"  num class_to_idx keys:   {len(ckpt.get('class_to_idx', {}))}")

# ── Best metrics ─────────────────────────────────────────────────────────────
m = json.loads((root / "models" / "best_metrics.json").read_text())
print("\n=== TEST METRICS ===")
print(f"  test_macro_f1:  {m['test_macro_f1']:.4f}")
print(f"  test_accuracy:  {m['test_acc']:.4f}")
print(f"  run_id:         {m['run_id']}")

# ── Split sizes ───────────────────────────────────────────────────────────────
print("\n=== SPLITS ===")
for split in ["train", "val", "test"]:
    with open(root / "data" / "splits" / f"{split}.csv") as f:
        rows = list(csv.DictReader(f))
    cls_counts = Counter(r["class_name"] for r in rows)
    print(f"  {split:5s}: {len(rows):6d} rows | {len(cls_counts)} classes | "
          f"min/cls={min(cls_counts.values())} max/cls={max(cls_counts.values())}")

# ── Per-crop metrics ──────────────────────────────────────────────────────────
print("\n=== PER-CROP METRICS (test) ===")
pc = json.loads((root / "reports" / "per_crop_metrics.json").read_text())
for crop, v in sorted(pc.items(), key=lambda x: x[1]["macro_f1"]):
    print(f"  {crop:45s} F1={v['macro_f1']:.4f}  acc={v['accuracy']:.4f}  n={v['n']}")

# ── Files present ─────────────────────────────────────────────────────────────
print("\n=== KEY FILES ===")
must_exist = [
    "models/best_model.pth", "reports/confusion_matrix.png",
    "reports/per_class_metrics.csv", "reports/per_crop_metrics.json",
    "reports/audit_report.md", "kb/advisories.yaml",
    "src/api.py", "src/model.py", "scripts/simulate_drift.py",
]
for f in must_exist:
    p = root / f
    status = "OK" if p.exists() else "MISSING"
    size = f"{p.stat().st_size:>8} B" if p.exists() else ""
    print(f"  [{status}] {f:45s} {size}")

print("\nDone.")
