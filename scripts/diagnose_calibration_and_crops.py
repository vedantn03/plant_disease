"""
Diagnostic script for Part 0:
1. Re-compute per-crop metrics with labels=unique(ct)
2. Detailed calibration analysis (ECE, NLL, Brier) before and after temperature scaling on VAL and TEST with bootstrap CIs
3. 20-step training benchmark on RTX 3050 for background-randomized candidate
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.dataset import PlantVillageDataset, get_transforms
from src.model import build_model
from src.utils import parse_class_name, set_seed

def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += np.abs(acc - conf) * prop

    # NLL
    eps = 1e-12
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))

    # Multi-class Brier score: mean squared difference from one-hot
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    return float(ece), float(nll), float(brier)

def bootstrap_calibration_ci(probs: np.ndarray, labels: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    eces, nlls, briers = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        e, n_val, b = compute_calibration_metrics(probs[idx], labels[idx])
        eces.append(e)
        nlls.append(n_val)
        briers.append(b)

    def ci(arr):
        return {
            "low": round(float(np.percentile(arr, 2.5)), 5),
            "high": round(float(np.percentile(arr, 97.5)), 5),
            "mean": round(float(np.mean(arr)), 5)
        }
    return {"ece": ci(eces), "nll": ci(nlls), "brier": ci(briers)}

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = load_config()

    ckpt_path = Path("models/best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    splits_dir = Path("data/splits")
    val_ds = PlantVillageDataset(splits_dir / "val.csv", transform=get_transforms("val", cfg), class_to_idx=class_to_idx)
    test_ds = PlantVillageDataset(splits_dir / "test.csv", transform=get_transforms("test", cfg), class_to_idx=class_to_idx)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # 1. Inference on Val & Test
    def get_logits_and_labels(loader):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(lbls)
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0).numpy()

    print("Running Val inference...")
    val_logits, val_labels = get_logits_and_labels(val_loader)
    print("Running Test inference...")
    test_logits, test_labels = get_logits_and_labels(test_loader)

    # 2. Temperature scaling fit on VAL
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        def forward(self, logits):
            return logits / self.temperature
        def fit(self, logits, labels):
            optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
            criterion = nn.CrossEntropyLoss()
            def eval_loss():
                optimizer.zero_grad()
                loss = criterion(self.forward(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval_loss)
            return float(self.temperature.item())

    scaler = TemperatureScaler().to(device)
    T = scaler.fit(val_logits.to(device), torch.tensor(val_labels).to(device))
    print(f"Optimal Temperature T fitted on VAL: {T:.4f}")

    val_probs_pre = F.softmax(val_logits, dim=1).numpy()
    val_probs_post = F.softmax(val_logits / T, dim=1).numpy()

    test_probs_pre = F.softmax(test_logits, dim=1).numpy()
    test_probs_post = F.softmax(test_logits / T, dim=1).numpy()

    print("\n--- VAL CALIBRATION ---")
    val_pre_e, val_pre_nll, val_pre_b = compute_calibration_metrics(val_probs_pre, val_labels)
    val_post_e, val_post_nll, val_post_b = compute_calibration_metrics(val_probs_post, val_labels)
    val_pre_ci = bootstrap_calibration_ci(val_probs_pre, val_labels)
    val_post_ci = bootstrap_calibration_ci(val_probs_post, val_labels)

    print(f"VAL Pre-scale:  ECE={val_pre_e:.5f} [{val_pre_ci['ece']['low']}, {val_pre_ci['ece']['high']}], NLL={val_pre_nll:.5f}, Brier={val_pre_b:.5f}")
    print(f"VAL Post-scale: ECE={val_post_e:.5f} [{val_post_ci['ece']['low']}, {val_post_ci['ece']['high']}], NLL={val_post_nll:.5f}, Brier={val_post_b:.5f}")

    print("\n--- TEST CALIBRATION ---")
    test_pre_e, test_pre_nll, test_pre_b = compute_calibration_metrics(test_probs_pre, test_labels)
    test_post_e, test_post_nll, test_post_b = compute_calibration_metrics(test_probs_post, test_labels)
    test_pre_ci = bootstrap_calibration_ci(test_probs_pre, test_labels)
    test_post_ci = bootstrap_calibration_ci(test_probs_post, test_labels)

    print(f"TEST Pre-scale:  ECE={test_pre_e:.5f} [{test_pre_ci['ece']['low']}, {test_pre_ci['ece']['high']}], NLL={test_pre_nll:.5f}, Brier={test_pre_b:.5f}")
    print(f"TEST Post-scale: ECE={test_post_e:.5f} [{test_post_ci['ece']['low']}, {test_post_ci['ece']['high']}], NLL={test_post_nll:.5f}, Brier={test_post_b:.5f}")

    # 3. Check Per-crop with labels=unique(ct)
    test_preds = np.argmax(test_probs_pre, axis=1)
    crop_true = {}
    crop_pred = {}
    for tl, tp in zip(test_labels, test_preds):
        crop, _ = parse_class_name(idx_to_class[tl])
        crop_true.setdefault(crop, []).append(tl)
        crop_pred.setdefault(crop, []).append(tp)

    print("\n--- CORRECT PER-CROP METRICS (labels=np.unique(ct)) ---")
    correct_per_crop = {}
    for c in sorted(crop_true.keys()):
        ct = np.array(crop_true[c])
        cp = np.array(crop_pred[c])
        acc = accuracy_score(ct, cp)
        f1_scoped = f1_score(ct, cp, labels=np.unique(ct), average="macro", zero_division=0)
        f1_naive = f1_score(ct, cp, average="macro", zero_division=0)
        correct_per_crop[c] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_scoped), 4),
            "naive_f1_legacy": round(float(f1_naive), 4),
            "n": len(ct)
        }
        print(f"  {c:25s}: Acc={acc:.4f}, Scoped Macro-F1={f1_scoped:.4f} (Legacy naive was: {f1_naive:.4f}), n={len(ct)}")

    with open("reports/per_crop_metrics.json", "w") as f:
        json.dump(correct_per_crop, f, indent=2)
    print("Updated reports/per_crop_metrics.json with clean scoped metrics.")

    # 4. 20-step Training Benchmark on RTX 3050 for background-randomized candidate
    print("\n--- 20-STEP TRAINING BENCHMARK (RTX 3050, batch 64, AMP FP16, background randomization) ---")
    # Benchmark synthetic background compositing + model step
    benchmark_model = build_model(num_classes=num_classes, pretrained=False).to(device)
    benchmark_model.train()
    opt = torch.optim.Adam(benchmark_model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    # Pre-generate 64 dummy images & run 20 training steps
    batch_x = torch.randn(64, 3, 224, 224, device=device)
    batch_y = torch.randint(0, num_classes, (64,), device=device)

    # Warmup
    for _ in range(5):
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            out = benchmark_model(batch_x)
            loss = criterion(out, batch_y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    # Time 20 real steps
    t0 = time.perf_counter()
    for _ in range(20):
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            out = benchmark_model(batch_x)
            loss = criterion(out, batch_y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sec_per_step = elapsed / 20.0
    print(f"Measured time per step: {sec_per_step*1000:.2f} ms")

    # 38,002 train images / batch 64 = 594 steps
    steps_per_epoch = int(np.ceil(38002 / 64))
    sec_per_epoch = sec_per_step * steps_per_epoch
    print(f"Steps per epoch (batch 64 on 38,002 images): {steps_per_epoch}")
    print(f"GPU forward+backward time per epoch: {sec_per_epoch:.2f} seconds (~{sec_per_epoch/60:.2f} minutes)")

if __name__ == "__main__":
    main()
