"""
train.py
========
Step 3: Train ResNet18 baseline on a sub-sampled PlantVillage split.

MLflow tracking:
  - Parameters: seed, arch, epochs, lr, batch_size, subsample_per_class, imbalance_strategy
  - Per-epoch: train_loss, train_acc, val_loss, val_acc, val_macro_f1
  - Artifacts: confusion matrix PNG, per-class metrics CSV, model checkpoint, model card
  - Model registry: registered under cfg.mlflow.registered_model_name as a candidate;
    alias movement is handled separately by scripts/promote.py

Usage:
  python scripts/train.py
  python scripts/train.py --config configs/config.yaml

AMP (Automatic Mixed Precision) is always enabled; batch size is a config value.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from src.config_loader import load_config
from src.dataset import make_dataloaders
from src.model import build_model
from src.registry import (
    config_hash,
    dvc_data_hash,
    git_commit,
    register_candidate_version,
)
from src.utils import ensure_dir, get_logger, parse_class_name, set_seed

log = get_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Training / Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    # Normalise for readability
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix (normalised per row)", fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved: {save_path}")


def save_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    idx_to_class: dict[int, str],
) -> None:
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "crop", "condition", "precision", "recall", "f1", "support"])
        for cls_name in class_names:
            if cls_name in report:
                try:
                    crop, cond = parse_class_name(cls_name)
                except ValueError:
                    crop, cond = cls_name, "unknown"
                r = report[cls_name]
                writer.writerow([
                    cls_name, crop, cond,
                    round(r["precision"], 4),
                    round(r["recall"], 4),
                    round(r["f1-score"], 4),
                    int(r["support"]),
                ])
    log.info(f"Per-class metrics saved: {save_path}")


def compute_per_crop_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: dict[int, str],
) -> dict[str, dict]:
    """Aggregate metrics by crop (macro-average of per-class F1s for that crop)."""
    from collections import defaultdict
    crop_preds: dict[str, list] = defaultdict(list)
    crop_true: dict[str, list] = defaultdict(list)

    for true_idx, pred_idx in zip(y_true, y_pred):
        try:
            crop, _ = parse_class_name(idx_to_class[true_idx])
        except (ValueError, KeyError):
            crop = "unknown"
        crop_true[crop].append(true_idx)
        crop_preds[crop].append(pred_idx)

    results = {}
    for crop in sorted(crop_true.keys()):
        ct = np.array(crop_true[crop])
        cp = np.array(crop_preds[crop])
        macro_f1 = f1_score(ct, cp, average="macro", zero_division=0)
        accuracy = (ct == cp).mean()
        results[crop] = {"macro_f1": round(float(macro_f1), 4), "accuracy": round(float(accuracy), 4), "n": len(ct)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for this run and all data sampling",
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Explicitly run in candidate mode (the default output mode)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    splits_dir = project_root / "data" / "splits"
    reports_dir = ensure_dir(project_root / "reports")
    models_dir = ensure_dir(project_root / "models")

    subsample = cfg.data.subsample_per_class
    log.info(f"Subsample per class: {subsample} (None = full data)")

    train_loader, val_loader, test_loader, class_to_idx = make_dataloaders(
        splits_dir, cfg, subsample_per_class=subsample
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    log.info(f"Classes: {num_classes}, Train batches: {len(train_loader)}")

    # ── Compute class weights for weighted loss ────────────────────────────────
    if cfg.model.imbalance_strategy == "weighted_loss":
        class_weights = train_loader.dataset.class_weights().to(device)
        log.info("Using weighted CE loss (strategy=weighted_loss)")
    else:
        class_weights = None
        log.info("Using standard CE loss")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, pretrained=cfg.model.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp and device.type == "cuda")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Load manifest hash for reproducibility tracking
    manifest_hash_file = splits_dir / "manifest_hashes.json"
    manifest_hash = json.loads(manifest_hash_file.read_text())["train"] if manifest_hash_file.exists() else "N/A"

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"MLflow run ID: {run_id}")
        mlflow.set_tag("model_role", "candidate")
        mlflow.set_tag("candidate", "true")

        # Log parameters
        mlflow.log_params({
            "seed": cfg.seed,
            "arch": cfg.model.arch,
            "pretrained": cfg.model.pretrained,
            "num_classes": num_classes,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "amp": cfg.training.amp,
            "imbalance_strategy": cfg.model.imbalance_strategy,
            "subsample_per_class": subsample,
            "train_manifest_hash": manifest_hash,
            "device": str(device),
            "git_commit": git_commit(),
            "config_hash": config_hash(),
            "dvc_data_hash": dvc_data_hash(),
        })

        # ── Training loop ──────────────────────────────────────────────────────
        # SAFETY RULE (Master Plan v2 / Task T7):
        # Candidate mode writes only to models/candidates/<run_id>.pth.
        # Promotion to champion is exclusively governed by the 4-criterion trust gate in scripts/promote.py.
        best_val_f1 = 0.0
        candidates_dir = models_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_checkpoint = candidates_dir / f"{run_id}.pth"

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            scheduler.step()
            elapsed = time.time() - t0

            log.info(
                f"Epoch {epoch}/{cfg.training.epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                f"time={elapsed:.1f}s"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "epoch_seconds": elapsed,
            }, step=epoch)

            # Save best candidate checkpoint based on val macro-F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "run_id": run_id,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "class_to_idx": class_to_idx,
                    "cfg": dict(cfg),
                    "seed": cfg.seed,
                    "git_commit": git_commit(),
                    "config_hash": config_hash(),
                    "dvc_data_hash": dvc_data_hash(),
                    "model_role": "candidate",
                }, candidate_checkpoint)
                log.info(f"  → Candidate model saved to {candidate_checkpoint} (val_macro_f1={val_f1:.4f})")

        # ── Test evaluation (ONE TIME, at the very end) ────────────────────────
        log.info("Loading candidate checkpoint for final test evaluation...")
        checkpoint = torch.load(candidate_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

        log.info(f"TEST RESULTS → loss={test_loss:.4f}, acc={test_acc:.4f}, macro_f1={test_macro_f1:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "best_val_macro_f1": best_val_f1,
        })

        # ── Artifacts ─────────────────────────────────────────────────────────
        cm_path = reports_dir / f"confusion_matrix_{run_id[:8]}.png"
        save_confusion_matrix(test_labels, test_preds, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        per_class_path = reports_dir / f"per_class_metrics_{run_id[:8]}.csv"
        save_per_class_metrics(test_labels, test_preds, class_names, per_class_path, idx_to_class)
        mlflow.log_artifact(str(per_class_path))

        candidate_per_crop = compute_per_crop_metrics(test_labels, test_preds, idx_to_class)
        per_crop_path = reports_dir / f"per_crop_metrics_{run_id[:8]}.json"
        per_crop_path.write_text(json.dumps(candidate_per_crop, indent=2))
        mlflow.log_artifact(str(per_crop_path))

        # Save candidate metrics for downstream (promotion gate)
        candidate_metrics_file = candidates_dir / f"{run_id}_metrics.json"
        candidate_metrics_file.write_text(json.dumps({
            "val_macro_f1": float(best_val_f1),
            "test_macro_f1": float(test_macro_f1),
            "test_acc": float(test_acc),
            "run_id": run_id,
            "checkpoint_path": str(candidate_checkpoint),
        }, indent=2))

        # Log candidate checkpoint as artifact
        mlflow.log_artifact(str(candidate_checkpoint), artifact_path="model")

        # ── Register model in MLflow Model Registry as Candidate ───────────────
        log.info(f"Registering candidate model: {cfg.mlflow.registered_model_name}")
        artifact_uri = run.info.artifact_uri
        source = f"{artifact_uri}/model"
        mv = register_candidate_version(
            source=source,
            run_id=run_id,
            tags={
                "val_macro_f1": f"{best_val_f1:.4f}",
                "split": "val",
            },
        )
        log.info(f"Registered candidate version: {mv.version} (Promotion to champion is governed by scripts/promote.py)")

        log.info(f"\n{'='*60}")
        log.info("TRAINING COMPLETE (CANDIDATE SAVED)")
        log.info(f"  Test macro-F1: {test_macro_f1:.4f}")
        log.info(f"  Test accuracy:  {test_acc:.4f}")
        log.info(f"  Candidate checkpoint: {candidate_checkpoint}")
        log.info(f"  MLflow run: {run_id}")
        log.info(f"  Registered version: v{mv.version} (Candidate)")
        log.info(f"{'='*60}")

        # Print per-crop summary
        log.info("\nPer-crop test macro-F1:")
        for crop, m in candidate_per_crop.items():
            log.info(f"  {crop:<40} F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  n={m['n']}")


if __name__ == "__main__":
    main()
