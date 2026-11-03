"""
scripts/compute_val_diagnostics.py
==================================
Part A diagnostics on the current champion checkpoint (read-only).

  A1. Refit temperature T on saved VAL logits (LBFGS + 1-D grid).
  A3. VAL gate baselines: clean / masked-background macro-F1, bg_drop,
      full and within-crop macro-F1, min-recall + tiny-class table.
  A5. MLflow wall-clock vs the synthetic 20-step benchmark.
  A7. Registered model versions and whether v1 is a safe rollback target.

Does NOT write models/best_model.pth.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (
    calibration_bundle,
    fit_temperature,
    softmax_np,
)
from src.config_loader import load_config
from src.dataset import PlantVillageDataset, get_transforms
from src.gates import one_extra_error_recall, wilson_lower_bound
from src.model import build_model
from src.robustness import evaluate_mask_fill_protocol
from src.utils import get_logger, parse_class_name, set_seed

log = get_logger("val_diagnostics")
SINGLE_CLASS_CROPS = {"blueberry", "raspberry", "soybean", "orange", "squash"}
CHAMPION_RUN_ID = "fb731412d82544e0abc231b8d7c62f95"


def _infer(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            logits_all.append(model(imgs).cpu())
            labels_all.append(lbls)
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def _within_crop_macro_f1(labels, preds, idx_to_class) -> dict:
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    rec = recall_score(labels, preds, average=None, zero_division=0)
    # per-class F1 excluding single-class crops (same definition as evaluate.py)
    f1s = f1_score(labels, preds, average=None, zero_division=0)
    multi = []
    for i, name in enumerate(class_names):
        crop, _ = parse_class_name(name)
        if crop not in SINGLE_CLASS_CROPS:
            multi.append(float(f1s[i]))
    crop_true, crop_pred = defaultdict(list), defaultdict(list)
    for y, p in zip(labels, preds):
        crop, _ = parse_class_name(idx_to_class[int(y)])
        crop_true[crop].append(int(y))
        crop_pred[crop].append(int(p))
    per_crop = {}
    for crop in sorted(crop_true):
        ct = np.array(crop_true[crop])
        cp = np.array(crop_pred[crop])
        per_crop[crop] = {
            "n": int(len(ct)),
            "macro_f1_scoped": round(
                float(f1_score(ct, cp, labels=np.unique(ct), average="macro", zero_division=0)),
                4,
            ),
            "is_single_class": crop in SINGLE_CLASS_CROPS,
        }
    return {
        "full_macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "within_crop_macro_f1_excluding_single_class_crops": round(float(np.mean(multi)), 4),
        "per_crop": per_crop,
        "per_class_recall": {
            class_names[i]: {
                "recall": round(float(rec[i]), 4),
                "n": int((labels == i).sum()),
            }
            for i in range(len(class_names))
        },
    }


def _mlflow_history():
    out = {
        "champion_run_id": CHAMPION_RUN_ID,
        "run_found": False,
        "epoch_seconds": [],
        "mean_epoch_seconds": None,
        "run_duration_seconds": None,
        "registered_versions": [],
        "champion_alias": None,
        "error": None,
    }
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.config_loader import load_config

        cfg = load_config()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = MlflowClient()
        run = client.get_run(CHAMPION_RUN_ID)
        out["run_found"] = True
        start = run.info.start_time
        end = run.info.end_time
        if start and end:
            out["run_duration_seconds"] = round((end - start) / 1000.0, 1)
        hist = client.get_metric_history(CHAMPION_RUN_ID, "epoch_seconds")
        secs = [float(m.value) for m in hist]
        out["epoch_seconds"] = [round(s, 1) for s in secs]
        if secs:
            out["mean_epoch_seconds"] = round(float(np.mean(secs)), 1)
            out["sum_epoch_seconds"] = round(float(np.sum(secs)), 1)
        try:
            versions = client.search_model_versions(
                f"name='{cfg.mlflow.registered_model_name}'"
            )
            for mv in sorted(versions, key=lambda x: int(x.version)):
                out["registered_versions"].append({
                    "version": mv.version,
                    "run_id": mv.run_id,
                    "tags": dict(mv.tags or {}),
                    "aliases": list(getattr(mv, "aliases", []) or []),
                    "current_stage": getattr(mv, "current_stage", None),
                })
            champ = client.get_model_version_by_alias(
                cfg.mlflow.registered_model_name, cfg.mlflow.champion_alias
            )
            out["champion_alias"] = {"version": champ.version, "run_id": champ.run_id}
        except Exception as exc:
            out["registry_error"] = str(exc)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> int:
    set_seed(42)
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    ckpt_path = PROJECT_ROOT / "models" / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    log.info(
        f"Loaded champion checkpoint keys={list(ckpt.keys())} "
        f"run_id={ckpt.get('run_id')} epoch={ckpt.get('epoch')} "
        f"(read-only; not rewritten)"
    )

    model = build_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    splits = PROJECT_ROOT / "data" / "splits"
    val_ds = PlantVillageDataset(
        splits / "val.csv", transform=get_transforms("val", cfg), class_to_idx=class_to_idx
    )
    test_ds = PlantVillageDataset(
        splits / "test.csv", transform=get_transforms("test", cfg), class_to_idx=class_to_idx
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    log.info(f"VAL n={len(val_ds)} TEST n={len(test_ds)} (JPEG decode, num_workers=0)")
    t0 = time.perf_counter()
    val_logits, val_labels_t = _infer(model, val_loader, device)
    val_infer_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_logits, test_labels_t = _infer(model, test_loader, device)
    test_infer_s = time.perf_counter() - t0
    log.info(f"Inference wall-clock VAL={val_infer_s:.1f}s TEST={test_infer_s:.1f}s")

    scratch = PROJECT_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    np.save(scratch / "val_logits.npy", val_logits.numpy())
    np.save(scratch / "val_labels.npy", val_labels_t.numpy())
    np.save(scratch / "test_logits.npy", test_logits.numpy())
    np.save(scratch / "test_labels.npy", test_labels_t.numpy())

    val_labels = val_labels_t.numpy().astype(np.int64)
    test_labels = test_labels_t.numpy().astype(np.int64)
    val_np = val_logits.numpy()
    test_np = test_logits.numpy()

    # A1 temperature refit on the SAME VAL logits used below
    log.info("Fitting temperature on saved VAL logits...")
    fit = fit_temperature(val_logits.to(device), val_labels_t.to(device))
    fit["historical_source"] = (
        "Earlier scripts/evaluate.py used a non-unit LBFGS initialization and "
        "stored a softened diagnostic fit; the value was not accepted as a "
        "serving calibration."
    )
    fit["checkpoint"] = {
        "path": "models/best_model.pth",
        "run_id": ckpt.get("run_id"),
        "epoch": ckpt.get("epoch"),
        "preprocessing": "Resize(256)+CenterCrop(224)+ImageNet normalize (get_transforms val/test)",
    }
    fit["same_logits_as_evaluation"] = True

    t_star = fit["optimal_T"]
    splits_cal = {}
    for split_name, logits_np, labels in (
        ("val", val_np, val_labels),
        ("test", test_np, test_labels),
    ):
        pre = softmax_np(logits_np, 1.0)
        post = softmax_np(logits_np, t_star)
        splits_cal[split_name] = {
            "before_T1": calibration_bundle(pre, labels),
            "after_optimal_T": calibration_bundle(post, labels),
        }
    fit["metrics"] = splits_cal

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "temperature_refit.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    log.info(f"A1 optimal T={fit['optimal_T']} source={fit['optimal_source']}")

    # A3 VAL classification + masked background
    val_preds = np.argmax(val_np, axis=1)
    crop_block = _within_crop_macro_f1(val_labels, val_preds, idx_to_class)
    rec_map = crop_block["per_class_recall"]
    min_cls = min(rec_map.items(), key=lambda kv: (kv[1]["recall"], -kv[1]["n"]))
    min_name, min_info = min_cls
    tp_est = int(round(min_info["recall"] * min_info["n"]))
    tiny = {
        "driving_class": min_name,
        "n_val": min_info["n"],
        "recall_point": min_info["recall"],
        "wilson_lower": round(wilson_lower_bound(tp_est, min_info["n"]), 4),
        "one_extra_error_on_driving_class": one_extra_error_recall(
            min_info["n"], current_errors=min_info["n"] - tp_est
        ),
        "example_n_23": {
            "note": "Potato___healthy TEST support is 23 in reports/eval_summary.json",
            "perfect_recall": one_extra_error_recall(23, current_errors=0),
            "already_one_error": one_extra_error_recall(23, current_errors=1),
        },
        "all_classes_n_lt_30": {
            k: v for k, v in rec_map.items() if v["n"] < 30
        },
        "proposed_rules": {
            "point_estimate": "reject if min recall < 0.85 (currently in gate_thresholds.yaml, pending)",
            "wilson_lower_bound": "reject if Wilson 95% lower bound of recall < 0.85",
            "minimum_n": "apply the floor only when n_val >= n_min (e.g. 50); smaller classes logged not gated",
        },
    }
    (reports / "tiny_class_recall_analysis.json").write_text(
        json.dumps(tiny, indent=2), encoding="utf-8"
    )

    log.info("VAL/TEST masked-background eval (ExG mask, three-fill protocol)...")
    t0 = time.perf_counter()
    val_masked = evaluate_mask_fill_protocol(
        model, val_ds.samples, device, image_size=cfg.data.image_size, batch_size=32
    )
    test_masked = evaluate_mask_fill_protocol(
        model, test_ds.samples, device, image_size=cfg.data.image_size, batch_size=32
    )
    masked_s = time.perf_counter() - t0
    for result in (val_masked, test_masked):
        for fill_result in result.values():
            fill_result["command"] = "python scripts/compute_val_diagnostics.py"
    log.info(
        "VAL bg_drop by fill: "
        + ", ".join(f"{name}={value['bg_drop']}" for name, value in val_masked.items())
        + f" ({masked_s:.1f}s)"
    )

    val_ece = splits_cal["val"]["before_T1"]["ece"]
    baselines = {
        "split": "val",
        "n": len(val_ds),
        "checkpoint": "models/best_model.pth",
        "mlflow_run_id": ckpt.get("run_id", CHAMPION_RUN_ID),
        "clean_macro_f1": crop_block["full_macro_f1"],
        "within_crop_macro_f1_excluding_single_class_crops": crop_block[
            "within_crop_macro_f1_excluding_single_class_crops"
        ],
        "per_crop": crop_block["per_crop"],
        "masked_background": val_masked["black"],
        "masked_background_by_fill": val_masked,
        "masked_background_test_by_fill": test_masked,
        "min_recall": {
            "value": min_info["recall"],
            "class": min_name,
            "n": min_info["n"],
            "wilson_lower": tiny["wilson_lower"],
        },
        "calibration": {
            "ece_before": val_ece,
            "ece": val_ece,
            "temperature_applied": False,
        },
        "inference_seconds": {"val": round(val_infer_s, 1), "test": round(test_infer_s, 1)},
    }
    (reports / "gate_baselines_val.json").write_text(
        json.dumps(baselines, indent=2), encoding="utf-8"
    )

    mlflow_info = _mlflow_history()
    mlflow_info["synthetic_20step_benchmark"] = {
        "source": "scripts/diagnose_calibration_and_crops.py",
        "included_real_data_loading": False,
        "description": (
        "The historical benchmark used torch.randn tensors already on GPU; "
        "it was not a real data-loading benchmark. The measured champion run "
        "was 287.6 seconds per epoch and 2321.9 seconds wall-clock for 8 epochs."
        ),
        "val_jpeg_inference_seconds_this_run": round(val_infer_s, 1),
        "test_jpeg_inference_seconds_this_run": round(test_infer_s, 1),
        "training_num_workers": cfg.training.num_workers,
    }
    (reports / "mlflow_champion_history.json").write_text(
        json.dumps(mlflow_info, indent=2), encoding="utf-8"
    )

    log.info("Wrote reports/temperature_refit.json, gate_baselines_val.json, "
             "tiny_class_recall_analysis.json, mlflow_champion_history.json")
    print(json.dumps({
        "A1_optimal_T": fit["optimal_T"],
        "A1_val_nll_T1": fit["val_nll_at_T1"],
        "A1_val_nll_optimal": fit["val_nll_at_optimal_T"],
        "A3_val_clean_macro_f1": baselines["clean_macro_f1"],
        "A3_val_masked_macro_f1_black": val_masked["black"]["masked_macro_f1"],
        "A3_val_bg_drop_black": val_masked["black"]["bg_drop"],
        "A3_val_within_crop_macro_f1": baselines["within_crop_macro_f1_excluding_single_class_crops"],
        "A5_mean_epoch_seconds": mlflow_info.get("mean_epoch_seconds"),
        "A5_run_duration_seconds": mlflow_info.get("run_duration_seconds"),
        "A7_champion_alias": mlflow_info.get("champion_alias"),
        "A7_n_versions": len(mlflow_info.get("registered_versions") or []),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
