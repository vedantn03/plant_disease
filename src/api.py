"""
api.py
======
Step 4: FastAPI serving layer for the plant disease pipeline.

Endpoints:
  POST /predict   – multipart image + optional temp/humidity
  GET  /health    – liveness check
  GET  /stats     – request count, mean latency, class distribution, mean confidence
  GET  /metrics   – Prometheus metrics (requires prometheus-client)

Request logging:
  Every request is appended as a JSON line to cfg.serving.log_file.
  Fields: timestamp, predicted_class, confidence, brightness, blur_score, latency_ms, status

Model loading:
  Resolves the configured MLflow champion alias, then loads the matching,
  MD5-verified checkpoint from the mounted models/ volume via its manifest.
"""

import io
import asyncio
import hmac
import ipaddress
import json
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    class _MetricStub:
        def labels(self, **_labels):
            return self

        def inc(self, *_args, **_kwargs):
            return None

        def observe(self, *_args, **_kwargs):
            return None

        def set(self, *_args, **_kwargs):
            return None

        def remove(self, *_args, **_kwargs):
            return None

    def _metric_stub(*_args, **_kwargs):
        return _MetricStub()

    Counter = Gauge = Histogram = _metric_stub
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest():
        return b"# prometheus-client is not installed\n"

# Project imports (run from plant_disease/ root)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.advisory_engine import AdvisoryEngine, DISCLAIMER
from src.background_check import build_trust_response_fields, run_background_sensitivity
from src.calibration import probs_from_logits
from src.config_loader import load_config
from src.image_features import blur_score, image_brightness
from src.model import GradCAM, build_model, heatmap_to_base64_png, preprocess_image
from src.model_loader import resolve_serving_checkpoint
from src.ood_detector import check_image_heuristics
from src.utils import ensure_dir, get_logger, parse_class_name

log = get_logger("api", level=logging.INFO)

PREDICT_REQUESTS = Counter(
    "predict_requests_total", "Prediction requests by result status and source",
    ["status", "traffic_source"],
)
ABSTAIN_REQUESTS = Counter(
    "predict_abstain_total", "Uncertain or rejected requests by reason and source",
    ["reason", "traffic_source"],
)
BG_CHECK_TOTAL = Counter(
    "bg_check_total", "Background sensitivity check outcomes", ["result"]
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds", "Prediction request latency in seconds", ["traffic_source"]
)
PREDICT_CONFIDENCE = Histogram(
    "predict_confidence", "Returned prediction confidence", ["traffic_source"]
)
MODEL_INFO = Gauge("model_info", "Loaded model identity", ["version", "model_source"])

# ─────────────────────────────────────────────────────────────────────────────
# App init & model loading
# ─────────────────────────────────────────────────────────────────────────────

# State (loaded on startup)
_state: dict = {
    "model": None,
    "grad_cam": None,
    "class_to_idx": None,
    "idx_to_class": None,
    "crop_to_classes": defaultdict(list),
    "cfg": None,
    "advisory_engine": None,
    "coverage": None,
    "request_count": 0,
    "total_latency_ms": 0.0,
    "class_counts": defaultdict(int),
    "confidence_sum": 0.0,
    "log_file": None,
    "model_provenance": {},
    "registry_match": "unknown",
    "checkpoint_md5": None,
    "registry_error": None,
    "strict_blocked": False,
    "model_version": "unknown",
}

_reload_lock = asyncio.Lock()


def _is_loopback_host(host: str | None) -> bool:
    """Return true only for IP loopback addresses."""
    try:
        return bool(host) and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_local_admin_request(request: Request) -> bool:
    client_host = request.client.host if request.client else None
    request_host = request.url.hostname
    local_request_host = str(request_host).lower() in {"localhost", "127.0.0.1", "::1"}
    try:
        client_ip = ipaddress.ip_address(client_host) if client_host else None
    except ValueError:
        return False
    # Docker Desktop forwards host-loopback requests from a private bridge address.
    # Compose binds the published API port to 127.0.0.1, and the admin token is still required.
    return bool(client_ip and (client_ip.is_loopback or client_ip.is_private) and local_request_host)


def _authorize_admin_reload(request: Request, admin_token: str | None) -> None:
    """Enforce local request origin and the configured reload token."""
    if not _is_local_admin_request(request):
        raise HTTPException(status_code=403, detail="Admin reload is local-use only")
    expected_token = os.environ.get("ADMIN_RELOAD_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(status_code=503, detail="Admin reload token is not configured")
    if not admin_token or not hmac.compare_digest(admin_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


def _build_model_bundle(
    cfg,
    project_root: Path,
    device: str,
    *,
    checkpoint_override: Path | None = None,
    provenance_override: dict | None = None,
) -> dict:
    """Resolve and verify the current checkpoint before constructing serving state."""
    if checkpoint_override is not None:
        checkpoint_path = checkpoint_override
        provenance = provenance_override or {}
    else:
        checkpoint_path, provenance = resolve_serving_checkpoint(cfg, project_root)
    if provenance.get("registry_error"):
        raise RuntimeError(str(provenance["registry_error"]))
    actual_source = str(
        provenance.get("model_source", getattr(cfg.serving, "model_source", "local"))
    ).lower()
    if actual_source == "registry":
        if provenance.get("registry_match") is not True:
            raise RuntimeError("Registry alias or checkpoint MD5 does not match the manifest")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Serving checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    model = build_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    crop_to_classes = defaultdict(list)
    for class_name in class_to_idx:
        try:
            crop, _ = parse_class_name(class_name)
            crop_to_classes[crop.lower()].append(class_name)
        except Exception:
            crop_to_classes["unknown"].append(class_name)
    return {
        "model": model,
        "grad_cam": GradCAM(model),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "crop_to_classes": crop_to_classes,
        "model_provenance": provenance,
        "registry_match": provenance.get("registry_match", "unknown"),
        "checkpoint_md5": provenance.get("checkpoint_md5"),
        "registry_error": provenance.get("registry_error"),
        "strict_blocked": False,
        "model_version": str(
            provenance.get("registry_version") or provenance.get("manifest_version") or "unknown"
        ),
        "device": device,
    }


def _install_model_bundle(bundle: dict) -> None:
    """Swap the active serving bundle and update the model_info gauge."""
    old_grad_cam = _state.get("grad_cam")
    old_version = _state.get("model_version", "unknown")
    old_source = _state.get("model_provenance", {}).get("model_source", "unknown")
    if old_grad_cam:
        old_grad_cam.remove_hooks()
    try:
        MODEL_INFO.remove(str(old_version), str(old_source))
    except (AttributeError, KeyError, ValueError):
        pass
    _state.update(bundle)
    MODEL_INFO.labels(
        version=str(bundle["model_version"]),
        model_source=str(bundle["model_provenance"].get("model_source", "unknown")),
    ).set(1)


def _check_test_model_environment() -> str | None:
    """Return the fixture path only in test mode; reject accidental production use."""
    fixture_checkpoint = os.environ.get("PLANT_DISEASE_TEST_MODEL")
    if fixture_checkpoint and os.environ.get("ENV", "").lower() != "test":
        raise RuntimeError("PLANT_DISEASE_TEST_MODEL is permitted only when ENV=test")
    return fixture_checkpoint


async def startup() -> None:
    """Load model, config, coverage matrix, and advisory engine on startup."""
    project_root = Path(__file__).resolve().parent.parent
    fixture_checkpoint = _check_test_model_environment()
    cfg = load_config(
        project_root / "configs" / "config.yaml",
        require_data_root=False,
    )
    _state["cfg"] = cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Inference device: {device}")

    try:
        if fixture_checkpoint:
            checkpoint_path = Path(fixture_checkpoint).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError("Test fixture checkpoint does not exist")
            provenance = {
                "model_source": "test_fixture",
                "model_uri": str(checkpoint_path),
                "manifest_version": "test-fixture",
                "checkpoint_md5": hashlib.md5(
                    checkpoint_path.read_bytes(), usedforsecurity=False
                ).hexdigest(),
                "registry_match": "unknown",
            }
            bundle = _build_model_bundle(
                cfg,
                project_root,
                device,
                checkpoint_override=checkpoint_path,
                provenance_override=provenance,
            )
        else:
            bundle = _build_model_bundle(cfg, project_root, device)
        _install_model_bundle(bundle)
        log.info(
            "Model loaded: %s classes, device=%s, source=%s",
            len(_state["class_to_idx"]), device,
            _state["model_provenance"].get("model_source"),
        )
    except Exception as exc:
        _state.update({
            "model": None,
            "grad_cam": None,
            "model_provenance": {
                "model_source": str(getattr(cfg.serving, "model_source", "unknown")),
                "registry_error": str(exc),
            },
            "registry_match": False,
            "checkpoint_md5": None,
            "registry_error": str(exc),
            "strict_blocked": True,
            "model_version": "unknown",
        })
        log.error("Serving model resolution failed: %s", exc)

    # Advisory engine
    _state["advisory_engine"] = AdvisoryEngine(project_root / "kb" / "advisories.yaml")

    # Load coverage matrix
    cov_path = project_root / "kb" / "coverage.yaml"
    if cov_path.exists():
        with open(cov_path, "r", encoding="utf-8") as f:
            _state["coverage"] = yaml.safe_load(f)
    else:
        _state["coverage"] = {}

    # Log file
    log_path = project_root / cfg.serving.log_file
    ensure_dir(log_path.parent)
    _state["log_file"] = open(log_path, "a", encoding="utf-8", buffering=1)
    log.info(f"Request log: {log_path}")


async def shutdown() -> None:
    if _state["log_file"]:
        _state["log_file"].close()
    if _state["grad_cam"]:
        _state["grad_cam"].remove_hooks()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(
    title="Plant Disease Advisory API",
    description=(
        "Classifies leaf images into PlantVillage classes and returns a "
        "treatment advisory from a cited knowledge base. "
        "NOT a field-ready diagnostic tool. See /docs for full schema."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Image quality helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    model_loaded = _state["model"] is not None
    model_version = _state.get("model_version", "unknown")
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": _state.get("device", "unknown"),
        "model_source": _state.get("model_provenance", {}).get("model_source", "unknown"),
        "model_uri": _state.get("model_provenance", {}).get("model_uri", "unknown"),
        "model_version": model_version,
        "checkpoint_md5_prefix": str(_state.get("checkpoint_md5") or "")[:8] or None,
        "registry_match": _state.get("registry_match", "unknown"),
        "registry_error": _state.get("registry_error"),
        "registry_strict_blocked": _state.get("strict_blocked", False),
    }


@app.post("/admin/reload")
async def admin_reload(
    request: Request,
    admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    """Reload the registry alias target from a verified local checkpoint."""
    _authorize_admin_reload(request, admin_token)

    async with _reload_lock:
        cfg = _state.get("cfg")
        if cfg is None:
            raise HTTPException(status_code=503, detail="API configuration is not loaded")
        project_root = Path(__file__).resolve().parent.parent
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            bundle = _build_model_bundle(cfg, project_root, device)
        except Exception as exc:
            old_grad_cam = _state.get("grad_cam")
            old_version = str(_state.get("model_version", "unknown"))
            old_source = str(
                _state.get("model_provenance", {}).get("model_source", "unknown")
            )
            if old_grad_cam:
                old_grad_cam.remove_hooks()
            try:
                MODEL_INFO.remove(old_version, old_source)
            except (AttributeError, KeyError, ValueError):
                pass
            _state.update({
                "model": None,
                "grad_cam": None,
                "model_provenance": {
                    "model_source": str(getattr(cfg.serving, "model_source", "unknown")),
                    "registry_error": str(exc),
                },
                "registry_match": False,
                "checkpoint_md5": None,
                "registry_error": str(exc),
                "strict_blocked": True,
                "model_version": "unknown",
            })
            raise HTTPException(
                status_code=503,
                detail=f"Reload refused; serving disabled: {exc}",
            ) from exc
        _install_model_bundle(bundle)
        return {
            "status": "reloaded",
            "model_version": bundle["model_version"],
            "model_source": bundle["model_provenance"].get("model_source"),
            "checkpoint_md5_prefix": str(bundle["checkpoint_md5"] or "")[:8] or None,
            "registry_match": bundle["registry_match"],
        }


@app.get("/metrics")
async def metrics() -> Response:
    if not PROMETHEUS_AVAILABLE:
        return Response(
            "prometheus-client is not installed\n",
            status_code=503,
            media_type=CONTENT_TYPE_LATEST,
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────────────────────────
# /stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/stats")
async def stats() -> dict:
    n = _state["request_count"]
    return {
        "request_count": n,
        "mean_latency_ms": round(_state["total_latency_ms"] / n, 2) if n > 0 else 0.0,
        "mean_confidence": round(_state["confidence_sum"] / n, 4) if n > 0 else 0.0,
        "class_distribution": dict(_state["class_counts"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    crop: Optional[str] = Form(default=None),
    temperature_c: Optional[float] = Form(default=None),
    humidity_pct: Optional[float] = Form(default=None),
    traffic_source: Optional[str] = Form(default=None),
) -> JSONResponse:
    t_start = time.perf_counter()
    source_header = "replay" if (
        traffic_source == "replay" or request.headers.get("X-Traffic-Source") == "replay"
    ) else "api"
    cfg = _state["cfg"]

    # ── Validate model is loaded ──────────────────────────────────────────────
    if _state["model"] is None:
        PREDICT_REQUESTS.labels(status="unavailable", traffic_source=source_header).inc()
        PREDICT_LATENCY.labels(traffic_source=source_header).observe(
            time.perf_counter() - t_start
        )
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    # ── Validate file type and size ───────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > cfg.serving.max_upload_bytes:
        max_mb = cfg.serving.max_upload_bytes / 1e6
        raise HTTPException(status_code=413, detail=f"File too large (max {max_mb:.0f} MB).")

    # ── Load & validate image ─────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    # Image quality features (for request log / drift detection)
    brightness = image_brightness(img)
    b_score = blur_score(img)
    bg_result = {
        "result": "unknown",
        "reason": "not_evaluated",
        "mask_area_fraction": None,
        "original_label": None,
        "neutralized_label": None,
        "original_confidence": None,
        "neutralized_confidence": None,
        "confidence_change": None,
        "neutralized_image_base64": None,
        "elapsed_ms": 0.0,
    }

    # ── Layer 1 OOD & Heuristic Pre-Check ────────────────────────────────────
    is_valid_leaf, reason_code, reason_msg = check_image_heuristics(img)
    if not is_valid_leaf:
        bg_result["reason"] = "ood_rejected"
        BG_CHECK_TOTAL.labels(result="unknown").inc()
        trust_fields = build_trust_response_fields(
            brightness=brightness,
            blur_score=b_score,
            ood_status="rejected",
            bg_result=bg_result,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000
        _record_prediction("rejected_not_a_leaf", latency_ms, 0.0, reason_code, source_header)
        _log_request(
            "REJECTED_OOD", 0.0, brightness, b_score, latency_ms,
            "rejected_not_a_leaf", source_header,
        )
        return JSONResponse({
            "status": "rejected_not_a_leaf",
            "reason_code": reason_code,
            "message": reason_msg,
            "confidence": 0.0,
            "top_k": [],
            "latency_ms": round(latency_ms, 1),
            "disclaimer": DISCLAIMER,
            **trust_fields,
        })

    # ── Inference ─────────────────────────────────────────────────────────────
    model = _state["model"]
    device = _state["device"]
    idx_to_class = _state["idx_to_class"]
    k = cfg.serving.top_k
    threshold = cfg.serving.confidence_threshold

    input_tensor = preprocess_image(img, cfg.data.image_size).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(input_tensor)

    # Unscaled softmax (T=1). Stored temperature is NOT applied at serving.
    # ── Crop-Aware Filtering / Hierarchical Inference ─────────────────────────
    raw_probs = probs_from_logits(logits, temperature=1.0).squeeze()  # (num_classes,)

    # Calculate crop-level aggregate probabilities
    crop_probs = defaultdict(float)
    for idx, prob_val in enumerate(raw_probs.cpu().numpy()):
        cls_name = idx_to_class[idx]
        try:
            c_name, _ = parse_class_name(cls_name)
        except Exception:
            c_name = "unknown"
        crop_probs[c_name.lower()] += float(prob_val)

    # If specific crop is requested, mask out other classes
    target_crop = crop.strip().lower() if crop and crop.strip().lower() not in {"auto", "all", ""} else None

    if target_crop and target_crop in _state["crop_to_classes"]:
        valid_classes = _state["crop_to_classes"][target_crop]
        masked_logits = logits.clone()
        for idx in range(len(idx_to_class)):
            if idx_to_class[idx] not in valid_classes:
                masked_logits[0, idx] = -1e9
        effective_probs = F.softmax(masked_logits, dim=1).squeeze()
        effective_k = min(k, len(valid_classes))
    else:
        effective_probs = raw_probs
        effective_k = k

    top_probs, top_idxs = effective_probs.topk(effective_k)
    top_probs = top_probs.cpu().numpy().tolist()
    top_idxs = top_idxs.cpu().numpy().tolist()

    top_k_list = [
        {"label": idx_to_class[i], "prob": round(float(p), 4)}
        for i, p in zip(top_idxs, top_probs)
    ]

    best_class = idx_to_class[top_idxs[0]]
    best_prob = top_probs[0]

    bg_cfg = cfg.background_check
    bg_result = run_background_sensitivity(
        img,
        model,
        device=device,
        image_size=cfg.data.image_size,
        idx_to_class=idx_to_class,
        original_label=best_class,
        original_confidence=best_prob,
        min_mask_area_fraction=float(bg_cfg.min_mask_area_fraction),
        max_mask_area_fraction=float(bg_cfg.max_mask_area_fraction),
        allowed_class_names=(
            valid_classes if target_crop and target_crop in _state["crop_to_classes"] else None
        ),
        enabled=bool(bg_cfg.bg_check_enabled),
    )
    bg_result_name = bg_result.get("result", "unknown")
    if bg_result_name not in {"consistent", "inconsistent", "unknown"}:
        bg_result_name = "unknown"
    BG_CHECK_TOTAL.labels(result=bg_result_name).inc()
    trust_fields = build_trust_response_fields(
        brightness=brightness,
        blur_score=b_score,
        ood_status="accepted",
        bg_result=bg_result,
    )

    # Parse predicted crop and condition
    try:
        pred_crop, pred_condition = parse_class_name(best_class)
    except ValueError:
        pred_crop, pred_condition = best_class, "unknown"

    # Coverage matrix caveats (structural gaps check)
    coverage_caveat = None
    if _state.get("coverage") and "crops" in _state["coverage"]:
        for c_key, c_info in _state["coverage"]["crops"].items():
            if c_key.lower() == pred_crop.lower():
                coverage_caveat = c_info.get("caveat")
                break

    if bool(bg_cfg.abstain_on_bg_inconsistent) and bg_result_name == "inconsistent":
        latency_ms = (time.perf_counter() - t_start) * 1000
        reason = "background_sensitivity_inconsistent"
        _record_prediction("uncertain", latency_ms, best_prob, reason, source_header)
        _log_request(
            best_class, best_prob, brightness, b_score, latency_ms, "uncertain", source_header
        )
        _update_stats(best_class, best_prob, latency_ms)
        return JSONResponse({
            "status": "uncertain",
            "reason_code": reason,
            "message": "Prediction abstained because background sensitivity was inconsistent.",
            "crop": pred_crop,
            "condition": pred_condition,
            "confidence": round(best_prob, 4),
            "top_k": top_k_list,
            "coverage_caveat": coverage_caveat,
            "disclaimer": DISCLAIMER,
            **trust_fields,
        })

    # ── Rejection / Uncertainty Check ─────────────────────────────────────────
    if best_prob < threshold:
        latency_ms = (time.perf_counter() - t_start) * 1000
        _record_prediction("uncertain", latency_ms, best_prob, "confidence_below_threshold", source_header)
        _log_request(best_class, best_prob, brightness, b_score, latency_ms, "uncertain", source_header)
        _update_stats(best_class, best_prob, latency_ms)
        return JSONResponse({
            "status": "uncertain",
            "reason_code": "confidence_below_threshold",
            "message": (
                f"Diagnostic confidence ({best_prob:.1%}) is below the minimum threshold ({threshold:.1%}). "
                "Please submit a sharper, well-lit photograph or confirm the crop type explicitly."
            ),
            "crop": pred_crop,
            "condition": pred_condition,
            "confidence": round(best_prob, 4),
            "top_k": top_k_list,
            "coverage_caveat": coverage_caveat,
            "disclaimer": DISCLAIMER,
            **trust_fields,
        })

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_b64 = ""
    try:
        grad_cam = _state["grad_cam"]
        heatmap = grad_cam.generate(input_tensor.requires_grad_(True), class_idx=top_idxs[0])
        gradcam_b64 = heatmap_to_base64_png(heatmap, img)
    except Exception as exc:
        log.warning(f"Grad-CAM failed: {exc}")

    # ── Advisory ──────────────────────────────────────────────────────────────
    advisory = _state["advisory_engine"].get_advisory(
        pred_crop, pred_condition, temperature_c, humidity_pct
    )

    # ── Build response ────────────────────────────────────────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000
    _record_prediction("ok", latency_ms, best_prob, "none", source_header)
    _log_request(best_class, best_prob, brightness, b_score, latency_ms, "ok", source_header)
    _update_stats(best_class, best_prob, latency_ms)

    return JSONResponse({
        "status": "ok",
        "crop": pred_crop,
        "condition": pred_condition,
        "confidence": round(best_prob, 4),
        "in_crop_constrained": bool(target_crop),
        "top_k": top_k_list,
        "coverage_caveat": coverage_caveat,
        "gradcam_png_base64": gradcam_b64,
        "advisory": advisory,
        "latency_ms": round(latency_ms, 1),
        "disclaimer": DISCLAIMER,
        **trust_fields,
    })


# ─────────────────────────────────────────────────────────────────────────────
# /predict_batch
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    crop: Optional[str] = Form(default=None),
) -> JSONResponse:
    """Batch inference endpoint for processing multiple leaves in a single call."""
    t_start = time.perf_counter()
    batch_results = []

    for f in files:
        raw = await f.read()
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            is_valid, code, msg = check_image_heuristics(im)
            if not is_valid:
                batch_results.append({
                    "filename": f.filename,
                    "status": "rejected_not_a_leaf",
                    "reason_code": code,
                    "crop": "unknown",
                    "condition": "unknown",
                    "confidence": 0.0,
                })
                continue

            input_tensor = preprocess_image(im, _state["cfg"].data.image_size).to(_state["device"])
            with torch.no_grad():
                with torch.autocast(device_type=_state["device"] if _state["device"] != "cpu" else "cpu", dtype=torch.float16, enabled=(_state["device"] == "cuda")):
                    logits = _state["model"](input_tensor)
            probs = probs_from_logits(logits, temperature=1.0).squeeze()
            top_prob, top_idx = probs.max(dim=0)
            cls_name = _state["idx_to_class"][top_idx.item()]
            c, cond = parse_class_name(cls_name)

            batch_results.append({
                "filename": f.filename,
                "status": "ok" if top_prob.item() >= 0.50 else "uncertain",
                "crop": c,
                "condition": cond,
                "confidence": round(top_prob.item(), 4),
            })
        except Exception as e:
            batch_results.append({
                "filename": f.filename,
                "status": "error",
                "error": str(e)
            })

    total_latency_ms = (time.perf_counter() - t_start) * 1000
    return JSONResponse({
        "batch_size": len(files),
        "total_latency_ms": round(total_latency_ms, 1),
        "throughput_images_per_sec": round(len(files) / (total_latency_ms / 1000.0), 1) if total_latency_ms > 0 else 0.0,
        "results": batch_results,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_request(
    predicted_class: str,
    confidence: float,
    brightness: float,
    b_score: float,
    latency_ms: float,
    status: str,
    traffic_source: str = "api",
) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "brightness": round(brightness, 2),
        "blur_score": round(b_score, 2),
        "latency_ms": round(latency_ms, 1),
        "status": status,
        "traffic_source": traffic_source,
    }
    if _state["log_file"]:
        _state["log_file"].write(json.dumps(record) + "\n")


def _update_stats(cls: str, confidence: float, latency_ms: float) -> None:
    _state["request_count"] += 1
    _state["total_latency_ms"] += latency_ms
    _state["class_counts"][cls] += 1
    _state["confidence_sum"] += confidence


def _record_prediction(
    status: str, latency_ms: float, confidence: float, reason: str, traffic_source: str
) -> None:
    PREDICT_REQUESTS.labels(status=status, traffic_source=traffic_source).inc()
    PREDICT_LATENCY.labels(traffic_source=traffic_source).observe(latency_ms / 1000.0)
    PREDICT_CONFIDENCE.labels(traffic_source=traffic_source).observe(confidence)
    if status in {"uncertain", "rejected_not_a_leaf"}:
        ABSTAIN_REQUESTS.labels(reason=reason or "unspecified", traffic_source=traffic_source).inc()
