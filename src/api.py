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

import asyncio
import hashlib
import hmac
import io
import ipaddress
import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
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

from src.advisory_engine import AdvisoryEngine, DISCLAIMER, decide_advice, evidence_grade
from src.background_check import build_trust_response_fields, run_background_sensitivity
from src.calibration import probs_from_logits
from src.config_loader import load_config
from src.explainability import (
    ExplanationJobStore,
    ExplanationQueueFull,
    ExplanationUnavailable,
    generate_explanation,
)
from src.image_features import blur_score, image_brightness
from src.insights import build_insights, latest_mlflow_run
from src.model import GradCAM, build_model, heatmap_to_base64_png, preprocess_image
from src.model_loader import resolve_serving_checkpoint
from src.ood_detector import check_image_heuristics
from src.presentation import build_prediction_views
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
ADVICE_TOTAL = Counter(
    "advice_total", "Advice responses by tier and evidence grade", ["tier", "evidence_grade"]
)
ADVICE_WITHHELD_TOTAL = Counter(
    "advice_withheld_total", "Advice withheld by reason", ["reason"]
)
ADVICE_GAP_TOTAL = Counter(
    "advice_gap_total", "Predicted classes without grade A or B advice", ["class"]
)

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
    "device": "unknown",
    "explanation_jobs": None,
    "started_at": None,
}

_reload_lock = asyncio.Lock()
_model_inference_lock = threading.RLock()


def _config_section(config, key: str, default=None):
    """Read a config section from the project's AttrDict or a test namespace."""
    if config is None:
        return {} if default is None else default
    if hasattr(config, "get"):
        return config.get(key, {} if default is None else default)
    return getattr(config, key, {} if default is None else default)


def _config_value(section, key: str, default):
    """Read one key from the project's mapping or a lightweight test namespace."""
    return section.get(key, default) if hasattr(section, "get") else getattr(section, key, default)


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
    _state["started_at"] = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["device"] = device
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
    advisory_cfg = getattr(cfg, "advisory", {})
    confused_pairs_path = (
        advisory_cfg.get("confused_pairs_path", "reports/confused_pairs.csv")
        if hasattr(advisory_cfg, "get") else "reports/confused_pairs.csv"
    )
    _state["advisory_engine"] = AdvisoryEngine(
        project_root / "kb" / "advisories.yaml",
        project_root / "kb" / "weather_rules.yaml",
        project_root / confused_pairs_path,
    )

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
    explain_cfg = _config_section(cfg, "explainability")
    _state["explanation_jobs"] = ExplanationJobStore(
        workers=int(explain_cfg.get("workers", 1)),
        cache_size=int(explain_cfg.get("cache_size", 0)),
        timeout_s=float(explain_cfg.get("timeout_s", 60)),
        max_pending=int(explain_cfg.get("max_pending_jobs", 1)),
        max_records=int(explain_cfg.get("max_job_records", 100)),
    )
    log.info(f"Request log: {log_path}")


async def shutdown() -> None:
    if _state.get("explanation_jobs"):
        _state["explanation_jobs"].shutdown()
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Traffic-Source"],
)


UI_REPORT_ALLOWLIST = frozenset({
    "drift_latest.json",
    "promotion_log.jsonl",
    "seed_variance.json",
    "candidate_gate_comparison.json",
})


def _prediction_presentation(
    predicted_class: str | None,
    confidence: float | None,
    status: str,
    trust_fields: dict,
    advice_payload: dict,
) -> dict:
    """Build additive farmer and developer views without changing raw outputs."""
    trace = advice_payload.get("advice_trace", {})
    engine = _state.get("advisory_engine")
    entry = None
    if engine and predicted_class:
        lookup = getattr(engine, "get_advisory_from_class_name", None)
        if callable(lookup):
            entry = lookup(predicted_class)
        else:
            try:
                entry_crop, entry_condition = parse_class_name(predicted_class)
                entry = engine.get_entry(entry_crop, entry_condition)
            except (ValueError, AttributeError):
                entry = None
    config = _state.get("cfg")
    serving_cfg = _config_section(config, "serving")
    if not isinstance(serving_cfg, dict):
        serving_cfg = vars(serving_cfg)
    crop, condition = (None, None)
    if predicted_class:
        try:
            crop, condition = parse_class_name(predicted_class)
        except ValueError:
            pass
    views = build_prediction_views(
        predicted_class=predicted_class,
        confidence=confidence,
        status=status,
        ood_status=str(trust_fields.get("ood_status", "unknown")),
        bg_check=str(trust_fields.get("bg_check", "unknown")),
        quality=trust_fields.get("quality"),
        coverage=_state.get("coverage"),
        confidence_bands=serving_cfg.get("confidence_bands", {}),
        pathogen_type=entry.get("pathogen_type") if isinstance(entry, dict) else None,
        rules_fired=trace.get("rules_fired", []),
        kb_version=str(trace.get("kb_version") or _kb_version()),
        evidence_grade=trace.get("evidence_grade"),
    )
    display = views["display"]
    display["crop"]["label"] = display["crop"]["label"] or (crop.title() if crop else None)
    display["condition"]["label"] = (
        display["condition"]["label"] or (condition.replace("_", " ").title() if condition else None)
    )
    return views


def resolve_ui_report(report_name: str) -> Path:
    """Resolve only a fixed set of read-only report names for the local UI."""
    if report_name not in UI_REPORT_ALLOWLIST:
        raise HTTPException(status_code=404, detail="Report not available")
    path = Path(__file__).resolve().parent.parent / "reports" / report_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not generated yet")
    return path


@app.get("/ui-data/telemetry", include_in_schema=False)
def ui_telemetry(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    """Return bounded, read-only request-log summaries for the local UI."""
    path = Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"
    if not path.is_file():
        return {
            "available": False,
            "message": "No request log has been generated yet.",
            "records": [],
            "summary": {},
        }
    records: list[dict[str, object]] = []
    malformed = 0
    truncated = False

    def finite_number(value: object) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) if math.isfinite(value) else None
        return None

    def text_value(value: object) -> str | None:
        return value if isinstance(value, str) else None

    try:
        # The append-only log may grow during long local runs. Read only its
        # tail so this dashboard endpoint has bounded memory and response size.
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            read_size = min(size, 2 * 1024 * 1024)
            truncated = size > read_size
            stream.seek(size - read_size)
            tail = stream.read(read_size).decode("utf-8", errors="replace")
        lines = tail.splitlines()
        if truncated and lines:
            lines = lines[1:]  # Discard a possibly partial first record.
        for line in reversed(lines):
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    # Expose only the fields used in the UI, even if a future
                    # log record grows to include additional internal details.
                    records.append({
                        "timestamp": text_value(value.get("timestamp")),
                        "predicted_class": text_value(value.get("predicted_class")),
                        "confidence": finite_number(value.get("confidence")),
                        "latency_ms": finite_number(value.get("latency_ms")),
                        "status": text_value(value.get("status")),
                        "traffic_source": text_value(value.get("traffic_source")),
                    })
                else:
                    malformed += 1
            except (json.JSONDecodeError, TypeError):
                malformed += 1
            if len(records) >= limit:
                break
        records.reverse()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Request log cannot be read") from exc
    latencies = [row["latency_ms"] for row in records if row["latency_ms"] is not None]
    confidences = [row["confidence"] for row in records if row["confidence"] is not None]
    statuses: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in records:
        status = str(row.get("status", "unknown"))
        source = str(row.get("traffic_source", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        sources[source] = sources.get(source, 0) + 1
    serving_cfg = _config_section(_state.get("cfg"), "serving")
    minimum = int(_config_value(serving_cfg, "percentile_min_samples", 30))
    percentiles = None
    if len(latencies) >= minimum:
        ordered = sorted(latencies)
        def at_percentile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)
        percentiles = {"p50_latency_ms": at_percentile(0.50), "p95_latency_ms": at_percentile(0.95)}
    return {
        "available": bool(records),
        "message": None if records else "Request log is empty.",
        "record_count": len(records),
        "malformed_lines": malformed,
        "truncated": truncated,
        "summary": {
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "statuses": statuses,
            "traffic_sources": sources,
            "latency_percentiles": percentiles,
            "percentile_sample_count": len(latencies),
            "percentile_min_samples": minimum,
        },
        "records": records[-limit:],
    }


@app.get("/ui-data/pipeline", include_in_schema=False)
def ui_pipeline() -> dict[str, object]:
    """Expose declared DVC stage names without running or changing the pipeline."""
    path = Path(__file__).resolve().parent.parent / "dvc.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise HTTPException(status_code=503, detail="DVC pipeline definition is unavailable") from exc
    stages = document.get("stages", {})
    if not isinstance(stages, dict):
        raise HTTPException(status_code=503, detail="DVC pipeline definition is invalid")
    return {"stages": list(stages), "source": "dvc.yaml", "reproduction_run": False}


@app.get("/ui-data/{report_name}", include_in_schema=False)
async def ui_report(report_name: str) -> FileResponse:
    """Read-only allow-listed report access for the local browser dashboard."""
    return FileResponse(resolve_ui_report(report_name))


@app.get("/classes")
async def classes() -> dict:
    """Return class/crop options from the loaded serving checkpoint."""
    names = sorted((_state.get("class_to_idx") or {}).keys())
    crops: dict[str, list[str]] = defaultdict(list)
    for name in names:
        try:
            crop_name, _condition = parse_class_name(name)
            crops[crop_name].append(name)
        except ValueError:
            continue
    return {
        "available": bool(names),
        "classes": names,
        "crops": [
            {"key": key, "label": key.replace("_", " ").replace(",", "").title(), "classes": values}
            for key, values in sorted(crops.items())
        ],
    }


@app.get("/ui-config")
async def ui_config() -> dict:
    """Expose browser-only display limits from the serving configuration."""
    cfg = _state.get("cfg")
    serving = _config_section(cfg, "serving")
    return {
        "browser_history_limit": int(_config_value(serving, "browser_history_limit", 8)),
        "drift_simulation_enabled": bool(
            _config_value(_config_section(cfg, "drift"), "ui_simulation_enabled", False)
        ),
    }


@app.get("/insights")
async def insights() -> dict:
    """Read-only evaluator payload built from process stats and saved artifacts."""
    cfg = _state.get("cfg")
    limit = int(_config_value(_config_section(cfg, "serving"), "telemetry_window_size", 100))
    telemetry = ui_telemetry(min(max(1, limit), 500))
    process_stats = await stats()
    serving = await health()
    project_root = Path(__file__).resolve().parent.parent
    drift_cfg = _config_section(cfg, "drift")
    monitoring_config = {
        key: _config_value(drift_cfg, key, None)
        for key in (
            "psi_threshold", "ks_alpha", "minimum_sample_count", "window_size",
            "drift_rule", "thresholds_provisional",
        )
    }
    mlflow_uri = _config_value(_config_section(cfg, "mlflow"), "tracking_uri", None)
    mlflow_summary = await run_in_threadpool(latest_mlflow_run, mlflow_uri)
    return build_insights(
        project_root,
        serving=serving,
        process_stats=process_stats,
        telemetry=telemetry,
        class_names=sorted((_state.get("class_to_idx") or {}).keys()),
        max_report_bytes=int(_config_value(
            _config_section(cfg, "serving"), "insights_report_max_bytes", 200000
        )),
        drift_config=monitoring_config,
        mlflow_summary=mlflow_summary,
    )


def _explanation_job_response(job) -> dict:
    payload = {"job_id": job.job_id, "status": job.status, "method": job.method}
    if job.result is not None:
        payload.update(job.result)
    if job.status in {"failed", "timeout"}:
        payload.update({"reason_code": job.reason_code, "message": job.message})
    return payload


@app.post("/explain/jobs", status_code=202)
async def create_explanation_job(
    file: UploadFile = File(...),
    method: str = Form(...),
    crop: Optional[str] = Form(default=None),
) -> dict:
    """Queue a bounded LIME/SHAP job; prediction requests never wait for explainers."""
    method = method.strip().lower()
    if method not in {"lime", "shap"}:
        raise HTTPException(status_code=400, detail="method must be lime or shap")
    cfg = _state.get("cfg")
    store = _state.get("explanation_jobs")
    if cfg is None or store is None:
        raise HTTPException(status_code=503, detail="Explanation service is not initialized")
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) > int(cfg.serving.max_upload_bytes):
        raise HTTPException(status_code=413, detail="Image is empty or exceeds the upload limit")
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Cannot open the uploaded image") from exc

    explain_cfg = _config_section(cfg, "explainability")
    if not explain_cfg.get("enabled", True):
        raise HTTPException(status_code=503, detail="Explainability jobs are disabled")
    lime_cfg = explain_cfg.get("lime", {})
    shap_cfg = explain_cfg.get("shap", {})
    project_root = Path(__file__).resolve().parent.parent
    train_manifest = project_root / explain_cfg.get("train_manifest", "data/splits/train.csv")
    train_hash = (
        hashlib.sha256(train_manifest.read_bytes()).hexdigest()
        if train_manifest.is_file() else "missing"
    )
    params = {
        "seed": int(explain_cfg.get("seed", getattr(cfg, "seed", 42))),
        "image_size": int(cfg.data.image_size),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": train_hash,
        "data_root": str(cfg.data.root),
        "lime_num_samples": int(lime_cfg.get("num_samples", 100)),
        "lime_num_features": int(lime_cfg.get("num_features", 5)),
        "lime_kernel_size": int(lime_cfg.get("kernel_size", 4)),
        "lime_max_dist": int(lime_cfg.get("max_dist", 200)),
        "lime_ratio": float(lime_cfg.get("ratio", 0.2)),
        "shap_n_background": int(shap_cfg.get("n_background", 4)),
        "shap_n_samples": int(shap_cfg.get("n_samples", 32)),
        "deletion_region_fraction": float(explain_cfg.get("deletion_region_fraction", 0.10)),
    }
    model = _state.get("model")
    device = str(_state.get("device", "cpu"))
    class_index = _state.get("class_to_idx") or {}
    idx_to_class = _state.get("idx_to_class") or {}
    class_names = None
    if crop and crop.strip().lower() not in {"", "auto", "all"}:
        allowed = _state.get("crop_to_classes", {}).get(crop.strip().lower())
        if allowed:
            class_names = [class_index[name] for name in allowed if name in class_index]
    version = str(_state.get("model_version", "unknown"))

    def work(cancel_event):
        if model is None:
            raise ExplanationUnavailable("model_unavailable", "The serving model is not loaded.")
        with _model_inference_lock:
            return generate_explanation(
                method=method,
                image=image,
                model=model,
                device=device,
                idx_to_class=idx_to_class,
                model_version=version,
                params=params,
                crop_class_ids=class_names,
                cancel_event=cancel_event,
            )

    try:
        job = store.submit(
            image_bytes=raw_bytes,
            method=method,
            params=params,
            model_version=version,
            work=work,
        )
    except ExplanationQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _explanation_job_response(job)


@app.get("/explain/jobs/{job_id}")
async def get_explanation_job(job_id: str) -> dict:
    store = _state.get("explanation_jobs")
    job = store.get(job_id) if store else None
    if job is None:
        raise HTTPException(status_code=404, detail="Explanation job not found")
    return _explanation_job_response(job)


@app.delete("/explain/jobs/{job_id}")
async def cancel_explanation_job(job_id: str) -> dict:
    store = _state.get("explanation_jobs")
    job = store.cancel(job_id) if store else None
    if job is None:
        raise HTTPException(status_code=404, detail="Explanation job not found")
    return _explanation_job_response(job)


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
        "uptime_seconds": (
            max(0.0, round(time.time() - _state["started_at"], 2))
            if _state.get("started_at") else None
        ),
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
        advice_payload = _advice_for_prediction(
            "rejected", None, reason_code, None, None, "unknown", brightness, b_score,
            temperature_c, humidity_pct,
        )
        presentation = _prediction_presentation(
            None, 0.0, "rejected_not_a_leaf", trust_fields, advice_payload
        )
        return JSONResponse({
            "status": "rejected_not_a_leaf",
            "reason_code": reason_code,
            "message": reason_msg,
            "confidence": 0.0,
            "top_k": [],
            "latency_ms": round(latency_ms, 1),
            "disclaimer": DISCLAIMER,
            "kb_version": _kb_version(),
            **advice_payload,
            **trust_fields,
            **presentation,
        })

    # ── Inference ─────────────────────────────────────────────────────────────
    model = _state["model"]
    device = _state["device"]
    idx_to_class = _state["idx_to_class"]
    k = cfg.serving.top_k
    threshold = cfg.serving.confidence_threshold

    input_tensor = preprocess_image(img, cfg.data.image_size).to(device)

    def run_forward() -> torch.Tensor:
        with _model_inference_lock:
            with torch.no_grad():
                with torch.autocast(
                    device_type=device if device != "cpu" else "cpu",
                    dtype=torch.float16,
                    enabled=(device == "cuda"),
                ):
                    return model(input_tensor)

    logits = await run_in_threadpool(run_forward)

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
    def run_bg_check() -> dict:
        with _model_inference_lock:
            return run_background_sensitivity(
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
                    valid_classes
                    if target_crop and target_crop in _state["crop_to_classes"] else None
                ),
                enabled=bool(bg_cfg.bg_check_enabled),
            )

    bg_result = await run_in_threadpool(run_bg_check)
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
        advice_payload = _advice_for_prediction(
            "uncertain", best_prob, reason, pred_crop, pred_condition,
            bg_result_name, brightness, b_score, temperature_c, humidity_pct,
        )
        presentation = _prediction_presentation(
            best_class, best_prob, "uncertain", trust_fields, advice_payload
        )
        return JSONResponse({
            "status": "uncertain",
            "reason_code": reason,
            "message": "Prediction abstained because background sensitivity was inconsistent.",
            "predicted_class": best_class,
            "crop": pred_crop,
            "condition": pred_condition,
            "confidence": round(best_prob, 4),
            "top_k": top_k_list,
            "coverage_caveat": coverage_caveat,
            "disclaimer": DISCLAIMER,
            "kb_version": _kb_version(),
            **advice_payload,
            **trust_fields,
            **presentation,
        })

    # ── Rejection / Uncertainty Check ─────────────────────────────────────────
    if best_prob < threshold:
        latency_ms = (time.perf_counter() - t_start) * 1000
        _record_prediction("uncertain", latency_ms, best_prob, "confidence_below_threshold", source_header)
        _log_request(best_class, best_prob, brightness, b_score, latency_ms, "uncertain", source_header)
        _update_stats(best_class, best_prob, latency_ms)
        advice_payload = _advice_for_prediction(
            "uncertain", best_prob, "confidence_below_threshold", pred_crop,
            pred_condition, bg_result_name, brightness, b_score, temperature_c, humidity_pct,
        )
        presentation = _prediction_presentation(
            best_class, best_prob, "uncertain", trust_fields, advice_payload
        )
        return JSONResponse({
            "status": "uncertain",
            "reason_code": "confidence_below_threshold",
            "message": (
                f"Diagnostic confidence ({best_prob:.1%}) is below the minimum threshold ({threshold:.1%}). "
                "Please submit a sharper, well-lit photograph or confirm the crop type explicitly."
            ),
            "predicted_class": best_class,
            "crop": pred_crop,
            "condition": pred_condition,
            "confidence": round(best_prob, 4),
            "top_k": top_k_list,
            "coverage_caveat": coverage_caveat,
            "disclaimer": DISCLAIMER,
            "kb_version": _kb_version(),
            **advice_payload,
            **trust_fields,
            **presentation,
        })

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_b64 = ""
    try:
        grad_cam = _state["grad_cam"]
        def run_gradcam():
            with _model_inference_lock:
                return grad_cam.generate(
                    input_tensor.requires_grad_(True), class_idx=top_idxs[0]
                )

        heatmap = await run_in_threadpool(run_gradcam)
        gradcam_b64 = heatmap_to_base64_png(heatmap, img)
    except Exception as exc:
        log.warning(f"Grad-CAM failed: {exc}")

    # ── Advisory ──────────────────────────────────────────────────────────────
    advice_payload = _advice_for_prediction(
        "ok", best_prob, "none", pred_crop, pred_condition,
        bg_result_name, brightness, b_score, temperature_c, humidity_pct,
    )
    presentation = _prediction_presentation(
        best_class, best_prob, "ok", trust_fields, advice_payload
    )

    # ── Build response ────────────────────────────────────────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000
    _record_prediction("ok", latency_ms, best_prob, "none", source_header)
    _log_request(best_class, best_prob, brightness, b_score, latency_ms, "ok", source_header)
    _update_stats(best_class, best_prob, latency_ms)

    return JSONResponse({
        "status": "ok",
        "predicted_class": best_class,
        "crop": pred_crop,
        "condition": pred_condition,
        "confidence": round(best_prob, 4),
        "in_crop_constrained": bool(target_crop),
        "top_k": top_k_list,
        "coverage_caveat": coverage_caveat,
        "gradcam_png_base64": gradcam_b64,
        **advice_payload,
        "kb_version": _kb_version(),
        "latency_ms": round(latency_ms, 1),
        "disclaimer": DISCLAIMER,
        **trust_fields,
        **presentation,
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
            def run_batch_forward():
                with _model_inference_lock, torch.no_grad():
                    with torch.autocast(
                        device_type=_state["device"] if _state["device"] != "cpu" else "cpu",
                        dtype=torch.float16,
                        enabled=(_state["device"] == "cuda"),
                    ):
                        return _state["model"](input_tensor)

            logits = await run_in_threadpool(run_batch_forward)
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
        "kb_version": _kb_version(),
    }
    if _state["log_file"]:
        _state["log_file"].write(json.dumps(record) + "\n")


def _kb_version() -> str:
    engine = _state.get("advisory_engine")
    return engine.kb_version if engine is not None else "unavailable"


def _advice_for_prediction(
    status: str,
    confidence: float | None,
    reason: str,
    crop: str | None,
    condition: str | None,
    bg_check: str,
    brightness: float | None,
    blur_value: float | None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
) -> dict:
    engine = _state.get("advisory_engine")
    entry = engine.get_entry(crop, condition) if engine and crop and condition else None
    grade = evidence_grade(entry or {})
    cfg = _state.get("cfg")
    advisory_cfg = getattr(cfg, "advisory", {}) if cfg else {}
    cutoffs = advisory_cfg.get("confidence_cutoffs", {}) if hasattr(advisory_cfg, "get") else {}
    cutoff = cutoffs.get(crop) if hasattr(cutoffs, "get") and crop else None
    advice, trace = decide_advice(
        entry,
        status=status,
        confidence=confidence,
        ood_status="accepted" if status not in {"rejected", "rejected_not_a_leaf"} else "rejected",
        quality={"brightness": brightness, "blur_score": blur_value},
        bg_check=bg_check,
        confidence_cutoff=float(cutoff) if cutoff is not None else None,
        kb_version=_kb_version(),
        rules_fired=[reason] if reason not in {"", "none"} else [],
    )
    weather = {"status": "not_provided", "note": "risk unknown", "source_citation": None}
    weather_fn = getattr(engine, "weather_risk", None) if engine else None
    if callable(weather_fn):
        weather = weather_fn(crop, condition, temperature_c, humidity_pct)
    advice["weather_note"] = weather.get("note", "risk unknown")
    advice["weather_status"] = weather.get("status", "risk_unknown")
    advice["weather_source_citation"] = weather.get("source_citation")
    advice["status"] = (entry or {}).get("status", "PLACEHOLDER")
    advice["translation_status"] = (entry or {}).get("translation_status", "MISSING")
    advice["scouting_mr"] = (entry or {}).get("scouting_mr", [])
    advice["cultural_mr"] = (entry or {}).get("cultural_mr", [])
    if weather.get("rule_id"):
        trace["rules_fired"].append(str(weather["rule_id"]))
    ADVICE_TOTAL.labels(tier=trace["tier"], evidence_grade=grade).inc()
    if trace["tier"] != "full":
        withheld_reason = next(
            (rule for rule in trace["rules_fired"] if rule != reason),
            reason if reason not in {"", "none"} else trace["tier"],
        )
        ADVICE_WITHHELD_TOTAL.labels(reason=withheld_reason).inc()
    if crop and grade == "C":
        ADVICE_GAP_TOTAL.labels(**{"class": f"{crop}___{condition or 'unknown'}"}).inc()
    class_name = f"{crop}___{condition}" if crop and condition else ""
    hints = engine.get_confusable_hints(class_name) if engine and class_name else []
    return {"advisory": advice, "advice_trace": trace, "confusable_hints": hints}


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
