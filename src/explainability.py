"""Lazy, bounded image-explanation jobs for the local serving API."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import preprocess_image


TERMINAL_STATES = frozenset({"done", "failed", "timeout"})


class ExplanationQueueFull(RuntimeError):
    """No configured explanation-worker capacity is available."""


@dataclass
class ExplanationJob:
    job_id: str
    cache_key: str
    method: str
    created_at: float
    status: str = "queued"
    result: dict[str, Any] | None = None
    reason_code: str | None = None
    message: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None


class ExplanationJobStore:
    """Small in-memory job store with bounded workers, cache, and cancellation."""

    def __init__(
        self,
        workers: int,
        cache_size: int,
        timeout_s: float,
        max_pending: int,
        max_records: int = 100,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self._max_pending = max(1, max_pending)
        self._max_records = max(1, max_records)
        self._cache_size = max(0, cache_size)
        self._timeout_s = max(0.1, timeout_s)
        self._lock = threading.RLock()
        self._jobs: dict[str, ExplanationJob] = {}
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @staticmethod
    def key_for(image_bytes: bytes, method: str, params: dict[str, Any], model_version: str) -> str:
        digest = hashlib.sha256(image_bytes).hexdigest()
        cache_data = json.dumps(
            {"image": digest, "method": method, "params": params, "model": model_version},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(cache_data).hexdigest()

    def submit(
        self,
        *,
        image_bytes: bytes,
        method: str,
        params: dict[str, Any],
        model_version: str,
        work: Callable[[threading.Event], dict[str, Any]],
    ) -> ExplanationJob:
        cache_key = self.key_for(image_bytes, method, params, model_version)
        job = ExplanationJob(uuid.uuid4().hex, cache_key, method, time.monotonic())
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                job.status = "done"
                job.result = dict(cached)
                job.result["cache_hit"] = True
                while len(self._jobs) >= self._max_records:
                    removable = next((
                        job_id for job_id, old_job in self._jobs.items()
                        if old_job.status in TERMINAL_STATES
                        and (old_job.future is None or old_job.future.done())
                    ), None)
                    if removable is None:
                        raise ExplanationQueueFull("Explanation history is full; retry shortly.")
                    del self._jobs[removable]
                self._jobs[job.job_id] = job
                return job
            while len(self._jobs) >= self._max_records:
                removable = next((
                    job_id for job_id, old_job in self._jobs.items()
                    if old_job.status in TERMINAL_STATES
                    and (old_job.future is None or old_job.future.done())
                ), None)
                if removable is None:
                    raise ExplanationQueueFull("Explanation history is full; retry shortly.")
                del self._jobs[removable]
            active = sum(
                item.future is not None and not item.future.done()
                for item in self._jobs.values()
            )
            if active >= self._max_pending:
                raise ExplanationQueueFull("The explanation queue is busy. Try again shortly.")
            self._jobs[job.job_id] = job
            job.future = self._executor.submit(self._run, job, work)
        return job

    def _run(
        self, job: ExplanationJob, work: Callable[[threading.Event], dict[str, Any]]
    ) -> None:
        with self._lock:
            if job.status in TERMINAL_STATES:
                return
            job.status = "running"
        try:
            result = work(job.cancel_event)
            with self._lock:
                if job.status in TERMINAL_STATES:
                    return
                if job.cancel_event.is_set():
                    job.status = "failed"
                    job.reason_code = "cancelled"
                    job.message = "Explanation cancelled."
                    return
                job.status = "done"
                job.result = result
                if self._cache_size:
                    self._cache[job.cache_key] = dict(result)
                    self._cache.move_to_end(job.cache_key)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        except Exception as exc:  # job errors are returned as data, not HTTP 500s
            with self._lock:
                if job.status in TERMINAL_STATES:
                    return
                job.status = "failed"
                job.reason_code = getattr(exc, "reason_code", "explanation_failed")
                job.message = str(exc) or "The explanation could not be generated."

    def get(self, job_id: str) -> ExplanationJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status not in TERMINAL_STATES:
                if time.monotonic() - job.created_at > self._timeout_s:
                    job.cancel_event.set()
                    if job.future:
                        job.future.cancel()
                    job.status = "timeout"
                    job.reason_code = "explanation_timeout"
                    job.message = "The explanation exceeded its configured time limit."
            return job

    def cancel(self, job_id: str) -> ExplanationJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status not in TERMINAL_STATES:
                job.cancel_event.set()
                if job.future:
                    job.future.cancel()
                job.status = "failed"
                job.reason_code = "cancelled"
                job.message = "Explanation cancelled."
            return job

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class ExplanationUnavailable(RuntimeError):
    """An expected missing dependency or data prerequisite for an explainer."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _pil_to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _overlay(
    image: Image.Image, heat: np.ndarray, region_fraction: float
) -> tuple[str, float, str]:
    heat = np.asarray(heat, dtype=np.float32)
    if heat.ndim != 2 or not np.isfinite(heat).all():
        raise ValueError("Attribution output was not a finite two-dimensional map.")
    heat = np.maximum(heat, 0)
    maximum = float(heat.max(initial=0.0))
    if maximum <= 0:
        raise ValueError("The explainer returned no positive attribution map.")
    if float(np.ptp(heat)) <= np.finfo(np.float32).eps:
        raise ValueError("The explainer returned no spatially distinguishable attribution map.")
    heat = heat / maximum
    heat_image = Image.fromarray(np.uint8(np.clip(heat * 255, 0, 255))).resize(
        image.size, Image.Resampling.BILINEAR
    )
    values = np.asarray(heat_image, dtype=np.float32) / 255.0
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(original)
    color[..., 0] = 255.0
    alpha = (values * 0.55)[..., None]
    result = np.clip(original * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    flat = values.reshape(-1)
    count = min(flat.size, max(1, int(np.ceil(flat.size * region_fraction))))
    selected = np.argpartition(flat, flat.size - count)[flat.size - count:]
    mask = np.zeros(flat.size, dtype=np.uint8)
    mask[selected] = 255
    mask_image = Image.fromarray(mask.reshape(values.shape))
    return _pil_to_png_base64(Image.fromarray(result)), float(count / flat.size), _pil_to_png_base64(mask_image)


def _prediction_fn(
    model,
    device: str,
    image_size: int,
    crop_class_ids: list[int] | None,
    *,
    prepared_rgb: bool = False,
):
    to_model_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def predict(images: np.ndarray) -> np.ndarray:
        batch = []
        for item in images:
            pil_image = Image.fromarray(np.asarray(item, dtype=np.uint8)).convert("RGB")
            tensor = (
                to_model_tensor(pil_image)
                if prepared_rgb else preprocess_image(pil_image, image_size).squeeze(0)
            )
            batch.append(tensor)
        tensor = torch.stack(batch).to(device)
        model.eval()
        with torch.inference_mode():
            logits = model(tensor)
            if crop_class_ids:
                allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
                allowed[crop_class_ids] = True
                logits = logits.masked_fill(~allowed.unsqueeze(0), -1e9)
            probabilities = torch.softmax(logits, dim=1)
        output = probabilities.detach().cpu().numpy().astype(np.float64)
        if output.ndim != 2 or not np.isfinite(output).all():
            raise ValueError("Model returned invalid class probabilities for explanation.")
        return output

    return predict


def _train_background(
    manifest_path: str | Path,
    *,
    data_root: str | Path,
    n_background: int,
    seed: int,
    image_size: int,
    device: str,
) -> torch.Tensor:
    """Load a deterministic subset from the TRAIN manifest only."""
    path = Path(manifest_path)
    if path.name.lower() != "train.csv" or not path.is_file():
        raise ExplanationUnavailable(
            "training_background_unavailable",
            "SHAP needs the configured train.csv manifest and its training images.",
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    root = Path(data_root)
    image_paths = []
    for row in rows:
        raw_path = row.get("path")
        if not raw_path:
            continue
        recorded = Path(raw_path)
        if recorded.is_file():
            image_paths.append(recorded)
            continue
        class_name = Path(row.get("class_name", "")).name
        filename = Path(row.get("filename", "")).name
        if class_name and filename:
            image_paths.append(root / class_name / filename)
    available = [item for item in image_paths if item.is_file()]
    if not available:
        raise ExplanationUnavailable(
            "training_background_unavailable",
            "No readable training images from the configured TRAIN manifest are available.",
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(available), size=min(n_background, len(available)), replace=False)
    tensors = []
    for index in selected:
        with Image.open(available[int(index)]) as source:
            tensors.append(preprocess_image(source.convert("RGB"), image_size).squeeze(0))
    return torch.stack(tensors).to(device)


def generate_explanation(
    *,
    method: str,
    image: Image.Image,
    model,
    device: str,
    idx_to_class: dict[int, str],
    model_version: str,
    params: dict[str, Any],
    crop_class_ids: list[int] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Generate a LIME or SHAP overlay without changing the model or image."""
    started = time.perf_counter()
    image = image.convert("RGB")
    image_size = int(params["image_size"])
    # Give LIME the same aspect-preserving resize and center crop used by /predict.
    explained_image = transforms.CenterCrop(image_size)(
        transforms.Resize(int(image_size * 256 / 224))(image)
    )
    pixels = np.asarray(explained_image, dtype=np.uint8)
    predict = _prediction_fn(
        model, device, image_size, crop_class_ids, prepared_rgb=(method == "lime")
    )
    probabilities = predict(pixels[None, ...])[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = idx_to_class.get(predicted_index, str(predicted_index))

    if cancel_event and cancel_event.is_set():
        raise ExplanationUnavailable("cancelled", "Explanation cancelled.")

    if method == "lime":
        try:
            from lime.lime_image import LimeImageExplainer
            from skimage.segmentation import quickshift
        except ImportError as exc:
            raise ExplanationUnavailable(
                "dependency_missing", "LIME requires the API image to include lime and scikit-image."
            ) from exc
        seed = int(params["seed"])
        explainer = LimeImageExplainer(random_state=seed)
        def segments(value):
            return quickshift(
            value, kernel_size=int(params["lime_kernel_size"]),
            max_dist=int(params["lime_max_dist"]), ratio=float(params["lime_ratio"]),
            )
        explanation = explainer.explain_instance(
            pixels,
            predict,
            labels=(predicted_index,),
            num_samples=int(params["lime_num_samples"]),
            num_features=int(params["lime_num_features"]),
            segmentation_fn=segments,
            hide_color=0,
        )
        if cancel_event and cancel_event.is_set():
            raise ExplanationUnavailable("cancelled", "Explanation cancelled.")
        segment_map = np.asarray(explanation.segments, dtype=np.int32)
        heat = np.zeros(segment_map.shape, dtype=np.float32)
        for segment_id, weight in explanation.local_exp[predicted_index]:
            heat[segment_map == segment_id] = abs(float(weight))
        parameters = {
            "num_samples": int(params["lime_num_samples"]),
            "num_features": int(params["lime_num_features"]),
            "seed": seed,
        }
    elif method == "shap":
        try:
            import shap
        except ImportError as exc:
            raise ExplanationUnavailable(
                "dependency_missing", "SHAP is not installed in the API environment."
            ) from exc
        background = _train_background(
            params["train_manifest"],
            data_root=params["data_root"],
            n_background=int(params["shap_n_background"]),
            seed=int(params["seed"]),
            image_size=int(params["image_size"]),
            device=device,
        )
        target = preprocess_image(image, int(params["image_size"])).to(device)
        explainer = shap.GradientExplainer(model, background)
        raw_values = explainer.shap_values(
            target, nsamples=int(params["shap_n_samples"]), ranked_outputs=1
        )
        values = raw_values[0] if isinstance(raw_values, tuple) else raw_values
        if isinstance(values, list):
            values = values[0]
        array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
        array = np.squeeze(array)
        if array.ndim == 4:
            # SHAP versions may return HWC+output or CHW+output layouts.
            array = array[..., 0] if array.shape[-1] == 1 else array[0]
        if array.ndim != 3:
            raise ValueError("SHAP returned an unsupported attribution shape.")
        channel_axis = 0 if array.shape[0] == 3 else -1 if array.shape[-1] == 3 else None
        if channel_axis is None:
            raise ValueError("SHAP output has no RGB channel dimension.")
        heat = np.mean(np.abs(array), axis=channel_axis)
        parameters = {
            "background_source": "configured TRAIN manifest",
            "n_background": int(background.shape[0]),
            "n_samples": int(params["shap_n_samples"]),
            "seed": int(params["seed"]),
        }
    else:
        raise ExplanationUnavailable("invalid_method", "Choose LIME or SHAP.")

    overlay, fraction, attribution_mask = _overlay(
        explained_image, heat, float(params["deletion_region_fraction"])
    )
    return {
        "status": "done",
        "method": method,
        "predicted_class": predicted_class,
        "confidence": float(probabilities[predicted_index]),
        "overlay_png_base64": overlay,
        "attribution_mask_png_base64": attribution_mask,
        "model_input_png_base64": _pil_to_png_base64(explained_image),
        "attribution_summary": {"highlighted_pixel_fraction": fraction},
        "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
        "parameters": parameters,
        "model_version": model_version,
        "caption": "Highlighted areas influenced this result. This shows what the model looked "
        "at, not proof of disease.",
        "cache_hit": False,
    }
