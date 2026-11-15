# LEAFWISE migration audit

This file records the frontend migration against repository source. Runtime-dependent items are
explicitly marked NOT RUN unless a command below reports an observed result.

## Routes and feature disposition

| Route / feature | Source and API | Disposition | Evidence / limit |
|---|---|---|---|
| Farmer landing, upload, camera capture, drag-and-drop, preview and replace | `web/index.html`, `web/app.js`, `GET /classes`, `POST /predict` | REBUILT | Uses API class options; actual browser interaction NOT RUN here. |
| Crop constraint and optional temperature/humidity | `web/app.js`, `POST /predict` form fields | MIGRATED | No automatic weather lookup. |
| Farmer result, confidence band, coverage warning, retake guidance | Additive `display` response field from `src/presentation.py` | MIGRATED | Confidence bands are provisional config values; model values are API returned. |
| OOD, image quality, rule trace, raw labels and background comparison | `debug`, `advice_trace`, `bg_check_details` | MOVE-TO-INSIGHTS / ADVANCED | Farmer route keeps these in technical details or Advanced checks. |
| Grad-CAM | `POST /predict` `gradcam_png_base64` | MIGRATED | Real model output only; failure is a text state. |
| LIME and SHAP | `POST /explain/jobs`, `GET /explain/jobs/{id}`, `DELETE /explain/jobs/{id}` | IMPLEMENTED, RUNTIME NOT VERIFIED | Worker jobs are lazy, bounded, cancellable, timeout-aware, and cache by image/method/config/model identity. SHAP requires the configured TRAIN manifest and images. |
| Advice, source, status, pathogen group, weather citation, confusable hints | `advisory`, `advice_trace`, `confusable_hints` | MIGRATED | Placeholder entries are labeled; chemicals are displayed only for a full advice tier. Untranslated advice uses source-language text with a notice. |
| English/Marathi explanation toggle | `web/explanation_strings.json`, `scouting_mr` where available | MIGRATED WITH DRAFT LABELS | Only explanation/advice text changes. Marathi is machine draft; no reviewed translation is claimed. |
| Scan history | Browser `localStorage` | MIGRATED | Stored in this browser only; clear action present. History is not API telemetry. |
| Evaluator/operations view | `/insights`, saved snapshots, logs, `dvc.yaml`, `/metrics` | SPLIT TO `/insights` | Process counters and bounded log-window data are labelled separately; missing evidence has an explicit empty state. |
| Saved TEST evaluation, per-crop/class metrics and confusions | `reports/snapshot/eval_summary.json` | MIGRATED | TEST snapshot stays separate from VAL gate reports; no static metric is inserted into the UI. |
| Seed/gate/promotion evidence | `reports/snapshot/seed_variance.json`, candidate gate snapshot, promotion log | MIGRATED | Promotion log is deduplicated by candidate/version, latest first. |
| Dataset deduplication, cross-split overlap and capture-source audit | No sanitized dataset-audit snapshot is present | NOT AVAILABLE | Insights reports this evidence as missing; it does not reuse historical prose or infer audit results from model metrics. |
| Drift and fill sensitivity | drift and background-sensitivity snapshots | MIGRATED | Replay is labelled staged/simulated; no field drift claim. Fill artifact caveat remains. |
| DVC/component status and limitations | `dvc.yaml`, workflow files, model card | MIGRATED | Describes repository configuration, not live CI, DVC, Prometheus or Grafana health. |
| Drift simulation action | No mutating simulation endpoint | HIDDEN | Config keeps browser simulation disabled; no API arbitrary command is exposed. |
| Batch prediction UI | Existing `POST /predict_batch` | NOT MIGRATED | Existing API capability is retained; no multi-image farmer workflow was added. |
| MLflow run detail | Insights Overview and local MLflow UI link | IMPLEMENTED, RUNTIME NOT VERIFIED | The API performs a read-only latest-run lookup from the configured tracking URI and returns allowlisted parameters/metrics only; local paths, run IDs, hashes and other parameter keys are omitted. |

## Explainability implementation boundary

LIME uses resized image pixels, an ImageNet-normalized prediction function, and a fixed configured
seed. SHAP uses `GradientExplainer` and only samples the configured TRAIN manifest for background
images. Both return overlays aligned with the model input crop, the predicted class/confidence,
parameters, runtime, and an attribution-region mask. Jobs execute outside the async request loop;
the API still serializes model inference with the shared model lock. The timeout is enforced when a
job is polled. A timed-out running worker may finish later but its output is discarded. Optional
library errors, absent model, missing TRAIN files, invalid image and queue saturation are explicit
errors. Attribution means model association, not proof of cause, field truth, or diagnosis.

`tests/test_ui_rebuild.py` covers job lifecycle, cache reuse, cancellation, finite overlay shape,
probability output, and missing artifacts using synthetic fixtures. It does not prove that installed
SHAP/LIME versions work with the local checkpoint or establish explanation faithfulness on real
photos.

## Language and advice boundary

English is the interface language. The selector changes explanatory and advisory text only. A
Marathi string is used only when the API returns a `MACHINE_DRAFT` translation; otherwise the
English/source-language field is shown with a fallback notice. Chemical options are displayed only
when the returned advice tier is `full`; source status, citation and label reminder remain visible.
The backend knowledge base currently has one Marathi machine-draft scouting field; there is no
claim of human review. Do not infer translation availability from the selected language alone.

## Metric and telemetry boundary

The evaluator route labels the saved evaluation as TEST, seed/gate evidence as VAL when the source
file says VAL, and reads figures only from `reports/snapshot/`. It does not compare these as one
metric series. API process counters reset when the process restarts; request-log metrics cover only
the bounded returned window. Replay rows are separated by `traffic_source` and are not field data.

## Verification performed during this implementation

- Initial Phase 0 read-only review: see `docs/UI_REVIEW.md`; it predates these changes and records
  the original UI/API state. Its checks were not HTTP observations unless explicitly stated.
- Focused Python tests, JavaScript syntax checks, full suite, lint, and static Compose checks are
  reported in the final work report with exact output. No browser automation is claimed unless
  explicitly listed there.
- Real model-backed prediction, SHAP/LIME on leaf samples, explanation deletion evaluation, Docker
  build, Docker Compose startup, Prometheus/Grafana runtime and viewport rendering remain NOT RUN
  unless reported otherwise in the final work report.

## Current frontend file and section inventory

| File | Rendered section / responsibility | Data source |
|---|---|---|
| `web/index.html` | Header, language selector, evaluator link, photo workflow, result placeholders, local scan history, persistent disclaimer, footer | Static markup; `web/app.js` fills data. |
| `web/app.js` | Service/model state | `GET /health`, `GET /ui-config`, `GET /classes`. |
| `web/app.js` | Photo preview, sample list, crop/temperature/humidity and submit | `web/demo_samples.json`; `POST /predict` form fields. |
| `web/app.js` | Result, status, confidence, coverage and next steps | `display`, `confidence`, `top_k`, `coverage_warning` from `POST /predict`. |
| `web/app.js` | Grad-CAM and LIME/SHAP tabs | Grad-CAM from `/predict`; async create/poll/cancel at `/explain/jobs`. |
| `web/app.js` | Advice, source, weather citation, confusable hints and technical trace | `advisory`, `advice_trace`, `confusable_hints` from `/predict`. |
| `web/app.js` | Scan history | Browser `localStorage`; no API endpoint. |
| `web/insights.html` | Overview and Data panels | `GET /insights` (`/health`, process stats and bounded log window are assembled server-side). |
| `web/insights.html` | Model evidence, gate/release and promotion history | `/insights` reads only saved `reports/snapshot/` artifacts and promotion log. |
| `web/insights.html` | Monitoring | `/insights` plus direct `GET /metrics`; Prometheus/Grafana links are external local services and are not asserted healthy. |
| `web/insights.html` | Explainability lab | Async `/explain/jobs` API; the report panel reads configured `reports/explain_eval.md` when present. |
| `web/insights.html` | Pipeline, components and responsible-AI evidence | `/insights` reads `dvc.yaml`, workflow declarations, coverage, model card and snapshots; no execution is triggered. |
| `web/style.css` | Shared layout/design tokens, responsive farmer and evaluator styling, reduced-motion and Devanagari system-font fallback | Local CSS only. |
| `web/explanation_strings.json` | English/Marathi explanatory strings and fallback copy | Local JSON; Marathi is unreviewed draft text. |
| `web/demo_samples.json` | Sample labels/filenames for selection | Local manifest; labels are human-readable, source paths are omitted. |
| `web/server.py` | Static `/` and `/insights` routing plus runtime API-origin configuration | Local server; `LEAFWISE_API_URL` env only. |

The evaluator panels are Overview, Data, Model evidence, Release gate, Monitoring, Explainability,
Pipeline and Responsible AI. Each reads `/insights`; Monitoring also requests `/metrics`. Saved
reports are read under the API process, not fetched by the browser from arbitrary paths.
