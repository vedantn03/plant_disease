# LEAFWISE frontend review

The first sections below are the pre-implementation Phase 0 review and intentionally preserve the original findings. The final section records implementation and verification status. The review records the
current checkout and available artifacts, not a claim that local inference or the browser has
been exercised. No API, model, registry, DVC, report, or dataset state was changed to produce this
review.

## Blockers and verification boundary

- The feature worktree has `models/champion_manifest.json` but does not have
  `models/best_model.pth`. Model-backed prediction, Grad-CAM, SHAP, and LIME cannot be validated
  from this worktree without the local checkpoint.
- At Phase 0 the API declared no `/explain/jobs` endpoint. SHAP and LIME job handling, cancellation,
  caching, and timeout behavior are therefore **MISSING**, not merely hidden by the frontend.
- Docker Desktop's Linux engine pipe was unavailable in the last check. Container startup, live
  endpoint calls, and browser rendering have not been verified in this phase.
- Do not call any advisory a treatment recommendation unless the API returns it from a cited
  knowledge-base entry. A `PLACEHOLDER` entry is not verified advice.

## Frontend inventory

The frontend bundle in this checkout consists of `web/index.html`, `web/style.css`,
`web/app.js`, and `web/demo_samples.json`. `app/ui.py` and `app/ui_helpers.py` are the retained
Streamlit implementation and helpers. No frontend framework or remote asset loader is used by
the current HTML page.

| Current page element | Current data source | Decision | Finding |
|---|---|---|---|
| Header/navigation and hero | Static HTML | REWRITE | Hero/introduction occupy space before the tool; move to slim header and start at photo input. |
| Language selector | Static HTML + `app.js` dictionaries | REWRITE | Current Marathi/Hindi strings include headings and advice UI chrome; requested scope is explanatory content only. |
| Photo upload/preview/sample picker | HTML, bundled sample manifest, `/predict` | KEEP / REWRITE | Keep core upload; add camera capture, replace action, single filename display and clear errors. |
| Crop constraint | Static HTML + `/predict` form | REWRITE | Crop list is incomplete and not sourced from the API class list. |
| Temperature/humidity | Not present in HTML form; `/predict` accepts `temperature_c` and `humidity_pct` | REWRITE | Current farmer cannot supply the values the API accepts. |
| Prediction result | `/predict` | REWRITE | Current rendering exposes raw labels and developer trust terms; add a display/debug boundary and human-readable labels. |
| Grad-CAM | `gradcam_png_base64` in `/predict` | KEEP / REWRITE | Current API response supports the immediate method; retain limits caption and graceful empty/error state. |
| SHAP/LIME | None in current API; message hardcoded in `app.js` | DELETE / REBUILD | Do not show placeholder figures. Implement asynchronous explanation jobs before claiming either method works. |
| Background sensitivity | `/predict` additive trust fields | MOVE-TO-ADVANCED | Show a plain-language result under a collapsed advanced section. |
| Advisory passport | `/predict` `advisory` and `advice_trace` | REWRITE | Separate cited guidance from metadata; make verified/placeholder status prominent and keep technical trace expandable. |
| Weather note | `/predict` accepts optional weather fields | REWRITE | No input currently exists. Only show a rule-based note when both inputs and a matching rule are returned. |
| Request telemetry | `/ui-data/telemetry` | MOVE-TO-INSIGHTS | Developer data should not sit in the farmer diagnosis flow. |
| Model/ops evidence | `/health`, `/stats`, `/metrics`, `/ui-data/*` reports and DVC stage list | MOVE-TO-INSIGHTS | Current Insights content is sparse; report absence honestly and avoid “loading” indefinitely. |
| Limits/principles, footer | Static HTML | REWRITE | Keep the persistent decision-support disclaimer on result views; remove repetitive marketing blocks. |

## Backend route and schema inventory

The route declarations in `src/api.py` are:

| Route | Current request / response facts from code | Runtime result |
|---|---|---|
| `GET /health` | JSON serving/process status and model/registry fields | NOT RUN; API unavailable in this environment. |
| `GET /stats` | JSON process request count, mean latency/confidence and class distribution | NOT RUN. |
| `GET /metrics` | Prometheus text exposition, or 503 when client is unavailable | NOT RUN. |
| `POST /predict` | multipart `file`; optional `crop`, `temperature_c`, `humidity_pct`, `traffic_source`; returns prediction, trust, advice, and latency fields in code | NOT RUN with an actual checkpoint. |
| `POST /predict_batch` | multipart `files`; optional crop; response is a batch result list in code | NOT RUN. |
| `POST /admin/reload` | local/token-guarded reload request; state-changing and not probed | NOT RUN by design in read-only review. |
| `GET /ui-data/telemetry` | bounded read of request-log window | NOT RUN. |
| `GET /ui-data/pipeline` | read-only DVC stage names | NOT RUN. |
| `GET /ui-data/{report_name}` | explicit allow-list of saved JSON/JSONL reports | NOT RUN. |

At Phase 0 there were no `/explain/jobs`, `/insights/*`, drift-simulation, or structured model-display routes
in this API checkout. The exact runtime schemas remain unverified until the server is reachable;
the descriptions above are based on endpoint signatures and response construction in source, not
observed HTTP responses.

## Streamlit parity at Phase 0

At Phase 0 the retained Streamlit app was `app/ui.py`; it has not been deleted from the Git history
available to this checkout. The parity map is based on the present Streamlit source and README
demo flow.

| Streamlit capability | New location | Status |
|---|---|---|
| Single-image prediction, crop constraint, sample upload | `/` photo workflow | PARTIAL; display layer and weather inputs are incomplete. |
| Trust panel and mean-fill background comparison | `/` advanced checks | PARTIAL; current frontend exposes too much technical terminology. |
| Grad-CAM | `/` explanation panel | PARTIAL; depends on API response and local checkpoint. |
| Per-image SHAP and LIME | `/` lazy explanation tabs | MISSING; no API job routes exist. |
| Advice and evidence trace | `/` advisory panel | PARTIAL; provenance must be farmer-readable with technical fields disclosed separately. |
| Production telemetry | `/insights` | PARTIAL; endpoint exists, current page mixes it with farmer content. |
| MLOps evidence, promotion/gate snapshots, drift | `/insights` | PARTIAL; saved-report paths exist, empty/loading behavior needs redesign. |
| Batch inference | No farmer-facing replacement required by the defined photo flow | MISSING from frontend; API endpoint remains available. |
| Explanation language selection | `/` explanation-only toggle | PARTIAL; current toggle translates interface labels and does not translate KB advice. |

## Metric reconciliation audit

- At Phase 0 the HTML/JS did not render numeric validation metrics. Remove any static model score from
  farmer pages; model metrics belong in `/insights` and must carry split, metric definition,
  model/version and source artifact.
- `scripts/evaluate.py` and the saved evaluation reports are the source for split metrics; the
  model card also contains a historical test metric. These are different reporting contexts and
  must not be shown as competing unlabeled headline numbers.
- Within-crop macro-F1 must be calculated using only the true crop's class labels, as recorded in
  `docs/DECISIONS.md` and the saved gate/evaluation reports. Black-fill background drop is
  segmentation/fill sensitive and must not be described as proof of shortcut learning.
- A per-metric UI table must be wired to the committed `reports/snapshot/` artifacts or read-only
  API responses. No metric should be copied into markup or invented to fill a gap.

## Phase 0 checks and output

- `pytest tests/ -q -rs`: **144 passed, 19 skipped, 1 warning** in this worktree. Skips include
  model-dependent tests, Windows socketpair limitations, unavailable split manifests, and an
  unavailable pre-commit executable.
- `node --check web/app.js`: passed.
- `docker compose --profile core config --quiet` and
  `docker compose --profile core --profile monitoring config --quiet`: passed.
- `git diff --check`: passed.
- `docker info --format '{{.ServerVersion}}'`: failed to connect to the Docker Desktop Linux
  engine named pipe. Thus no container, real inference, or browser test was run.
- Read-only `Invoke-WebRequest` probes for `/health`, `/stats`, `/metrics`, telemetry, pipeline,
  and all reports exposed by the current UI each failed with Windows' socket access-permission
  error for `localhost:8000`. These are NOT recorded as HTTP 404/503 responses; no API response
  schema was observed over HTTP.
- `Get-ChildItem demo_samples -File`: lists bundled samples for healthy, diseased, and non-leaf
  cases. This checkout has no checkpoint (`Test-Path models\best_model.pth` returned false), and
  there is no dedicated blurry validation photo in the listed demo set. No sample was inferred or
  labeled as blurry without evidence.

## Phase 0 recommendation (historical)

Implement in small increments: split the farmer and evaluator routes; add a response presentation
layer and read-only Insights aggregation; implement bounded asynchronous explanation jobs only if
the installed SHAP/LIME packages and preprocessing can be verified; add translation fields as
explicit machine drafts; then add contract tests. Reconcile model-card numbers against snapshot
artifacts before changing its claims. Keep local checkpoint-dependent acceptance checks open until
the API can load the existing user-managed checkpoint.


## Implementation status after Phase 0

The frontend now has two routes: `/` for farmer screening and `/insights` for evaluator/MLOps
content. The normal result reads the additive `display` response, while raw class, unscaled
confidence, quality values, OOD, rules and evidence grade stay under the technical details view.
Advice status and citations are shown; unavailable evidence remains explicit. The explanation
selector is English/Marathi only; Marathi remains a machine draft where available and missing
fields fall back to source-language text.

Backend routes now include `GET /classes`, `GET /ui-config`, `GET /insights`, and asynchronous
`POST /explain/jobs`, `GET /explain/jobs/{job_id}`, `DELETE /explain/jobs/{job_id}`. The new `/predict`
response fields `display` and `debug` are additive. The explanation route produces LIME/SHAP jobs;
SHAP depends on the configured TRAIN manifest and images. Test coverage uses temporary fixtures
and synthetic model/image data; it does not confirm real-model accuracy or XAI faithfulness.

## Remaining verification

- The environment does not provide the project checkpoint in this feature worktree; actual
  healthy/diseased photos and real model output for Grad-CAM, LIME and SHAP are NOT RUN here.
- Docker Desktop's local engine was unavailable in the previously recorded Phase 0 check. Docker
  build/start, Prometheus/Grafana runtime and browser-driven end-to-end checks remain NOT RUN unless
  the final report below records new output.
- Viewport checks at 360 px, 768 px and 1440 px require a real browser; no Playwright/browser run is
  claimed by this source-level review.
- The offline `scripts/explain_eval.py` is not run unless the local checkpoint and TRAIN/VAL
  manifests exist. Its report is generated only by that explicit user command.
