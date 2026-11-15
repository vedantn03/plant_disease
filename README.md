# Plant Disease MLOps Pipeline

Academic prototype for leaf-image classification, evidence-aware advice, model governance, and
local monitoring. It is not validated for field diagnosis or treatment decisions. See
[`docs/STATUS.md`](docs/STATUS.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), and
[`docs/ADVISORY_AUDIT.md`](docs/ADVISORY_AUDIT.md).

## Local quickstart

Run PowerShell commands from the repository root. The local Python environment and model files
are user-managed; no dataset, checkpoint, `.env`, MLflow database, or DVC cache belongs in Git.

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
docker compose --profile core config --quiet
.\tasks.ps1 up-core
```

Set local Grafana credentials in `.env` before monitoring. Then:

```powershell
docker compose --profile core --profile monitoring config --quiet
.\tasks.ps1 up-monitoring
```

Open API health at `http://localhost:8000/health`, the browser dashboard at `http://localhost:8501`,
Prometheus targets at `http://localhost:9090/targets`, and Grafana at `http://localhost:3000`.
The API needs the mounted model checkpoint and registry configuration to serve predictions. If
Docker Desktop is stopped, start its Linux engine before running these commands.

To stop the stack without deleting persistent monitoring data:

```powershell
.\tasks.ps1 down
```

## Development checks

```powershell
..\.venv\Scripts\python.exe -m pytest tests/ -q -rs
..\.venv\Scripts\pre-commit.exe run --all-files
..\.venv\Scripts\python.exe scripts\gen_model_card.py --check
```

Some tests skip when local data or model artifacts are not present. CI exercises API plumbing
with a temporary fixture model; that is not an accuracy evaluation.

## Browser dashboard

The farmer workflow is a local static HTML/CSS/JavaScript page at `/`; operational and saved
evidence is on the separate `/insights` route. Start the API and UI together with the core Compose
profile above. For local development without Docker, run the API and `make ui` in separate
PowerShell windows from the repository root. The UI can load while the API is unavailable, but
prediction and explanation requests need a healthy API and a verified model checkpoint.

The explanation panel shows Grad-CAM when returned by `/predict`. LIME and SHAP are requested only
when their tabs are opened and run as bounded asynchronous jobs in the API. SHAP requires the
configured TRAIN manifest and its images. An unavailable dependency, missing TRAIN data, or worker
failure is shown as an explicit error; no saved figure is presented as an explanation of the
current upload. Attribution maps describe model behavior and do not establish a diagnosis.
`make explain-eval` runs a separate read-only VAL comparison and writes its report only when the
local checkpoint, split manifests, and required images are available.

`docs/UI_TEST_CHECKLIST.md` records the manual checks for upload, explanation tabs, language,
history, evaluator tabs, and narrow/desktop viewports. `docs/UI_REVIEW.md` preserves the initial
read-only audit and records which runtime checks remain unverified in this environment.
The evaluator Overview performs a read-only query for the latest MLflow run. It renders only
allowlisted training parameters and metrics; identifiers, hashes, filesystem paths, and other
parameter keys are omitted. When the configured store cannot be read, the panel reports that
instead of showing cached or placeholder values.

| Retained Streamlit capability | Browser location | Migration state |
|---|---|---|
| Single-photo upload, crop constraint and sample selection | Farmer route `/` | Migrated; API/model required for prediction. |
| Prediction trust and neutral-background comparison | Farmer result and Advanced checks | Migrated; values shown only when returned. |
| Grad-CAM | Farmer explanation panel | Migrated from prediction response. |
| LIME and SHAP | Lazy tabs on farmer route; Explainability tab in `/insights` | Async jobs implemented; real checkpoint behavior still needs local validation. |
| Advice passport and source trace | Farmer advice section | Migrated with status/source visible and technical trace collapsed. |
| Local scan history | Farmer route | Browser-local only, with clear action. |
| Production telemetry and MLOps evidence | `/insights` | Split into labeled live API values and saved report snapshots. |
| Batch prediction | Existing API endpoint | Not added to farmer UI; endpoint remains available. |

## Current implementation status

| Area | Status | Scope |
|---|---|---|
| Data and DVC workflow | Implemented | Pipeline code exists; local data/remote execution is machine-specific. |
| Training and evaluation | Partial | Code exists; no training is performed by the quickstart. |
| Promotion governance | Implemented | Dry-run and approval controls are present; alias changes require operator action. |
| API, browser UI, and local Compose | Implemented | The static HTML/CSS/JS UI is independent of the API process; predictions require a valid local serving model. |
| Drift monitoring | Partial | Real request logs are supported; replay is staged traffic and field drift is not established. |
| Airflow | Simulated | The script output is labelled SIMULATION. |
| Managed SageMaker | Planned | Local pipeline compilation only; no managed execution is claimed. |
| Advice provenance | Partial | Most KB entries are placeholders; see the source audit. |

## Known limitations

- The English/Marathi selector changes explanatory text, not navigation or technical labels.
  Available Marathi advisory text is marked as a machine draft; other advice falls back to its
  source language. A fluent agronomist review is still required before treating it as farmer-ready.
- Grad-CAM and the per-image background sensitivity result are shown when returned by the API.
  LIME and SHAP are generated on demand by the API; SHAP requires the configured TRAIN manifest.
  Real-model runs and their latency have not been verified in this development environment.

- Evaluation images are lab-style; field performance has not been validated.
- Background-fill results vary with the fill and segmentation. Black-fill drops include masking
  artifacts and are not proof of shortcut learning.
- Corn Cercospora / Gray leaf spot is a known weaker class; use saved evaluation artifacts for
  current numeric claims.
- The API serves unscaled softmax confidence at T=1. See `docs/DECISIONS.md` for the rationale.
- Advisory content is provenance-gated. Do not treat placeholders as agronomic guidance.
- Verify the dataset license and redistribution rights before public release.

## Documentation and evidence

- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Rehearsal commands: [`docs/REHEARSAL.md`](docs/REHEARSAL.md)
- Demo script: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)
- Viva Q&A: [`docs/VIVA_QA.md`](docs/VIVA_QA.md)
- Evidence inventory: [`docs/evidence/INDEX.md`](docs/evidence/INDEX.md)
- Model card and datasheet inputs: committed, sanitized files in `reports/snapshot/`. Refresh them
  with the snapshot commands below when local reports change; review generated files before commit.

To prepare local evidence snapshots, use the read-only sanitized registry export, collect the
available reports, then regenerate and check the card:

```powershell
..\.venv\Scripts\python.exe scripts\inspect_registry.py --snapshot reports\registry_export.json
..\.venv\Scripts\python.exe scripts\snapshot_reports.py
..\.venv\Scripts\python.exe scripts\gen_model_card.py
..\.venv\Scripts\python.exe scripts\gen_model_card.py --check
```

The snapshot command omits unavailable inputs from its copy list, redacts local paths/usernames,
and rejects files above the pre-commit configured limit. Review generated snapshots before
committing them; do not stage raw model files, databases, logs, or local data.
