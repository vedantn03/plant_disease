# Manual UI test checklist

Run commands from the `feat/farmer-web-ui` linked-worktree root in Windows PowerShell. The user's
local `.env`, model, reports, database and data remain in the parent checkout; the path overrides
below point Compose there without copying them into the feature branch. Start Docker Desktop's
Linux engine before Compose.

## Start and verify the local app

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests\ -q -rs
$env:DATA_DIR = (Resolve-Path ..\data).Path
$env:MODEL_DIR = (Resolve-Path ..\models).Path
$env:REPORTS_DIR = (Resolve-Path ..\reports).Path
$env:MLFLOW_DB_PATH = (Resolve-Path ..\mlflow.db).Path
docker info --format '{{.ServerVersion}}'
docker compose --env-file ..\.env --profile core config --quiet
docker compose --env-file ..\.env --profile core build api ui
docker compose --env-file ..\.env --profile core up -d
docker compose --env-file ..\.env --profile core ps
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/
```

Expected: Compose config exits silently; `ps` shows API healthy and UI running/healthy after
startup. If `Resolve-Path` reports a missing parent artifact, stop and confirm the parent checkout
has that local file/folder. If Docker reports an engine pipe or image-storage I/O error, treat it as
an engine failure and use the non-destructive checks in `docs/REHEARSAL.md`.

Open `http://127.0.0.1:8501/`. If `/health` says the model is unavailable, check the model mount,
manifest and registry resolution before testing predictions. Do not train as a troubleshooting
step.

## Farmer workflow

- At a narrow phone width, select a photo using camera/gallery; on desktop test drag-and-drop and
  replace-photo. Confirm filename is not repeated and image preview stays within the card.
- Run one existing healthy sample, one diseased sample, one non-leaf image, and one locally chosen
  blurry photo. Record only what the API returns; do not present sample images as field validation.
- Confirm rejected/uncertain/low-trust results ask for a retake and do not present a disease result
  as confirmed. Confirm coverage gaps display as their own warning.
- Open Grad-CAM. Open LIME and SHAP separately; each must start an async job and display its own
  current-image result, or an explicit reason such as missing TRAIN data/dependency/timeout. There
  must be no unrelated saved image or fabricated fallback.
- Switch English and Marathi. Confirm headings/buttons remain English, the Marathi machine-draft
  notice is present where a draft is shown, and missing Marathi advice falls back with a note.
- Check advice status, source URL, weather rule/citation behavior, and background-sensitivity details.
- Confirm the decision-support disclaimer remains visible. Add multiple scans, reload the browser,
  then clear local history.

## Evaluator route and responsive layout

- Open `http://127.0.0.1:8501/insights`; visit Overview, Data, Model, Release gate, Monitoring,
  Explainability, Pipeline, and Responsible AI. Confirm every empty state says what evidence is
  missing and no unobserved job/monitor is labeled live.
- In Overview, verify that the latest MLflow run is either shown with only allowlisted params and
  metrics or reports the actual unavailable/not-configured state. Confirm no run ID, hash or local
  filesystem path is exposed.
- Compare process counters with the separately labeled request-log window. Check whether replay
  is marked as staged traffic.
- If the monitoring profile is started separately, visit Prometheus targets and Grafana; these
  links alone are not proof the services are healthy.
- Inspect farmer and evaluator routes at 360 px, 768 px, and 1440 px. Confirm there is no horizontal
  page overflow, clipped Devanagari, hidden controls, JavaScript console error, or image layout
  break.
- Capture screenshots only after successful local checks. Do not commit screenshots with private
  farmer photos or local file paths.

## Stop

```powershell
docker compose --profile core down
```

If Docker reports an engine/npipe or image-storage error, capture the complete output and resolve
Docker Desktop before repeating. Do not run prune/reset or delete Docker storage as a blind fix.
