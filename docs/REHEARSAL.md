# Demo Rehearsal Checklist

Run these commands from the `feat/farmer-web-ui` linked-worktree root in Windows PowerShell. Leave
the timing and result columns blank until you run the steps and record your observations. The Docker Desktop image-export
I/O error reported during the prior rebuild is an environment failure; no successful rebuild is
claimed here. The supplied `no matching manifest for ...` line was marked as an example, not an
observed error, so it is not used as evidence or diagnosed here.

| Check | My time | Pass/fail and notes |
|---|---|---|
| Core config, build, and cold start | | |
| API health and browser dashboard availability | | |
| Prediction trust panel and explanation fallback | | |
| Prometheus and Grafana availability | | |
| Normal replay and drift state | | |
| Degraded replay and alert state | | |
| Candidate gate dry-runs | | |
| Teardown | | |

## Core app and explanation view

```powershell
$env:DATA_DIR = (Resolve-Path ..\data).Path
$env:MODEL_DIR = (Resolve-Path ..\models).Path
$env:REPORTS_DIR = (Resolve-Path ..\reports).Path
$env:MLFLOW_DB_PATH = (Resolve-Path ..\mlflow.db).Path
docker compose --env-file ..\.env --profile core config --quiet
docker compose --env-file ..\.env --profile core build api ui
docker compose --env-file ..\.env --profile core up -d
docker compose --env-file ..\.env --profile core ps
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/
```

Open `http://127.0.0.1:8501`, upload a local sample image, and inspect the farmer result. Grad-CAM
and background sensitivity appear only when returned by the API. Open the LIME and SHAP tabs to
request current-image asynchronous jobs; record their actual result or explicit failure state.
SHAP needs the configured TRAIN manifest and images. No unrelated saved figure is shown as an
explanation for the uploaded photo. The advice section shows status and cited source; the technical
trace is expandable. See `UI_TEST_CHECKLIST.md` for the full manual workflow and responsive widths.
Save screenshots only after checking that they contain no private image or local path you do not
want committed.

Open `http://127.0.0.1:8501/insights` to inspect the separate read-only evaluator route. Its process
counters, bounded log window, and saved snapshots are different evidence scopes. Replayed traffic
is staged traffic. If a metric panel says its report is not generated, leave it empty rather than
substituting a value from another split or run.

## Monitoring and staged replay

```powershell
docker compose --env-file ..\.env --profile core --profile monitoring config --quiet
docker compose --env-file ..\.env --profile core --profile monitoring up -d
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/targets
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000/api/health
..\..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1
Get-Content ..\reports\drift_latest.json
..\..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade blur
Get-Content ..\reports\drift_latest.json
..\..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade dark
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/api/v1/alerts
```

Replay requires the local TEST images and manifest. Confirm the request logs label replay traffic
with `traffic_source=replay`. Record the actual drift window and alert result; do not assume that a
clean replay will yield no drift or that degraded replay will meet the alert duration. These are
staged requests, not field traffic. Capture Grafana panels only after verifying the displayed
traffic source and saved evidence.

## Promotion evidence

The following are read-only dry-runs. They require the local candidate and champion metric reports.
They do not change the registry alias. Do not run `--apply` or rollback as part of this rehearsal.

```powershell
..\..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics ..\reports\candidate_gate_48ecf14d.json --champion-metrics ..\reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 3
..\..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics ..\reports\candidate_gate_0e2c51ab.json --champion-metrics ..\reports\candidate_gate_comparison.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 4
Get-Content ..\reports\promotion_log.jsonl
```

Show only saved gate verdicts and actual promotion-log entries. If no real apply or rollback entry
exists, say that promotion/rollback evidence was not captured. Do not substitute a simulation or
hand-written entry.

## Known environment failure and troubleshooting

An earlier image export failed with `failed to create temp dir ... input/output error`. The latest
Docker check in this worktree returned `permission denied` for the Docker Desktop Linux engine pipe;
no image build or container startup was run. Check the engine before rebuilding:

```powershell
docker info --format '{{.ServerVersion}}'
docker system df
docker compose --env-file ..\.env --profile core ps -a
docker compose --env-file ..\.env --profile core logs --tail 80 api ui
```

If `docker info` cannot connect, start Docker Desktop and wait for its Linux engine to become
ready. If the storage I/O error remains after a Docker Desktop restart and available disk-space
check, preserve the exact output and use Docker Desktop's diagnostics/support workflow. Do not run
prune, reset, or delete Docker data as a blind fix. If a genuine `no matching manifest` error occurs,
record the complete, untruncated message and the image/service line; the example text supplied for
this rehearsal was not enough to identify an image or platform mismatch.

## Teardown

```powershell
docker compose --env-file ..\.env --profile core --profile monitoring down
```

This stops the containers while retaining named-volume data. Do not add `-v` unless intentionally
discarding the local Prometheus and Grafana history.
