# Demo Rehearsal Checklist

Run these commands from the repository root in Windows PowerShell. Leave the timing and result
columns blank until you run the steps and record your observations. The Docker Desktop image-export
I/O error reported during the prior rebuild is an environment failure; no successful rebuild is
claimed here. The supplied `no matching manifest for ...` line was marked as an example, not an
observed error, so it is not used as evidence or diagnosed here.

| Check | My time | Pass/fail and notes |
|---|---|---|
| Core config, build, and cold start | | |
| API health and Streamlit availability | | |
| Prediction trust panel and explanation fallback | | |
| Prometheus and Grafana availability | | |
| Normal replay and drift state | | |
| Degraded replay and alert state | | |
| Candidate gate dry-runs | | |
| Teardown | | |

## Core app and explanation view

```powershell
docker compose --profile core config
docker compose --profile core build ui
docker compose --profile core up -d
docker compose --profile core ps
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/_stcore/health
```

Open `http://127.0.0.1:8501`, upload a local sample image, and open Explainability. Grad-CAM is
provided by the API when present. The UI image does not install SHAP or LIME; it should show saved
offline comparison figures when available and label them as offline rather than as explanations
computed for the uploaded image. If those figures are absent, the UI should say “Not generated yet.”
Attribution images show model feature attribution; they do not prove a diagnosis or explain a
disease's cause. Save the screenshot under `docs/evidence/ui/` only after checking that it contains
no private image or local path you do not want committed.

## Monitoring and staged replay

```powershell
docker compose --profile core --profile monitoring config
docker compose --profile core --profile monitoring up -d
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/targets
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000/api/health
..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1
Get-Content reports\drift_latest.json
..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade blur
Get-Content reports\drift_latest.json
..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade dark
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
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 3
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 4
Get-Content reports\promotion_log.jsonl
```

Show only saved gate verdicts and actual promotion-log entries. If no real apply or rollback entry
exists, say that promotion/rollback evidence was not captured. Do not substitute a simulation or
hand-written entry.

## Known environment failure and troubleshooting

The reported image-export error was `failed to create temp dir ... input/output error`. This points
to Docker Desktop's local image storage/engine, not to an application-level Python exception; the
cloud sandbox cannot inspect or repair that Windows engine. Check the engine before rebuilding:

```powershell
docker info --format '{{.ServerVersion}}'
docker system df
docker compose --profile core ps -a
docker compose --profile core logs --tail 80 api ui
```

If `docker info` cannot connect, start Docker Desktop and wait for its Linux engine to become
ready. If the storage I/O error remains after a Docker Desktop restart and available disk-space
check, preserve the exact output and use Docker Desktop's diagnostics/support workflow. Do not run
prune, reset, or delete Docker data as a blind fix. If a genuine `no matching manifest` error occurs,
record the complete, untruncated message and the image/service line; the example text supplied for
this rehearsal was not enough to identify an image or platform mismatch.

## Teardown

```powershell
docker compose --profile core --profile monitoring down
```

This stops the containers while retaining named-volume data. Do not add `-v` unless intentionally
discarding the local Prometheus and Grafana history.
