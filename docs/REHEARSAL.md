# Demo Rehearsal Checklist

Run from the repository root in Windows PowerShell. Fill in the observed timings and outcomes;
they are intentionally blank here. Replayed TEST images are staged traffic and must remain labeled
as replay.

| Check | My time | Pass/fail and notes |
|---|---|---|
| Clean rebuild and cold start | | |
| Core API/UI ready | | |
| Trust panel and advice passport | | |
| Prometheus targets and Grafana | | |
| Normal replay and drift report | | |
| Degraded replay and alert | | |
| Gate dry-run for each candidate | | |
| DVC remote status and CI | | |
| Teardown | | |

## Rehearsal commands

### Clean rebuild

`down -v` removes persisted Prometheus and Grafana named-volume data. Run it only when clearing
that local monitoring history is intended.

```powershell
docker compose --profile core --profile monitoring down -v
docker compose --profile core --profile monitoring build
docker compose --profile core up -d
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
```

If the model is unavailable, inspect `docker compose --profile core logs api` and confirm the
local `models/` mount and registry configuration. Do not copy, overwrite, or train a checkpoint
as part of rehearsal.

### UI trust view

Open `http://127.0.0.1:8501`, select Diagnostic Screening, upload an available leaf image, and
show the returned trust fields, background sensitivity result, original/neutralized comparison,
and Advice passport. Missing API fields should remain “unknown” or “not available.” A second photo
may be used for an out-of-distribution example; describe only what the API returned.

### Monitoring and replay traffic

```powershell
docker compose --profile core --profile monitoring up -d
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/targets
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000/api/health
..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1
Get-Content reports\drift_latest.json
..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade blur
..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade dark
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/api/v1/alerts
```

Replay requires the TEST manifest and images on the local machine. Confirm that log entries mark
`traffic_source` as `replay`. The configured drift rule and alert duration determine whether an
alert fires; do not promise a result before observing it. Capture Grafana panels and alert state
under `docs/evidence/` only if the screenshots are safe to commit.

### Gate dry-runs and saved lifecycle evidence

These commands are read-only dry-runs; required report snapshots, metrics and registry evidence
must exist locally. They never move a registry alias.

```powershell
..\.venv\Scripts\python.exe scripts\inspect_registry.py --snapshot reports\registry_export.json
..\.venv\Scripts\python.exe scripts\snapshot_reports.py
..\.venv\Scripts\python.exe scripts\gen_model_card.py
..\.venv\Scripts\python.exe scripts\gen_model_card.py --check
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 3
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 4
Get-Content reports\promotion_log.jsonl
..\.venv\Scripts\dvc.exe status -c
```

Do not use `--apply` or rollback for the rehearsal unless separately authorized. If no successful
apply/rollback entry exists in the saved promotion log, report that lifecycle evidence was not
captured. Do not substitute a simulated or hand-written log.

### Teardown

```powershell
docker compose --profile core --profile monitoring down
```

Likely failures: Docker daemon stopped (start Docker Desktop's Linux engine); port occupied (stop
the process using the port); model missing or registry mismatch (verify local mounts and health
output); TEST manifest missing (restore local DVC data without running training); cold image build
may take longer than a warm start. Record actual observations in the table above.
