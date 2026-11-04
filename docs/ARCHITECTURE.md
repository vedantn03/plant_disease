# System Architecture

This diagram reflects the repository's Compose services and configured data paths. It describes
the local demo stack and does not imply a deployed production environment.

```mermaid
flowchart LR
    user[User image and browser] --> compose_ui[Compose ui]
    compose_ui -->|HTTP API_URL| compose_api[Compose api]
    compose_api -->|prediction response| compose_ui
    compose_api --> requestlog[(logs/requests.jsonl)]
    compose_api -->|GET /metrics| compose_prometheus[Compose prometheus]
    requestlog --> compose_drift-exporter[Compose drift-exporter]
    compose_drift-exporter -->|GET /metrics| compose_prometheus
    compose_prometheus -->|PromQL datasource| compose_grafana[Compose grafana]
    compose_api -->|read-only mounts| model[models/]
    compose_api -->|read-only mounts| kb[kb/]
    compose_prometheus --> rules[monitoring/rules/drift.yml]
    train[scripts/train.py] --> candidate[models/candidates/]
    candidate --> evaluate[VAL evaluation reports]
    evaluate --> dryrun[scripts/promote.py --dry-run]
    dryrun -. "approved apply only" .-> registry[(MLflow registry and manifest governance)]
    compose_api -. "read alias and verify manifest" .-> registry
```

Compose service nodes are `api`, `ui`, `prometheus`, `drift-exporter`, and `grafana`, from
`docker-compose.yml`. Prometheus scrapes the API and drift exporter using
`monitoring/prometheus.yml`; alert rules are loaded from `monitoring/rules/drift.yml`.

## Data flow

| Flow | Evidence | Behavior |
|---|---|---|
| Browser to API | `app/ui.py`, `docker-compose.yml` | Streamlit sends an uploaded image to the configured API URL. |
| Model load and health | `src/api.py`, `src/model_loader.py`, `models/champion_manifest.json` | API resolves serving identity and reports health; weights are read from the mounted model path. |
| Prediction audit log | `src/api.py`, `configs/config.yaml` | Request records go to the configured JSONL log; replay requests are labeled as replay. |
| Metrics to monitoring | `src/api.py`, `scripts/drift_exporter.py`, `monitoring/prometheus.yml` | Prometheus scrapes observed API and drift-exporter metrics. |
| Dashboard | `monitoring/grafana/` | Provisioned Grafana dashboards query the Prometheus datasource. |
| Candidate governance | `scripts/train.py`, `scripts/promote.py`, `src/promotion_safety.py` | Training writes candidate checkpoints; promotion has a dry-run and approval-gated apply path. |
| Drift trigger | `scripts/drift_exporter.py`, `monitoring/rules/drift.yml` | Real request-log windows can set the configured drift flag; replay traffic is staged. |

The Compose API mounts `models/`, `kb/`, and `reports/` read-only, `logs/` read/write, and
`mlflow.db` read-only. The UI mounts demo samples, reports, and logs read-only. Prometheus and
Grafana use named data volumes. Inspect `docker-compose.yml` for exact configuration.

## Deliberate boundaries

- `scripts/run_airflow_simulation.py` is a simulation; the repo does not establish a live Airflow
  scheduler or DAG execution.
- Drift checks can use real API logs, but replay is staged traffic and is not field traffic.
- `scripts/promote.py --dry-run` does not move the registry alias. Apply requires the configured
  governance checks; this documentation does not perform registry writes.
- The Streamlit interface has Diagnostic Screening, Production Telemetry, and MLOps System tabs,
  as defined in `app/ui.py`.
