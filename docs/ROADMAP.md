# System Engineering Roadmap: End-to-End MLOps Lifecycle

This document is the merged single-source engineering plan for the Plant Disease Diagnostic & Precision Advisory Platform. Every phase lists the responsible execution model (**Agent-only** vs **Manual User Action**), technology dependencies, and verifiable artifacts.

---

## Phase 1: Foundations & Integrity (COMPLETED)
- **T0 — Repository Audit & Ground Truth Table** [Agent: DONE | User: DONE]
  - *Artifacts*: `docs/AUDIT.md`, `docs/STATUS.md`
- **T1 — Configuration Centralization** [Agent: DONE | User: DONE]
  - *Artifacts*: `configs/config.yaml`
- **T2 — Preflight Runner, Git Hooks & Hardware Profiling** [Agent: DONE | User: DONE]
  - *Artifacts*: `scripts/preflight.ps1`, `tasks.ps1`, `.pre-commit-config.yaml`, `docs/RESOURCE_BUDGET.md`
- **T3 — DVC Data Versioning & Local Storage Remote** [Agent: DONE | User: DONE]
  - *Artifacts*: `dvc.yaml`, `dvc.lock`, `.dvc/config`, `D:\dvc_remote` (20 objects, 146.3 MB)
- **T4 — S3/MinIO Object Store** [DEFERRED]
  - *Decision*: Local DVC remote is sufficient for offline local-first viva demonstration; MinIO deferred.
- **T5 — Honest Evaluation Suite & Confused Pairs** [Agent: DONE | User: N/A]
  - *Artifacts*: `reports/eval_summary.json`, `reports/per_crop_metrics.json`
- **T6 — Shortcut-Robustness & Capture Bias Suite** [Agent: DONE | User: N/A]
  - *Artifacts*: `src/robustness.py`, `scripts/robustness_eval.py`, `reports/robustness_metrics.json`, `reports/robustness_report.md`

---

## Phase 2: Lifecycle Core & Trust Governance (CURRENT)
- **T7 — Registry Lifecycle, 4-Criterion Trust Gate & Rollback**
  - *Agent*: Build `configs/gates/gate_thresholds.yaml`, `src/registry.py`, `scripts/promote.py`, `tasks.ps1 rollback`, unit tests for gate rejection/first-model/rollback.
  - *Manual (M7.1)*: Run `python scripts/promote.py` to register champion v2 with alias `@champion`; run real 1-epoch weak candidate training to prove rejection in terminal and MLflow UI.
  - *Artifacts*: `reports/promotion_log.jsonl`, SQLite `mlflow.db` (`@champion` alias).
- **T8 — Calibration, OOD, Quality & Abstention Engine**
  - *Agent*: Extend `src/ood_detector.py` and calibration into unified serving abstention engine (`src/abstention.py`): Layer 1 (blur/brightness quality heuristic), Layer 2 (Helmholtz Energy Score OOD), Layer 3 (selective prediction confidence threshold 0.50).
  - *Manual*: Validate test cases via terminal script.
  - *Artifacts*: `src/abstention.py`, `tests/test_abstention.py`.
- **T8b — Background-Consistency Verification**
  - *Agent*: Foreground segmentation consistency check; flag inputs where background features contradict leaf predictions.
  - *Artifacts*: `src/consistency.py`.
- **T9 — Registry-Backed Serving with FastAPI Lifespan Handlers**
  - *Agent*: Refactor `src/api.py` to eliminate `@app.on_event` deprecations; load model directly from MLflow registry alias `models:/PlantDiseaseClassifier@champion` with fallback.
  - *Manual*: Run `uvicorn` and verify `/health` and `/predict` endpoints.
  - *Artifacts*: `src/api.py`, `tests/test_api_registry.py`.

---

## Phase 3: Platform Telemetry & Observability
- **T10 — Multi-Stage Docker Images & Compose Profiles**
  - *Agent*: Multi-target `Dockerfile` (api, ui, monitoring), `docker-compose.yml` with health checks.
  - *Manual (M10.1)*: Execute `docker compose up --build -d` and confirm container status.
  - *Artifacts*: `Dockerfile`, `docker-compose.yml`, `docs/evidence/docker/`.
- **T11 — Production Metrics, Prometheus & Grafana Dashboards**
  - *Agent*: Instrument FastAPI with `prometheus-fastapi-instrumentator`; configure `prometheus.yml` and pre-provisioned Grafana dashboard JSONs (request latency p95, abstention rate, error rate).
  - *Manual (M11.1)*: View Grafana at `http://localhost:3000`.
  - *Artifacts*: `monitoring/prometheus.yml`, `monitoring/grafana/`.
- **T12 — Production Drift Monitoring on Real Request Logs**
  - *Agent*: `scripts/monitor_drift.py` analyzing `logs/requests.jsonl` against `data/splits/train.csv` baseline using two-sample KS test and PSI on brightness, blur, and prediction entropy. Trigger alert if drift breaches threshold.
  - *Manual (M12.1)*: Run drift benchmark and view trigger output.
  - *Artifacts*: `scripts/monitor_drift.py`, `reports/drift_report.json`.

---

## Phase 4: Agronomic Advisory Control Tower (AC1 to AC11)
- **AC1 to AC11 — Streamlit Interactive Governance Control Center**
  - *Agent*: Wire UI Tab 1 to 5 to live endpoints: Tab 1 (Diagnostic Screening with Abstention & XAI), Tab 2 (Service Logs & Live Telemetry), Tab 3 (Empirical Bias & Robustness Dashboard), Tab 4 (Structural Gaps & Single-Crop Analysis), Tab 5 (MLOps Governance, Gate Logs & Model Card Generator).
  - *Manual*: Open Streamlit dashboard at `http://localhost:8501`.

---

## Phase 5: Challenger Models & Chaos Testing
- **Challenger Model — Shortcut-Resistant Candidate (E2 Mixture)**
  - *Agent*: Candidate training configuration with 40% randomized background compositing.
  - *Manual*: Execute training run on RTX 3050 (~11 minutes for 8 epochs); run `scripts/promote.py` to evaluate whether challenger displaces champion v2.
  - *Artifacts*: `models/candidates/challenger_e2.pth`.
- **Chaos Testing & Shadow Deployment**
  - *Agent*: `scripts/replay_traffic.py --degrade` introducing blur, darkening, and OOD non-leaf inputs to test real-time alerting.

---

## Phase 6: Automation & Deployment Pipelines
- **T13 — Feedback Loop & Field Probing**
  - *Agent*: `/feedback` endpoint in FastAPI accumulating human-corrected labels into `data/feedback/`; privacy-preserving EXIF metadata stripper.
- **T14 — Continuous Integration (GitHub Actions)**
  - *Agent*: `.github/workflows/ci.yml` running linting, pre-commit, and test suite on CPU.
- **T15 — Continuous Deployment (GHCR Container Tagging)**
  - *Agent*: `.github/workflows/cd.yml` building and publishing multi-arch image to GitHub Container Registry.
- **T16 — Apache Airflow Orchestration in Container**
  - *Agent*: Containerized Airflow DAG for scheduled dataset validation, drift detection, and retraining simulation.

---

## Phase 7: Viva Governance Pack & Live Demonstration
- **T20 — AWS SageMaker Cloud Architecture Mapping** (`docs/CLOUD_PATH.md`)
- **T21 — System Viva & Governance Pack** (`docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`)
- **T22 — Live Demo Script & Faculty Defense Checklist** (`docs/DEMO.md`)
