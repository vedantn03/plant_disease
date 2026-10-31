# Master Plan v2 Progress Tracker

| Task ID | Component / Tool | Agent Part Status | Manual Part Status | Evidence Path |
| :--- | :--- | :--- | :--- | :--- |
| **T0** | Repository Audit & Baseline Truth Table | DONE | DONE (Verified by User) | `docs/AUDIT.md` |
| **T1** | Config Centralization & Single Source of Truth | DONE | DONE | `configs/config.yaml`, `docs/AUDIT.md` |
| **T2** | Preflight, Runner, Hooks & Resource Budget | DONE | DONE (M2.1 & M2.2 Completed) | `scripts/preflight.ps1`, `tasks.ps1`, `docs/RESOURCE_BUDGET.md`, `docs/evidence/preflight/` |
| **T3** | DVC Data Pipeline & Versioning | DONE | DONE (Verified: 20 objects in D:\dvc_remote, test passed) | `dvc.yaml`, `dvc.lock`, `docs/evidence/dvc/` |
| **T4** | MinIO Object Store (S3-Compatible) | DEFERRED | DEFERRED | See DECISIONS.md 2026-10-31 |
| **T5** | Honest Evaluation Suite & Confused Pairs | DONE | DONE (Agent-only) | `reports/eval_summary.json` |
| **T6** | Shortcut-Robustness Suite (Novelty) | DONE | DONE (Agent-only) | `src/robustness.py`, `reports/robustness_report.md` |
| **T7** | Registry Lifecycle + Trust Gate + Rollback | TODO | TODO (M7.1) | `src/registry.py`, `reports/promotion_log.jsonl` |
| **T8** | Calibration, OOD, Quality & Abstention | TODO | TODO (M8.1) | `src/calibration.py`, `src/ood.py` |
| **T9** | Registry-Backed Serving (`models:/...@champion`) | TODO | N/A | `src/model_loader.py` |
| **T10** | Docker Images & Compose Profiles | TODO | TODO (M10.1) | `docker-compose.yml`, `docs/evidence/docker/` |
| **T11** | Metrics, Prometheus & Grafana Panels | TODO | TODO (M11.1) | `monitoring/`, `docs/evidence/grafana/` |
| **T12** | Live Drift Detection & Alert Rules | TODO | TODO (M12.1) | `scripts/monitor_drift.py` |
| **T13** | Feedback Loop, Privacy & Field Probe | TODO | TODO (M13.1) | `src/feedback.py`, `scripts/field_probe.py` |
| **T14** | Continuous Integration (GitHub Actions) | TODO | TODO (M14.1) | `.github/workflows/ci.yml` |
| **T15** | Continuous Deployment (GHCR Container Tag) | TODO | TODO (M15.1) | `.github/workflows/cd.yml` |
| **T16** | Airflow Orchestration in Container | TODO | TODO (M16.1) | `airflow/dags/`, `docs/evidence/airflow/` |
| **T17** | SHAP + LIME + Grad-CAM Comparison | TODO | N/A | `reports/explain/` |
| **T18** | Advisory Knowledge Base Hardening | TODO | N/A | `kb/advisories.yaml` |
| **T19** | Dynamic Model & Dataset Card Generator | TODO | N/A | `scripts/gen_model_card.py` |
| **T20** | Cloud Architecture Mapping (AWS SageMaker) | TODO | N/A | `docs/CLOUD_PATH.md` |
| **T21** | System Viva & Governance Pack | TODO | N/A | `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md` |
| **T22** | Live Demo Script & Checklist | TODO | N/A | `docs/DEMO.md` |
