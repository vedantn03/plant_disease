# Project report outline

## Abstract

- Scope and system boundary (`README.md`, `docs/ARCHITECTURE.md`).
- Evaluation summary and limitations (`reports/snapshot/eval_summary.json`, `docs/STATUS.md`).

## Problem statement

- Plant disease image classification and decision-support context (`README.md`).
- Dataset-to-field generalization limits (`docs/DATASHEET.md`, `docs/STATUS.md`).

## Objectives

- Reproducible data and model lineage (`dvc.yaml`, `docs/ARCHITECTURE.md`).
- Governed candidate evaluation and serving (`scripts/promote.py`, `src/model_loader.py`).
- Observable inference, drift signals, and evidence-aware advice (`src/api.py`, `src/advisory_engine.py`).

## System architecture

- UI, API, model loading, registry and manifest (`docs/ARCHITECTURE.md`, `docker-compose.yml`).
- Metrics, dashboard, and drift exporter (`monitoring/`, `scripts/drift_exporter.py`).

## Data and versioning

- Dataset audit, split manifests, DVC stages, and remote configuration (`scripts/audit_and_split.py`, `dvc.yaml`, `.dvc/config`).
- Experiment tracking, registry export, and serving manifest (`reports/snapshot/registry_export.json`, `models/champion_manifest.json`).

## Model development and evaluation

- Training and saved evaluation artifacts (`scripts/train.py`, `reports/snapshot/eval_summary.json`).
- Headline, within-crop, calibration, and confusion results (`reports/snapshot/eval_summary.json`).
- Per-crop macro-F1 bug, correction, and regression test (`scripts/train.py`, `tests/test_wp1_close_gate.py`); identify old outputs as stale.
- Three-seed descriptive comparison and uncertainty (`reports/snapshot/seed_variance.json`).

## Promotion governance and rollback

- Threshold rationale and approval/hash mechanism (`configs/gates/gate_thresholds.yaml`, `configs/gates/approval.yaml`).
- Recorded dry-runs and absence of live alias movement (`reports/snapshot/promotion_log.jsonl`).
- Manifest, checkpoint hash, and rollback implementation (`src/promotion_safety.py`, `src/registry.py`).

## Serving, UI, and monitoring

- API response, health, and Prometheus metrics (`src/api.py`).
- Trust interface and evidence panels (`app/ui.py`, `monitoring/grafana/dashboards/system-health-model-trust.json`).
- Runtime evidence to be supplied from local screenshots (`docs/evidence/INDEX.md`).

## Drift monitoring

- Baseline and window method (`scripts/compute_baseline_features.py`, `scripts/drift_exporter.py`).
- Staged replay evidence and interpretation (`reports/snapshot/drift_latest.json`, `scripts/replay_traffic.py`).
- Clean-control and alert screenshot status (`docs/STATUS.md`, local evidence directory if captured).

## Background sensitivity

- Segmentation and fill protocol (`src/segmentation.py`, `src/background_check.py`).
- Per-fill seed measurements and caveat (`reports/snapshot/seed_variance.json`).
- VAL sample rates and added latency (`reports/snapshot/bg_check_rates.json`).

## Advisory provenance and graded advice

- Entry audit and evidence grades (`docs/ADVISORY_AUDIT.md`, `reports/snapshot/advisory_audit.json`).
- API advice trace and withheld advice behavior (`src/advisory_engine.py`, `src/api.py`).

## CI/CD and release

- Independent CI jobs and fixture-model scope (`.github/workflows/ci.yml`, `docs/CI_NOTES.md`).
- Manual GHCR workflow and trigger controls (`.github/workflows/publish-api.yml`).

## Limitations

- Lab-style dataset; no field validation (`docs/DATASHEET.md`, `docs/STATUS.md`).
- Three seeds are a small descriptive sample (`reports/snapshot/seed_variance.json`).
- Staged replay, Airflow simulation, local-only data/model/registry artifacts, and unverified dataset licensing (`docs/STATUS.md`, `docs/DECISIONS.md`).
- Advisory placeholders and absence of live promotion evidence (`reports/snapshot/advisory_audit.json`, `reports/snapshot/promotion_log.jsonl`).

## Future work

- Review and sign off source-backed advisory entries.
- Run clean-control and controlled degraded-traffic monitoring with captured evidence.
- Evaluate on appropriately consented and licensed field imagery.
- Assess multi-machine artifact storage and managed deployment only after local evidence is complete.

## References

- Noyan, M. A. (2022), *Uncovering bias in the PlantVillage dataset*, arXiv:2206.04374; existing project citation in `docs/AUDIT.md`.
- DVC user guide, data management and remote storage: `https://dvc.org/doc/user-guide/data-management/remote-storage` (also cited in `docs/DECISIONS.md`).
- DVC pipeline output documentation: `https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#outs` (cited in `docs/DECISIONS.md`).
- Prometheus installation: `https://prometheus.io/docs/prometheus/latest/installation/` (cited in `docs/DECISIONS.md`).
- Grafana Docker installation: `https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/` (cited in `docs/DECISIONS.md`).
- GitHub Actions token permissions: `https://docs.github.com/en/actions/tutorials/authenticate-with-github_token` (cited in `docs/CI_NOTES.md`).
- GitHub Packages permissions and billing: `https://docs.github.com/en/packages/learn-github-packages/about-permissions-for-github-packages` and `https://docs.github.com/en/billing/concepts/product-billing/github-packages` (cited in `docs/CI_NOTES.md`).
- detect-secrets documentation: `https://github.com/Yelp/detect-secrets/tree/v1.5.0` (cited in `docs/CI_NOTES.md`).
- MLflow and application-library references: cite the exact package versions recorded by the final environment’s `pip freeze` and their official documentation pages; the cloud workspace did not inspect the owner’s local environment.
