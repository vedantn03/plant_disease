# Project Progress

This records repository state, not that the local dataset, model, MLflow store, Docker stack, or
DVC remote was exercised in this checkout.

| Work area | Repository state | Evidence |
|---|---|---|
| Data audit and DVC pipeline | Implemented in code; local data state not checked here | `scripts/audit_and_split.py`, `dvc.yaml` |
| Candidate evaluation and promotion governance | Implemented with dry-run and approval checks | `scripts/promote.py`, `src/promotion_safety.py`, `configs/gates/` |
| API and trust responses | Implemented in repository | `src/api.py`, `src/background_check.py`, `src/advisory_engine.py` |
| Compose monitoring | Implemented in repository; runtime not checked here | `docker-compose.yml`, `monitoring/` |
| Request-log drift | Implemented for configured logs; replay is staged traffic | `scripts/drift_exporter.py`, `scripts/replay_traffic.py` |
| Airflow | Simulated | `scripts/run_airflow_simulation.py` |
| Model-card evidence snapshots | Pending export when source reports are unavailable in Git | `reports/snapshot/`, `scripts/snapshot_reports.py` |

## Verification

Tests and CI are the verification mechanisms. Check actual command output before reporting pass
counts. Dataset evaluation, inference, Docker startup, registry access, and DVC remote status are
not implied by repository tests.
