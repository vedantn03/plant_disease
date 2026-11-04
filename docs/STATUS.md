# Current Status

| Area | Status | Evidence / limitation |
|---|---|---|
| Data/versioning pipeline | Implemented | `dvc.yaml`; execution requires local dataset and DVC state. |
| Train/evaluate | Partial | Code and stages exist; this documentation task did not train or evaluate a model. |
| Promotion gate | Implemented | `scripts/promote.py`; use dry-run for rehearsal. Approval state is separately governed. |
| API serving and UI | Implemented | `src/api.py`, `app/ui.py`; runtime requires local model artifacts and configured secrets. |
| Prometheus/Grafana monitoring | Implemented | Compose and provisioning files are tracked; stack runtime was not verified here. |
| Drift monitoring | Partial | Exporter reads request logs. Replay is staged; no field deployment is claimed. |
| Airflow workflow | Simulated | `scripts/run_airflow_simulation.py` is SIMULATION. |
| SageMaker workflow | Planned | Local pipeline compilation exists; managed execution is not established. |
| Advisory provenance | Partial | `docs/ADVISORY_AUDIT.md`; retained source-backed entry is unreviewed, remaining entries are placeholders. |
| Documentation evidence pack | Partial | Sanitized report snapshots are committed under `reports/snapshot/`; runtime screenshots and real promotion/rollback evidence are not in the snapshot. |

## Limitations and release notes

- The model uses lab-style leaf images; field validation is absent.
- Background-fill sensitivity depends on segmentation and fill choice. Black-fill drops include
  masking artifacts and do not prove shortcut learning.
- Corn Cercospora / Gray leaf spot is a known weak-recall class in saved evaluation; use a current
  report snapshot for any numeric value.
- Serving confidence is unscaled softmax at T=1; fitted temperature is not applied by the API.
- The active advisory KB has placeholders. Verify dataset license and redistribution rights
  before public release.
