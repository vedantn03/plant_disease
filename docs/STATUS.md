# Current Status

The current browser dashboard is a static, farmer-facing HTML/CSS/JavaScript route at `/` with a
separate read-only evaluator route at `/insights`. Marathi explanation/advisory text is a machine
draft where present; source-backed advice falls back per field when no Marathi translation exists.
No human-reviewed translation is claimed. See `docs/UI_REVIEW.md` and
`docs/UI_MIGRATION_AUDIT.md` for the implementation boundary and local verification state.

| Area | Status | Evidence / limitation |
|---|---|---|
| Data/versioning pipeline | Implemented | `dvc.yaml`; execution requires local dataset and DVC state. |
| Train/evaluate | Partial | Code and stages exist; this documentation task did not train or evaluate a model. |
| Promotion gate | Implemented | `scripts/promote.py`; use dry-run for rehearsal. Approval state is separately governed. |
| API serving and browser UI | Implemented in code | `src/api.py`, `web/`; real prediction and image explanation require the local model and, for SHAP, configured TRAIN images. Browser/Compose runtime still requires local verification. |
| Evaluator evidence | Implemented in code | `/insights` reads saved snapshots and local telemetry; latest MLflow run lookup is read-only and only returns allowlisted training parameters/metrics. Runtime database access remains NOT RUN here. |
| Per-image explainability | Implemented in code | Grad-CAM is returned by `/predict`; LIME/SHAP are async API jobs. Real-checkpoint explanation quality and latency remain unverified here. |
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
- Marathi text is marked as a machine draft; missing advisory translations display source-language
  text. Fluent agronomist review is required before farmer-facing reliance.
