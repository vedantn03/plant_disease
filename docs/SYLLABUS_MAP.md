# Course Topic Crosswalk

No course syllabus document was present in the repository when this crosswalk was prepared. The
units below map project work to MLOps course topics; they are not quoted official unit titles.

| Unit | Project topic | Status | Evidence and scope |
|---|---|---|---|
| I | Data audit, split validation, and DVC tracking | Implemented | `scripts/audit_and_split.py`, `src/data_validation.py`, `dvc.yaml`; local data and remote state must be checked on the project machine. |
| II | Model training, evaluation, and experiment tracking | Partial | `scripts/train.py`, `scripts/evaluate.py`, `scripts/compute_val_diagnostics.py`; execution depends on local artifacts. |
| III | Model registry, gate, approval, and rollback governance | Implemented | `scripts/promote.py`, `src/promotion_safety.py`, `configs/gates/`; live mutations require operator action. |
| IV | API, container packaging, and Streamlit serving | Implemented | `src/api.py`, `app/ui.py`, `Dockerfile`, `docker-compose.yml`; runtime needs local model artifacts and secrets. |
| V | Observability and drift monitoring | Partial | `scripts/drift_exporter.py`, `monitoring/`; replay is staged, and field drift has not been established. |
| VI | Workflow orchestration and managed cloud deployment | Planned | `scripts/run_airflow_simulation.py` is SIMULATION; `scripts/compile_sagemaker_pipeline.py` only compiles a local definition, not managed execution. |

“Implemented” means repository code and tests exist for the named local workflow. “Partial” means
the full operational claim is not established. “Simulated” means the repository labels the path
as a simulation or local compilation. No managed cloud deployment is claimed as implemented.
