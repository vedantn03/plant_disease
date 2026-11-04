# Viva Questions and Evidence Pointers

Answers are limited to the behavior visible in this repository. A report-dependent value should
be read from that report at presentation time; absence means the value is not available.

| # | Question | Concise answer | Evidence pointer |
|---:|---|---|---|
| 1 | What does the project do? | It demonstrates an image classifier and MLOps workflow; it is not field validated. | `README.md` — overview and limitations |
| 2 | How are train, validation, and test partitions managed? | The audit/split script creates manifests and tests check split properties when manifests exist. | `scripts/audit_and_split.py`; `tests/test_split.py` |
| 3 | Why use DVC? | DVC declares data/model/report dependencies and outputs; local remote status remains machine-specific. | `dvc.yaml`; `docs/evidence/dvc/README.md` |
| 4 | What architecture is configured? | The config specifies ResNet-18; inspect the model builder for implementation. | `configs/config.yaml` (`model.arch`); `src/model.py::build_model` |
| 5 | Where does training write checkpoints? | Training writes candidate checkpoints under `models/candidates`; it must not overwrite the champion path. | `scripts/train.py`; `tests/test_dvc_pipeline.py` |
| 6 | What data selects a promotion candidate? | VAL metrics select candidates; TEST is for reporting only. | `docs/DECISIONS.md` — “Candidate selection uses VAL metrics” |
| 7 | What does promotion dry-run do? | It evaluates the gate and records a decision without moving the registry alias. | `scripts/promote.py::main`; `tests/test_promote_dry_run.py` |
| 8 | What guards an apply operation? | Approval state and threshold-file hash are checked alongside gate and registry safety checks. | `src/promotion_safety.py`; `configs/gates/approval.yaml` |
| 9 | How does serving identify the champion? | Registry alias and manifest are checked against a mounted checkpoint; health reports the match. | `src/model_loader.py`; `src/api.py::health` |
| 10 | Why keep weights out of Git? | Model weights are local/DVC-managed artifacts, not source files. | `.gitignore`; `scripts/check_tracked_files.py` |
| 11 | What does API confidence mean, and why is temperature T=1 used? | The API returns unscaled softmax confidence; fitted temperature is diagnostic and is not applied in the serving path. | `src/api.py::predict`; `src/calibration.py`; `docs/DECISIONS.md` |
| 12 | What happens to a low-trust image? | API status, OOD checks, and advice rules determine whether advice is withheld or downgraded. | `src/api.py::predict`; `src/advisory_engine.py::decide_advice` |
| 13 | What does background sensitivity show? | It compares the original prediction with a per-image mean-background version; segmentation/fill artifacts limit interpretation. | `src/background_check.py`; `docs/STATUS.md` |
| 14 | Does a black-fill drop prove shortcut learning? | No. The fill/segmentation can introduce masking artifacts; compare the three-fill protocol. | `docs/GATE_PROPOSAL.md`; `src/robustness.py` |
| 15 | How is the advisory content governed? | Entries carry provenance and evidence grades; placeholders are not full advice. Confusable hints come from saved pairs, and no “how to tell apart” text appears without a cited passage. | `kb/advisories.yaml`; `docs/ADVISORY_AUDIT.md`; `src/advisory_engine.py`; `reports/confused_pairs.csv` |
| 16 | How are chemical dosages handled? | The system does not invent a dose; a chemical option must be quoted from a cited source. | `src/advisory_engine.py`; `tests/test_advisory_kb.py` |
| 17 | How do the API, UI, and monitoring containers connect? | Compose starts API/UI in core and Prometheus/Grafana/drift exporter in monitoring; scrape targets are defined in Prometheus config. | `docker-compose.yml`; `monitoring/prometheus.yml`; `docs/ARCHITECTURE.md` |
| 18 | What does drift monitoring compare? | A request-log sliding window is compared with a stored TRAIN baseline; replay is staged traffic, not field traffic. | `scripts/drift_exporter.py`; `scripts/replay_traffic.py`; `monitoring/prometheus.yml` |
| 19 | What does CI prove about the API? | The temporary fixture model checks plumbing and response paths, not predictive accuracy. | `tests/test_api_fixture_model.py`; `docs/CI_NOTES.md`; `.github/workflows/ci.yml` |
| 20 | Is this ready for field or commercial use? | No. Field validation, advice review, and dataset license checks remain necessary. | `docs/MODEL_CARD.md`; `docs/STATUS.md`; `docs/ADVISORY_AUDIT.md` |
