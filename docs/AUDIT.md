# System Audit & Baseline Truth Table (T0)

Generated as part of Task T0 (Repository audit & ground truth verification). Guardrail strictly observed: no feature code was modified.

---

## 1. Executive Summary & Verification Matrix

| Component / Claim | Claimed State | Actual State | Evidence (file:line / output) |
| :--- | :--- | :--- | :--- |
| **Model Serving & Registry Connection** | Model Registry + `champion` alias + `make retrain` promotion gate serving live | **Disconnected (Local `.pth`)** | `src/api.py:92-106`: loads `models/best_model.pth` directly via `torch.load()`. No MLflow Registry client or `@champion` alias resolver in serving path. |
| **Docker Containerization** | Listed under "Planned" in early README; `Dockerfile` exists | **Working (File Present, Multi-stage)** | `Dockerfile`: multi-stage target `api` and `ui`. `docker-compose.yml` orchestrates services. Container exists in tree. |
| **Data Drift Detection** | Production drift monitoring via KS test & PSI | **SIMULATION ONLY (synthetic 25 vs 25)** | SIMULATION: `reports/drift_report.md` recorded `Reference images: 25 | Drifted images: 25`; its Gaussian blur/brightness output, including PSI 29.26 at n=25, is not evidence of real-log drift. It had no TRAIN baseline and did not read live API logs. WP4 adds a real-log monitor; replay is staged traffic. |
| **Near-Duplicate & Group Source Bias** | Zero duplicate pairs found; 0/38 classes with single source code confirms bias | **Partial Sampling & Non-Sequitur** | `scripts/audit_and_split.py:323`: near-duplicate check sampled only 1,500 of 38,002 images (`NEAR_DUP_SAMPLE = min(1500, ...)`). In `reports/audit_report.md`: "Classes with single source code: 0/38 -> capture-source bias confirmed" is mathematically contradictory (0 single-source classes means all classes had multiple sources; conclusion was inverted). |
| **Model Card Accuracy & Metrics** | 100% accuracy on single-class crops; 99.03% macro-F1 | **Confirmed on Test Split, but Hardcoded** | `reports/per_crop_metrics.json`: Blueberry, Raspberry, Soybean, Orange, Squash all exhibit 100% Macro-F1 / 100% Accuracy on test split. However, `docs/MODEL_CARD.md` values are manually typed markdown, not dynamically fetched from MLflow runs. |
| **Advisory Source Attribution** | Sourced from Cornell Cooperative Extension (`plantdisease.net`) | **Spurious Domain / Invalid Attribution** | `docs/SOURCE_AUDIT.md:15` & `kb/advisories.yaml:97, 102`: cites `https://plantdisease.net/tomato_late_blight` attributing to Cornell CALS. Web audit confirms `plantdisease.net` is not an official Cornell domain (Cornell's official domain is `plantclinic.cornell.edu`). |
| **Softmax Confidence Threshold** | Selective prediction via `confidence_threshold: 0.50` | **Uncalibrated Softmax** | `configs/config.yaml:70` & `src/api.py:328`: raw softmax confidence thresholded at 0.50. Softmax probabilities are uncalibrated and overconfident on out-of-distribution inputs. |
| **Single-Crop Shortcut & Macro-F1 Dominance** | Headline Macro-F1 = 99.03% | **Confirmed Dominance** | Across 38 classes, 5 crops have only a single class (all healthy or single disease), yielding trivial 1.0 F1. Within-crop multi-disease performance drops (e.g. Potato Macro-F1 = 73.71%, Tomato Macro-F1 = 82.10% in `per_crop_metrics.json`). |
| **Training Dataset Scale** | Subsample (~300/class) vs Full Data | **Full-data model checkpoint present** | Baseline subsample was replaced on 2026-10-14 with 38,002 image run (`models/best_model.pth` has 38 classes, evaluation report evaluates 8,134 test images). |
| **CI / CD Pipeline** | GitHub Actions automation | **Working** | `.github/workflows/ci.yml` exists; runs pytest, linting, and smoke tests. |
| **DVC Pipeline** | DVC data versioning | **Present but local only** | `dvc.yaml` defines stages `audit_split`, `train`, `evaluate`, `report`. `.dvc/` initialized. |
| **Apache Airflow & Cloud SageMaker** | Airflow DAG & SageMaker Pipeline | **SIMULATION only** | `dags/plant_disease_retraining_dag.py` and `pipelines/sagemaker_pipeline.py` are definitions; `scripts/run_airflow_simulation.py` output is SIMULATION. No live Airflow or AWS daemon is implied. |
| **SHAP / XAI Explainability** | SHAP / LIME explainability | **Working (Scripts & UI on-demand)** | `scripts/benchmark_xai.py` and `reports/xai_benchmark.json` exist. SHAP GradientExplainer and LIME implemented in `app/ui.py`. Offline figures stored in `reports/figures/shap/`. |

---

## 2. Command Execution Evidence

### A. Test Suite (`pytest -q`)
Command: `..\.venv\Scripts\pytest.exe -q`
Result: **23 passed, 1 failed, 5 warnings** (19.58s)
- **Failure**: `tests/test_advisory_kb.py::test_no_fungicide_in_viral_pest_hlb`
  - Cause: `orange/haunglongbing_(citrus_greening)` entry contains the word "fungicide" in a negative context (*"no fungicide or chemical cures the bacteria... negative assertion: fungicides must never be sprayed for hlb"*), tripping the strict substring assertion `assert keyword not in rec_clean`.

### B. Training Script (`python scripts/train.py --help`)
Command: `..\.venv\Scripts\python.exe scripts/train.py --help`
Output:
```text
usage: train.py [-h] [--config CONFIG]

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to config YAML
```

### C. Live API Endpoints (`/health`, `/predict`, `/stats`)
Tested via in-process ASGI runner (`src.api:app`):
- `GET /health` -> `200 OK`: `{"status": "ok", "model_loaded": true, "device": "cuda"}`
- `GET /stats` -> `200 OK`: `{"request_count": 0, "mean_latency_ms": 0.0, "mean_confidence": 0.0, "class_distribution": {}}`
- `POST /predict` (with `demo_samples/Apple___Apple_scab.jpg`) -> `200 OK`:
  - `crop`: `"apple"`
  - `condition`: `"apple_scab"`
  - `confidence`: `1.0`
  - `status`: `"ok"`
  - `latency_ms`: `22.4`

---

## 3. Prior Art on Shortcut & Background Bias in Plant Pathology

1. **Noyan, M. A. (2022)**. *Uncovering bias in the PlantVillage dataset*. [arXiv:2206.04374](https://arxiv.org/abs/2206.04374).
   - *Key finding*: A classifier trained strictly on an 8-pixel border background patch (completely discarding the leaf) achieved **49.0% test accuracy** across PlantVillage classes (random guessing = 2.6%). Demonstrates severe background shortcut learning.
2. **Winkler, M., et al. (2021)**. *Association of Background and Disease Symptoms in Plant Classification Models*.
   - Evaluated out-of-domain transfer from lab-captured leaves to in-field conditions (PlantDoc benchmark); observed dramatic drops exceeding 40% in accuracy when natural field backgrounds were introduced.

---

## 4. Hardcoded Constants, Thresholds & Seeds Audit

- `scripts/run_bias_experiments.py`: Hardcoded `seed=42`, `seed=123`, and probe sampling caps `15 images/class`.
- `scripts/reproduce_check.py`: Hardcoded seeds `seed=42` across two forward passes instead of importing from `cfg.seed`.
- `scripts/audit_and_split.py:323`: Hardcoded `NEAR_DUP_SAMPLE = min(1500, len(deduped_records))`.
- `scripts/audit_and_split.py:231`: Hardcoded `SIZE_SAMPLE_MAX = 200`.
- `configs/config.yaml:70`: `confidence_threshold: 0.50` (raw uncalibrated threshold).
- `kb/advisories.yaml:97, 102`: Hardcoded invalid URL `https://plantdisease.net/tomato_late_blight`.

---

## 5. Addendum: Files Found After T0

Classification of additional operational scripts, pipelines, and documentation against Master Plan v2:

| File / Component | State | Real Implementation / Replacement Task in Master Plan v2 | Evidence & Source Analysis |
| :--- | :--- | :--- | :--- |
| `scripts/run_airflow_simulation.py` & `dags/plant_disease_retraining_dag.py` | **SIMULATION** | Replaced by **T16 (Airflow Orchestration in Container)** | The script output is SIMULATION, executed locally without a live Airflow server. T16 runs containerized Airflow with DAG import tests. |
| `pipelines/sagemaker_pipeline.py` & `scripts/compile_sagemaker_pipeline.py` | **Docs-only / Compile-only** | Replaced / documented by **T20 (Cloud Path Docs)** | Compiles pipeline definition to local JSON without AWS credentials (`scripts/compile_sagemaker_pipeline.py`). Consistent with AGENTS.md rule: local-first, zero paid cloud services. |
| `scripts/check_drift_and_trigger.py` | **SIMULATION ONLY** | Replaced by `scripts/drift_exporter.py` and the Prometheus alert | Reads only an existing legacy simulation artifact; it refuses to create synthetic observations and never starts retraining. |
| `scripts/reproduce_check.py` | **Working** | Preserved & integrated in **T5 (Honest Evaluation & Verification)** | Forward pass test runs on GPU/CPU with seed 42; verified 100% agreement and zero logit delta. |
| `scripts/verify_state.py` | **Working** | Used for environment state validation | Standalone sanity script verifying class distributions, GPU, and paths. |
| `dvc.yaml` | **Partial** | Upgraded in **T3 (DVC Pipeline)** | Tracks stages `audit_split`, `train`, `evaluate`, `report`. Needs local remote configuration, stage expansion (`robustness`, `register`), and `dvc.lock` synchronization. |
| `.github/workflows/ci.yml` | **Working** | Upgraded in **T14 (CI GitHub Actions)** | Real GitHub Actions workflow running ruff linting, pytest, and safety checks on push/PR. |
| `src/segmentation.py` & `scripts/benchmark_segmentation.py` | **Working** | Integrated in **T6 (Shortcut-Robustness Suite)** | Woebbecke ExG leaf segmentation verified functional across 40 benchmark samples; generated `reports/segmentation_quality.json`. |
| `src/ood_detector.py` & `scripts/evaluate_ood.py` | **Working** | Refactored in **T8 (Calibration, OOD, Quality, Abstention)** | Layer 1 optical heuristics + Layer 2 Helmholtz Energy score benchmarked with AUROC 0.9548 in `reports/ood_report.md`. |
| `scripts/evaluate_hierarchy.py` | **Working** | Integrated in **T5 & T8** | Evaluates masked softmax crop constraint; eliminates Solanaceae cross-genus confusion. |
| `scripts/run_bias_experiments.py` | **Working** | Integrated in **T6 (Shortcut-Robustness Suite)** | Evaluates background swap retention (E0/E1/E2); recorded in `reports/BIAS_STUDY.md`. |
| `docs/DATASHEET.md`, `docs/RISK_REGISTER.md` | **Docs-only (Accurate)** | Preserved in **T21 (Viva & Governance Pack)** | Documentation aligned with Gebru datasheet format and ISO/IEC 23894 Risk Register. |
| `docs/CLOUD_MAPPING.md`, `docs/SYLLABUS_MAP.md` | **Docs-only (Accurate)** | Preserved in **T20 & T21** | Aligned with VIT AY 2026-27 MLOps syllabus Units I-VI. |
| `docs/VIVA_QA.md`, `docs/SLIDE_OUTLINE.md`, `docs/DEMO_SCRIPT.md` | **Docs-only** | Refreshed in **T21 & T22 (Demo Rehearsal)** | Scripts and presentation materials for examination defense. |
