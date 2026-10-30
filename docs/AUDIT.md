# System Audit & Baseline Truth Table (T0)

Generated as part of Task T0 (Repository audit & ground truth verification). Guardrail strictly observed: no feature code was modified.

---

## 1. Executive Summary & Verification Matrix

| Component / Claim | Claimed State | Actual State | Evidence (file:line / output) |
| :--- | :--- | :--- | :--- |
| **Model Serving & Registry Connection** | Model Registry + `champion` alias + `make retrain` promotion gate serving live | **Disconnected (Local `.pth`)** | `src/api.py:92-106`: loads `models/best_model.pth` directly via `torch.load()`. No MLflow Registry client or `@champion` alias resolver in serving path. |
| **Docker Containerization** | Listed under "Planned" in early README; `Dockerfile` exists | **Working (File Present, Multi-stage)** | `Dockerfile`: multi-stage target `api` and `ui`. `docker-compose.yml` orchestrates services. Container exists in tree. |
| **Data Drift Detection** | Production drift monitoring via KS test & PSI | **Script-generated (Synthetic 25 vs 25)** | `reports/drift_report.md`: `Reference images: 25 | Drifted images: 25`. Synthetic Gaussian blur ($\sigma=3.0$), brightness (0.6). PSI = 29.26 on $n=25$. No baseline stored from training set, not reading live API logs. |
| **Near-Duplicate & Group Source Bias** | Zero duplicate pairs found; 0/38 classes with single source code confirms bias | **Partial Sampling & Non-Sequitur** | `scripts/audit_and_split.py:323`: near-duplicate check sampled only 1,500 of 38,002 images (`NEAR_DUP_SAMPLE = min(1500, ...)`). In `reports/audit_report.md`: "Classes with single source code: 0/38 -> capture-source bias confirmed" is mathematically contradictory (0 single-source classes means all classes had multiple sources; conclusion was inverted). |
| **Model Card Accuracy & Metrics** | 100% accuracy on single-class crops; 99.03% macro-F1 | **Confirmed on Test Split, but Hardcoded** | `reports/per_crop_metrics.json`: Blueberry, Raspberry, Soybean, Orange, Squash all exhibit 100% Macro-F1 / 100% Accuracy on test split. However, `docs/MODEL_CARD.md` values are manually typed markdown, not dynamically fetched from MLflow runs. |
| **Advisory Source Attribution** | Sourced from Cornell Cooperative Extension (`plantdisease.net`) | **Spurious Domain / Invalid Attribution** | `docs/SOURCE_AUDIT.md:15` & `kb/advisories.yaml:97, 102`: cites `https://plantdisease.net/tomato_late_blight` attributing to Cornell CALS. Web audit confirms `plantdisease.net` is not an official Cornell domain (Cornell's official domain is `plantclinic.cornell.edu`). |
| **Softmax Confidence Threshold** | Selective prediction via `confidence_threshold: 0.50` | **Uncalibrated Softmax** | `configs/config.yaml:70` & `src/api.py:328`: raw softmax confidence thresholded at 0.50. Softmax probabilities are uncalibrated and overconfident on out-of-distribution inputs. |
| **Single-Crop Shortcut & Macro-F1 Dominance** | Headline Macro-F1 = 99.03% | **Confirmed Dominance** | Across 38 classes, 5 crops have only a single class (all healthy or single disease), yielding trivial 1.0 F1. Within-crop multi-disease performance drops (e.g. Potato Macro-F1 = 73.71%, Tomato Macro-F1 = 82.10% in `per_crop_metrics.json`). |
| **Training Dataset Scale** | Subsample (~300/class) vs Full Data | **Full-data model checkpoint present** | Baseline subsample was replaced on 2026-10-14 with 38,002 image run (`models/best_model.pth` has 38 classes, evaluation report evaluates 8,134 test images). |
| **CI / CD Pipeline** | GitHub Actions automation | **Working** | `.github/workflows/ci.yml` exists; runs pytest, linting, and smoke tests. |
| **DVC Pipeline** | DVC data versioning | **Present but local only** | `dvc.yaml` defines stages `audit_split`, `train`, `evaluate`, `report`. `.dvc/` initialized. |
| **Apache Airflow & Cloud SageMaker** | Airflow DAG & SageMaker Pipeline | **Code & Simulation Only** | `dags/plant_disease_retraining_dag.py` and `pipelines/sagemaker_pipeline.py` exist as standalone definitions. Simulated via `scripts/run_airflow_simulation.py`; no live Airflow or AWS daemon. |
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
