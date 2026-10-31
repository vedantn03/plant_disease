# Plant Disease Identification & Advisory Pipeline

> **Course Project — B.Tech CSE-AI (MLOps)**
> Sponsored by Syngenta. Evaluated by Syngenta professionals and faculty.
>
> ⚠️ **THIS IS A PROTOTYPE DEMO PIPELINE. NOT A FIELD-READY DIAGNOSTIC TOOL.**
> See [Limitations](#limitations) and the [Model Card](docs/MODEL_CARD.md).

---

## What This Is

A local-first, reproducible MLOps pipeline that:
1. Classifies a leaf image into one of **38 PlantVillage classes** (14 crops).
2. Returns prediction, confidence, and a **Grad-CAM** heatmap highlighting the evidence region.
3. Returns a **treatment advisory** from a cited, auditable knowledge base — never model-generated.
4. Demonstrates core MLOps concepts end-to-end.

---

## Quick Setup (One Command)

```powershell
# From the plant_disease/ directory

# 1. Install PyTorch (CUDA 12.8 wheel for RTX 3050)
..\\.venv\\Scripts\\pip.exe install torch==2.11.0+cu128 torchvision==0.26.0+cu128 `
    --index-url https://download.pytorch.org/whl/cu128

# 2. Install remaining dependencies
..\\.venv\\Scripts\\pip.exe install -r requirements.txt

# 3. Copy env file and set DATA_ROOT
copy .env.example .env
# Edit .env: set DATA_ROOT=<path to plantvillage dataset/color folder>

# 4. Verify GPU
..\\.venv\\Scripts\\python.exe scripts\\verify_gpu.py
```

---

## Running Each Step

### Step 2 — Audit, Deduplicate, Split
```powershell
..\\.venv\\Scripts\\python.exe scripts\\audit_and_split.py
# Output: data/splits/{train,val,test}.csv, reports/audit_report.md
```

### Step 3 — Train (Full data, 8 epochs)
```powershell
..\\.venv\\Scripts\\python.exe scripts\\train.py
# Output: models/candidates/<run_id>.pth, reports/<run artifacts>,
#         and an MLflow candidate run. This command never overwrites
#         models/best_model.pth; promotion is a separate dry-run gate.
```

### MLflow UI
```powershell
..\\.venv\\Scripts\\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open: http://localhost:5000
```

### Step 4 — Start FastAPI Server
```powershell
..\\.venv\\Scripts\\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000 --reload
# Health check: http://localhost:8000/health
# Interactive docs: http://localhost:8000/docs
```

### Test the API (curl or PowerShell)
```powershell
# Predict with a leaf image
$form = @{
    file = Get-Item "path\\to\\leaf.jpg"
    temperature_c = 28.0
    humidity_pct = 80.0
}
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Form $form

# Or use curl:
curl -X POST http://localhost:8000/predict -F "file=@leaf.jpg" -F "temperature_c=28"
```

### Step 5 — Tests
```powershell
..\\.venv\\Scripts\\pytest.exe tests/ -v
```

### Step 6 — Drift Simulation
```powershell
# Start the API first, then:
..\\.venv\\Scripts\\python.exe scripts\\simulate_drift.py
# Output: reports/drift_report.md, reports/drift_plot.png
```

---

## 10-Minute Demo Script

| # | Action | Command / URL |
|---|--------|---------------|
| 1 | Show audit report | `type reports\audit_report.md` |
| 2 | Show split sizes | `type data\splits\manifest_hashes.json` |
| 3 | Open MLflow UI | `http://localhost:5000` — show run params, metrics, confusion matrix |
| 4 | Predict healthy leaf | `POST /predict` with a healthy leaf image |
| 5 | Predict diseased leaf | `POST /predict` with a tomato early blight image |
| 6 | Show Grad-CAM | Decode `gradcam_png_base64` and display |
| 7 | Show advisory | Show VERIFIED advisory (Tomato Early Blight) + PLACEHOLDER |
| 8 | Show uncertain | Send a non-leaf / blurry image → `status=uncertain` |
| 9 | Show /stats | `GET http://localhost:8000/stats` |
| 10| Drift demo | `python scripts/simulate_drift.py` → show drift_plot.png |

---

## MLOps Concepts Demonstrated

| Concept | Where |
|---------|-------|
## Performance & Headline Evaluation (Held-Out Test Split, N=8,134)

> All numbers from `reports/metrics.json` (run `make evaluate` to reproduce).

| Metric | Point Estimate | 95% Bootstrap Confidence Interval | Evaluation Scope |
| :--- | :---: | :---: | :--- |
| **Macro-F1 (Headline)** | **99.03%** | [98.71%, 99.31%] | 38 classes, equal class weighting |
| **Weighted-F1** | **99.30%** | [99.12%, 99.47%] | Frequency-weighted |
| **Overall Accuracy** | **99.30%** | [99.13%, 99.47%] | Held-out zero-leakage split |
| **GPU Latency (RTX 3050)** | **2.54 ms** | p95: 4.19 ms | PyTorch AMP FP16 |
| **CPU Latency (Host)** | **61.98 ms** | p95: 64.74 ms | Single-image synchronous |

> **Training Dataset Disclosure**: Champion model trained on the **full training split —
> 38,002 images (no subsampling), 8 epochs, AMP FP16, RTX 3050 GPU**.
> MLflow run ID: `fb731412d82544e0abc231b8d7c62f95`.
>
> ⚠️ These numbers reflect the PlantVillage lab-style dataset. Real-field accuracy is
> unknown. See [Limitations](#limitations) and `reports/STATE_VERIFICATION.md`.

---

## One-Command Quickstart Stack

```powershell
# 1. Launch FastAPI Serving Backend (Terminal 1)
make serve
# API runs on http://127.0.0.1:8000 (Swagger docs at /docs)

# 2. Launch Interactive Streamlit UI (Terminal 2)
make ui
# Opens in browser at http://localhost:8501

# 3. Run Full Evaluation Suite (Regenerate 11 Figures + Reports)
make evaluate

# 4. Run Test Suite (18 Unit Tests)
make test
```

---

## Evaluation Screenshots Checklist

Capture the following 4 screenshots during your live demo for report/slide submission:

1. **`screenshot_01_diagnose_tomato.png`**: Tab 1 (Diagnose) with `Tomato___Early_blight.jpg` selected, showing 99.5% confidence, Grad-CAM heatmap overlay, and the **VERIFIED** UC IPM advisory with the weather warning box.
2. **`screenshot_02_rejection_non_leaf.png`**: Tab 1 (Diagnose) with `non_leaf_sample.jpg` selected, displaying the **UNCERTAIN** orange status banner rejecting the low-confidence non-leaf input.
3. **`screenshot_03_monitoring_telemetry.png`**: Tab 2 (Monitoring) showing live request volume, latency distribution (median ~26ms), confidence histogram, and Laplacian blur score distributions.
4. **`screenshot_04_bias_probe_evidence.png`**: Tab 3 (Model & Evidence) displaying the Noyan Capture-Bias Probe graph (`fig_09_bias_probe.png`) and the Per-Crop Slice table with single-class crops flagged.

---

## MLOps Concepts Demonstrated

| Concept | Implementation in this Repository |
| :--- | :--- |
| **Data Versioning / Validation** | `scripts/audit_and_split.py` — SHA-256 deduplication (21 copies removed), group-aware split with `<crop>::<id>` scoping, zero group leakage unit-asserted. |
| **Experiment Tracking** | MLflow with persistent SQLite backend (`sqlite:///mlflow.db`) logging parameters, epoch metrics, confusion matrix, and per-class reports. |
| **Model Registry & Gate** | MLflow registration and the `champion` alias are implemented in `src/registry.py` and `scripts/promote.py`; registry state is runtime-local and must be verified from MLflow artifacts. |
| **Inference Serving** | FastAPI (`src/api.py`) with automatic input validation, Grad-CAM attribution, confidence thresholding (0.50), and JSON lines request logging. |
| **Interactive Dashboard** | Streamlit single-page application (`app/ui.py`) with Diagnose, Monitoring, and Model & Evidence tabs. |
| **Drift Monitoring** | `scripts/simulate_drift.py` applying Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) to detect optical distribution shifts. |
| **Responsible AI & Governance** | Pure PyTorch Grad-CAM hooks, temperature-scaled calibration (ECE), empirical bias probe replication, and strictly cited agricultural advisories (`docs/SOURCE_AUDIT.md`). |

---

## Limitations

> **Critical**: This model is trained on lab-style single-leaf images with uniform backgrounds.
> Replicating the Noyan (2022) study on our split, a Random Forest classifier trained on **only 4 corner pixels** achieved **46.1% test accuracy** (chance is 2.6%), proving that background capture lighting carries strong non-biological signals.

- **Lab vs. field gap**: Background bias is severe. Model accuracy in field conditions with soil backgrounds and shadows is unknown.
- **Single-class crops**: Blueberry, Raspberry, Soybean (healthy only); Orange, Squash (disease only) represent **26% of all data**, meaning crop identity alone trivially solves those classes.
- **Class imbalance**: ~36:1 ratio. Handled via class-weighted CE, but small classes (`Potato___healthy`, support 23) show higher metric noise.
- **No severity rating**: Binary diagnostic output; does not rate percentage of leaf surface infected.
- **Advisory disclaimer**: No chemical dosages are invented; users are directed to physical registered product labels.

---

## Documentation Pack

- [System Architecture & Mermaid Diagram](docs/ARCHITECTURE.md)
- [10-Minute Timed Viva Demo Script](docs/DEMO_SCRIPT.md)
- [25 Comprehensive Viva Q&A Guide](docs/VIVA_QA.md)
- [12-Slide Evaluation Presentation Outline](docs/SLIDE_OUTLINE.md)
- [Advisory Citation Source Audit](docs/SOURCE_AUDIT.md)
- [Model Card (Metrics, Robustness, Bias Probe)](docs/MODEL_CARD.md)
- [Evaluation Report with Embedded Figures](reports/EVALUATION_REPORT.md)
