# System Architecture & Technical Specification

**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline
**Course**: B.Tech CSE-AI (Sponsored by Syngenta MLOps Evaluation Pilot)
**Status**: Prototype Engineering Pipeline (Phase 3 Expanded Evaluation-Grade)

---

## 1. End-to-End System Architecture

The pipeline adheres to modern MLOps principles, decoupling data versioning, neural training, model registration, containerized serving, selective prediction, and telemetry monitoring into auditable, reproducible stages:

```mermaid
flowchart TD
    subgraph Data_Engineering["1. Data Engineering & Group Partitioning"]
        Raw["Raw PlantVillage (54,305 Images)"] --> Audit["scripts/audit_and_split.py\n(SHA-256 Dedup: -21 duplicates)"]
        Audit --> ScopedGroup["Group Scoping:\n<crop>::<id>"]
        ScopedGroup --> Splits["Stratified Splits (70/15/15)\nTrain: 38,002 | Val: 8,148 | Test: 8,134\n(Zero Group Overlap Asserted)"]
        Splits --> DVC["DVC Pipeline Tracking\n(dvc.yaml)"]
    end

    subgraph Training_MLOps["2. Training, Experiment Tracking & Gate"]
        DVC --> Train["scripts/train.py\nResNet-18 (AMP FP16, Class-Weighted CE)"]
        Train --> MLflow["MLflow Tracking (SQLite Store)\nParams, Epoch Loss, F1 curves"]
        Train --> Eval["scripts/evaluate.py\n11 Diagnostic Figures, Bootstrap CIs,\nBias Probe & Calibration"]
        BiasExp["scripts/run_bias_experiments.py\n(E0 vs E1 vs E2 Background Ablation)"] --> Gate["scripts/promote.py\n(4-Criterion Promotion Gate)"]
        OODEval["scripts/evaluate_ood.py\n(Energy AUROC >= 0.85)"] --> Gate
        Eval --> Gate
        Gate --> Registry["MLflow Model Registry\nAlias: 'champion' after approved gate"]
    end

    subgraph Serving_Layer["3. Multi-Layer Serving & Explainability"]
        Registry -.-> PthCheck["models/best_model.pth\n(Mounted Checkpoint)"]
        PthCheck --> API["src/api.py\nFastAPI Serving (:8000)"]

        InputImg["Inbound Image Query"] --> L1PreCheck{"Layer 1: Heuristic Pre-Check\n(Min 64x64, std >= 8.0,\nVeg/Lesion Color Ratio)"}
        L1PreCheck -- Fail --> RejL1["Status: rejected_not_a_leaf\n(Code: NOT_A_LEAF_COLOR / FLAT)"]

        L1PreCheck -- Pass --> L2Energy{"Layer 2: Free Energy Score\nE(x) = -T * logsumexp(z/T)\nThreshold = -12.44"}
        L2Energy -- Out-of-Dist --> RejL2["Status: rejected_not_a_leaf\n(Code: OOD_ENERGY_SCORE)"]

        L2Energy -- In-Dist --> CropSelect{"Hierarchical Mode\n(Crop specified?)"}
        CropSelect -- Yes --> MaskedSoftmax["Masked Softmax\nRenormalize over Crop Classes"]
        CropSelect -- No (Auto) --> GlobalSoftmax["Global 38-Class Softmax"]

        MaskedSoftmax --> ConfCheck{"In-Crop Confidence\n>= 0.50?"}
        GlobalSoftmax --> ConfCheck

        ConfCheck -- High Conf --> OkResp["Status: OK\nTop-5 Predictions"]
        ConfCheck -- Low Conf --> UncertResp["Status: UNCERTAIN\nConfidence < 0.50"]

        OkResp --> AdvEngine["src/advisory_engine.py\n(Schema v2 + Weather Rules)"]
        AdvEngine --> AdvCard["Advisory Payload:\nCultural, Chemical, PPE, PHI, Weather Risk"]

        OkResp --> Explainers["Explainability Suite:\n- Grad-CAM (Layer 4 Hooks)\n- SHAP GradientExplainer"]

        API --> JSONL["logs/requests.jsonl\n(Latency, Brightness, Blur, Class)"]
        API --> Prom["GET /metrics\nPrometheus Metric Instrumentation"]
    end

    subgraph User_Interface["4. Presentation, Monitoring & Governance"]
        API --> Streamlit["app/ui.py (Streamlit :8501)\n5 Interactive Tabs"]
        JSONL --> Streamlit
        JSONL --> Drift["scripts/simulate_drift.py\nTwo-sample KS-Test & PSI Shift"]
        Drift --> DriftTrigger["scripts/check_drift_and_trigger.py\n(retraining_trigger.json)"]
    end
```

---

## 2. Component Responsibilities

| Subsystem | Primary Module / Path | Key Technical Responsibility |
| :--- | :--- | :--- |
| **Data Audit & Partitioning** | `scripts/audit_and_split.py` | Detects SHA-256 exact duplicates, verifies class counts, groups multi-shot images via `<crop>::<source_code> <id>`, and outputs zero-leakage manifest CSVs. |
| **Foreground Segmentation** | `src/segmentation.py` | Implements Woebbecke Excess Green Index ($ExG = 2G - R - B$) with morphological opening and largest connected component extraction for background ablation. |
| **Model Architecture** | `src/model.py` | Instantiates ResNet-18 with ImageNet weights, replaces the final linear layer for 38 classes, and attaches forward/backward hooks on `layer4` for Grad-CAM. |
| **Experiment Tracker** | `scripts/train.py` | Implements automatic mixed precision (AMP FP16) training, class-weighted cross-entropy loss, cosine annealing, and records metrics to an SQLite-backed MLflow instance. |
| **Capture Bias Ablation** | `scripts/run_bias_experiments.py` | Benchmarks E0 (Baseline), E1 (Background-Randomized 80%), and E2 (Mixed 40/60) across clean, background-swapped, and center-masked evaluation sets. |
| **Out-of-Distribution Detector** | `src/ood_detector.py` | Multi-layered rejection engine: Layer 1 heuristic optical filtering + Layer 2 Helmholtz free energy score $E(x) = -T \cdot \log \sum \exp(z_i / T)$ preventing confident non-leaf misclassifications. |
| **Hierarchical Inference** | `src/api.py` (`/predict`) | Evaluates optional crop guidance; restricts prediction to genus/species candidate classes via masked softmax renormalization, resolving Potato vs. Tomato confusion. |
| **Explainability Suite** | `scripts/run_explainability.py` | Produces dual Grad-CAM and SHAP attribution heatmaps, calculates attribution mass on leaf foreground, and validates faithfulness via greedy pixel deletion curves. |
| **Model Registry & Gate** | `scripts/promote.py` | Enforces a provisional four-criterion gate: within-crop VAL macro-F1 retention, per-class recall floor, champion-relative masked-background drop, and ECE. |
| **Advisory Engine v2** | `src/advisory_engine.py` | Matches Schema v2 advisories from `kb/advisories.yaml` and foliar epidemiological risk from `kb/weather_rules.yaml` with negative constraint checks. |
| **Batch Inference Engine** | `scripts/batch_infer.py` | High-throughput batch inference engine processing image manifests via mini-batched tensors (110.5 fps single, 317.4 fps batch-64). |
| **Telemetry & Drift Engine** | `scripts/simulate_drift.py` & `check_drift_and_trigger.py` | Logs incoming query features (blur, brightness, latency), runs two-sample KS and PSI tests, and issues automated retraining trigger manifests upon drift breach. |
| **Reproducibility Harness** | `scripts/reproduce_check.py` | Validates strict deterministic forward passes, asserting zero logit divergence across identical seeds. |
| **Web Dashboard** | `app/ui.py` | Five-tab Streamlit dashboard offering Diagnostic Screening, Telemetry Monitoring, Bias Study Evidence, Coverage & Structural Gaps, and MLOps Governance. |

---

## 3. Layered Selective Prediction Architecture

A critical vulnerability identified in standard image classifiers is overconfidence on out-of-distribution (OOD) noise, grey tiles, or non-leaf objects:

```mermaid
flowchart LR
    A["Raw Inbound Image"] --> B["Pre-Check Heuristics"]
    B -- "Flat / Low std (< 8.0) / Non-Veg Color" --> R1["REJECT: Not A Leaf\n(Layer 1 - 0ms overhead)"]
    B -- "Plausible Leaf Image" --> C["ResNet-18 Logits z"]
    C --> D["Free Energy Score E(x)\nE(x) = -T * logsumexp(z / T)"]
    D -- "E(x) > -12.44 (OOD)" --> R2["REJECT: OOD Input\n(Layer 2 - AUROC 0.9548)"]
    D -- "E(x) <= -12.44 (ID Leaf)" --> E["Crop-Aware Softmax Gate"]
    E -- "Top Probability < 0.50" --> R3["UNCERTAIN: Low Confidence"]
    E -- "Top Probability >= 0.50" --> S["STATUS: OK\nDiagnosis + Verified Advisory"]
```

---

## 4. API Contract & Schema Specification

The FastAPI serving service (`src/api.py`) exposes the following endpoints:

| Endpoint | HTTP Method | Input Parameters | Output Response |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `{"status": "ok", "model_version": "v2", "device": "cuda", "champion_model": "models/best_model.pth"}` |
| `/predict` | `POST` | `file: UploadFile`<br>`crop: Optional[str]`<br>`temperature: Optional[float]`<br>`humidity: Optional[float]` | JSON containing `status` (`ok`, `uncertain`, `rejected_not_a_leaf`), `predicted_class`, `confidence`, `crop_mode`, `top5`, `advisory`, and base64 `gradcam_overlay`. |
| `/predict_batch`| `POST` | `files: List[UploadFile]`<br>`crop: Optional[str]` | JSON list of diagnosis payloads. |
| `/stats` | `GET` | None | Summary statistics of requests logged in `logs/requests.jsonl`. |
| `/metrics` | `GET` | None | Prometheus-formatted metric gauges and request counters. |

---

## 5. Process Lifecycle & Docker Compose

The production deployment runs via Docker Compose with continuous health monitoring and restart policies:

```yaml
services:
  api:
    build: .
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  ui:
    build: .
    command: streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0
    ports: ["8501:8501"]
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
```
