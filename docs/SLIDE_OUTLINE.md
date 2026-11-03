# Evaluation Presentation Slide Deck Outline (12 Slides)

**Project Title**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline
**Presenter**: Lead Machine Learning Engineer / Student Presenter
**Context**: B.Tech CSE-AI Final Evaluation (Sponsored by Syngenta MLOps Pilot)
**Hardware Profile**: NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) | Local Deterministic Execution

---

### Slide 1: Title & Engineering Objective
- **Title**: Engineering an Auditable, Edge-Ready MLOps Pipeline for Crop Leaf Diagnosis
- **Core Positioning**: An evaluation-grade engineering pipeline prototype behind a photo-based advisory—prioritizing data integrity, safe failures, background robustness, and non-hallucinating extension guidance over naive metric chasing.
- **System Pillars**:
  1. *Zero-Leakage Group Splitting* (`<crop>::<id>`).
  2. *Layered Selective Prediction* (Heuristic Pre-check + Free Energy OOD).
  3. *Capture Bias Mitigation* (E0/E1/E2 background randomization: 91% retention).
  4. *Dual Explainability & Faithfulness* (Grad-CAM, SHAP, and pixel deletion curves).
  5. *Audited Advisory Engine v2* (University extension citations + weather rules).
- **Visual**: Project banner, system badge matrix, and Docker service topology.

---

### Slide 2: Data Audit, Group Partitioning & Leakage Prevention
- **Headline**: Auditing PlantVillage Images & Resolving Multi-Shot Capture Leakage
- **Talking Points**:
  - Exact duplicate removal count: **unverified here**; check the current audit report.
  - Multi-shot leaves share camera sequence tokens (e.g. `GH_HL Leaf 173.1`, `173.2`); naive splitting places identical physical leaves in both train and test splits.
  - Furthermore, independent lab photographers reused generic IDs across different crops (`RS_HL 5672` in Apple and Soybean).
  - Implemented crop-prefixed scoping `<crop>::<id>` with group-aware partitioning; split sizes: **unverified here**.
  - Automated unit test proves mathematically zero intersection between train and test groups.
- **Embedded Visual**: Data manifest audit table from `reports/audit_report.md`.

---

### Slide 3: End-to-End System Architecture
- **Headline**: Decoupled, Observable MLOps Lifecycle Architecture
- **Talking Points**:
  - Decoupled stages: Data Versioning (DVC) $\rightarrow$ Training & MLflow $\rightarrow$ provisional Promotion Gate $\rightarrow$ Multi-Layer Serving (FastAPI) $\rightarrow$ three-tab Streamlit Dashboard $\rightarrow$ simulated drift assessment.
  - Layered input validation: Layer 1 optical heuristics immediately reject non-leaf objects ($<0.1$ ms); Layer 2 Helmholtz energy score handles OOD distribution shifts.
  - Crop-aware hierarchical switch dynamically renormalizes class probabilities.
- **Embedded Visual**: Mermaid architecture diagram from `docs/ARCHITECTURE.md`.

---

### Slide 4: Neural Model Training & MLflow Tracking
- **Headline**: ResNet-18 with Class-Weighted Cross-Entropy on Host GPU
- **Talking Points**:
  - Class imbalance: use counts from the current dataset report.
  - Training configuration and hardware: use current run provenance.
  - MLflow records configured run parameters and metrics.
- **Embedded Visual**: MLflow loss and Macro-F1 progression charts.

---

### Slide 5: Baseline Evaluation & Non-Parametric Bootstrap Validation
- **Headline**: High Evaluation Performance with Rigorous Statistical Bounds
- **Talking Points**:
  - Evaluation sample count and bootstrap settings: **unverified here**; confirm from current report.
  - Saved report `reports/metrics.json`: accuracy 99.30%; macro-F1 0.9903 with 95% bootstrap interval 0.9871–0.9931.
  - Expected Calibration Error (ECE): use the reproducible unscaled VAL/TEST artifact values; temperature scaling is diagnostic only and is not applied at serving unless a current VAL refit improves VAL NLL.
  - Serving latency: **unverified here**; confirm from current benchmark.
- **Embedded Visual**: Precision-Recall curves (`reports/figures/pr_curve.png`) and Calibration curve (`reports/figures/calibration_curve.png`).

---

### Slide 6: Capture Bias Probe & Background Randomization
- **Headline**: Attacking Spurious Laboratory Background Memorization
- **Talking Points**:
  - Noyan (2022) background probe: Random Forest trained solely on 4 corner background pixels achieved 46.1% accuracy (chance is 2.6%).
  - Built automated foreground segmentation via Woebbecke Excess Green Index ($ExG = 2G - R - B$); quantitative extraction results are unverified here.
  - Experiment metrics: **unverified here**; insert only values present in the current bias-study report.
- **Embedded Visual**: `reports/figures/bias_ablation_comparison.png` and `reports/figures/segmentation_40_grid.png`.

---

### Slide 7: Hierarchical Crop-Aware Inference & Solanaceae Disambiguation
- **Headline**: Resolving Cross-Genus Confusion via Hierarchical Selective Prediction
- **Talking Points**:
  - Solanaceae crops (Potato and Tomato) share visual morphology and common pathogens (*Alternaria*, *Phytophthora*).
  - Evaluation sample count and comparative accuracy: **unverified here**; check the current hierarchy report.
- **Embedded Visual**: Solanaceae Confusion Matrix (`reports/figures/potato_tomato_confusion.png`).

---

### Slide 8: Layered OOD Detection: Heuristic Pre-Check + Energy Score
- **Headline**: Multi-Layer Input Validation Defeating Overconfidence on Noise & Grey Tiles
- **Talking Points**:
  - Specific grey-tile confidence: **unverified here**; verify against a saved OOD report.
  - **Layer 1 Heuristic Pre-Check**: Evaluates flatness, decodability, and color balance; measured latency is unverified here.
  - **Layer 2 Helmholtz Free Energy Score**: $E(x) = -T \cdot \log \sum \exp(z_i/T)$. Out-of-distribution inputs yield high energy.
  - Saved `reports/ood_metrics.json`: layered prefilter+energy AUROC 0.9548 and FPR95 0.1 on that report's benchmark.
- **Embedded Visual**: OOD ROC Curves (`reports/figures/ood_roc_curves.png`).

---

### Slide 9: Dual Explainability: Grad-CAM, SHAP & Deletion Curves
- **Headline**: Proving Saliency Faithfulness via Attribution Mass & Greedy Pixel Deletion
- **Talking Points**:
  - Implemented dual explainability: Hook-based Grad-CAM (layer4) and game-theoretic SHAP GradientExplainer.
  - Attribution mass and deletion benchmark values: **unverified here**; confirm from current explainability reports.
- **Embedded Visual**: Deletion Curve (`reports/figures/explain_deletion_curve.png`) and side-by-side SHAP visual (`reports/figures/shap/`).

---

### Slide 10: Agronomic Advisory Engine v2 & Weather-Driven Risk
- **Headline**: Traceable Knowledge Base with Epidemiological Risk Rules
- **Talking Points**:
  - Schema v2 advisory engine citing verified extension guides (UC IPM, Cornell, Penn State, ICAR-CPRI, ICAR-IIHR, TNAU).
  - 6 foliar epidemiological risk rules in `kb/weather_rules.yaml` elevating clinical urgency based on temperature and humidity thresholds.
  - Advisory constraints: describe from the current knowledge base; avoid unverified counts or claims.
  - Regulatory Safety: Dosages are deliberately omitted and deferred to physical registered container labels.
- **Embedded Visual**: UI Advisory Card & Source Traceability Matrix (`docs/SOURCE_AUDIT.md`).

---

### Slide 11: Production Serving, Batch Latency & Automated Drift Triggers
- **Headline**: High-Throughput Serving & Closed-Loop Retraining Governance
- **Talking Points**:
  - Asynchronous FastAPI serving with Prometheus instrumentation and JSON lines audit logging.
  - Saved `reports/INFERENCE_BENCHMARK.md`: 110.5 images/sec at batch 1 and 317.4 images/sec at batch 64 under its stated setup.
  - Drift assessment and retraining trigger outputs are SIMULATION, not production telemetry.
- **Embedded Visual**: Throughput scaling chart from `reports/INFERENCE_BENCHMARK.md`.

---

### Slide 12: Responsible AI, Structural Gap Audit & Viva Defense Summary
- **Headline**: Complete Transparency: A Robust, Defensible Engineering Pipeline
- **Talking Points**:
  - Transparent structural gap auditing in `kb/coverage.yaml`: Orange/Squash (no healthy class), Blueberry/Soybean (no disease class).
  - 4-Criterion promotion gate enforcing holistic pipeline health.
  - Reproducibility and test outcomes: report only from current command output.
- **Embedded Visual**: Three-tab Streamlit dashboard overview and coverage matrix.
