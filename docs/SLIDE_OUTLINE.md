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
- **Headline**: Auditing 54,305 PlantVillage Images & Resolving Multi-Shot Capture Leakage
- **Talking Points**:
  - SHA-256 hash deduplication removed 21 exact identical duplicate images.
  - Multi-shot leaves share camera sequence tokens (e.g. `GH_HL Leaf 173.1`, `173.2`); naive splitting places identical physical leaves in both train and test splits.
  - Furthermore, independent lab photographers reused generic IDs across different crops (`RS_HL 5672` in Apple and Soybean).
  - Implemented crop-prefixed scoping `<crop>::<id>` with stratified 70/15/15 partitioning (Train: 38,002, Val: 8,148, Test: 8,134).
  - Automated unit test proves mathematically zero intersection between train and test groups.
- **Embedded Visual**: Data manifest audit table from `reports/audit_report.md`.

---

### Slide 3: End-to-End System Architecture
- **Headline**: Decoupled, Observable MLOps Lifecycle Architecture
- **Talking Points**:
  - Decoupled stages: Data Versioning (DVC) $\rightarrow$ Training & MLflow $\rightarrow$ 4-Criterion Promotion Gate $\rightarrow$ Multi-Layer Serving (FastAPI) $\rightarrow$ 5-Tab Streamlit Dashboard $\rightarrow$ Live Drift Telemetry & Retraining Trigger.
  - Layered input validation: Layer 1 optical heuristics immediately reject non-leaf objects ($<0.1$ ms); Layer 2 Helmholtz energy score handles OOD distribution shifts.
  - Crop-aware hierarchical switch dynamically renormalizes class probabilities.
- **Embedded Visual**: Mermaid architecture diagram from `docs/ARCHITECTURE.md`.

---

### Slide 4: Neural Model Training & MLflow Tracking
- **Headline**: ResNet-18 with Class-Weighted Cross-Entropy on Host GPU
- **Talking Points**:
  - Handled 36:1 severe class imbalance (5,507 majority vs 152 minority) using inverse-frequency loss weighting, avoiding epoch distortion.
  - Automatic Mixed Precision (AMP FP16) training executed on local RTX 3050 Laptop GPU for 8 epochs with Cosine Annealing.
  - 100% parameter, code version, and metric provenance logged to SQLite-backed MLflow tracking server.
- **Embedded Visual**: MLflow loss and Macro-F1 progression charts.

---

### Slide 5: Baseline Evaluation & Non-Parametric Bootstrap Validation
- **Headline**: High Evaluation Performance with Rigorous Statistical Bounds
- **Talking Points**:
  - Evaluated on 8,134 held-out test images across 1,000 non-parametric bootstrap iterations.
  - **Overall Accuracy**: **99.30%** [95% CI: 99.13%, 99.47%].
  - **Macro-F1**: **0.9903** [95% CI: 98.71%, 99.31%].
  - Expected Calibration Error (ECE): use the reproducible unscaled VAL/TEST artifact values; temperature scaling is diagnostic only and is not applied at serving unless a current VAL refit improves VAL NLL.
  - Sub-30ms serving: 2.52ms latency on GPU; 26.67ms on CPU (edge-compatible).
- **Embedded Visual**: Precision-Recall curves (`reports/figures/pr_curve.png`) and Calibration curve (`reports/figures/calibration_curve.png`).

---

### Slide 6: Capture Bias Probe & Background Randomization
- **Headline**: Attacking Spurious Laboratory Background Memorization
- **Talking Points**:
  - Noyan (2022) background probe: Random Forest trained solely on 4 corner background pixels achieved 46.1% accuracy (chance is 2.6%).
  - Built automated foreground segmentation via Woebbecke Excess Green Index ($ExG = 2G - R - B$) with 100% clean extraction across 40 samples.
  - Conducted three controlled experiments:
    - **E0 (Baseline)**: Collapsed from 56.5% to 31.5% on background-swapped leaves (a 25% drop).
    - **E2 (Mixed 40/60)**: Achieved 66.5% clean and **60.5% background-swapped** (**91.0% retention** under complete background shifts!).
    - Center-masked spurious reliance slashed from 37% down to 25%.
- **Embedded Visual**: `reports/figures/bias_ablation_comparison.png` and `reports/figures/segmentation_40_grid.png`.

---

### Slide 7: Hierarchical Crop-Aware Inference & Solanaceae Disambiguation
- **Headline**: Resolving Cross-Genus Confusion via Hierarchical Selective Prediction
- **Talking Points**:
  - Solanaceae crops (Potato and Tomato) share visual morphology and common pathogens (*Alternaria*, *Phytophthora*).
  - Evaluated 8,134 test images under Auto vs Crop-Guided modes.
  - Crop-guided masked softmax eliminates cross-genus confusion, boosting accuracy to 99.32% and providing disambiguated in-crop probabilities.
- **Embedded Visual**: Solanaceae Confusion Matrix (`reports/figures/potato_tomato_confusion.png`).

---

### Slide 8: Layered OOD Detection: Heuristic Pre-Check + Energy Score
- **Headline**: Multi-Layer Input Validation Defeating Overconfidence on Noise & Grey Tiles
- **Talking Points**:
  - Softmax thresholding fails: a flat grey tile bypasses 0.50 threshold with 99.95% confidence!
  - **Layer 1 Heuristic Pre-Check**: Evaluates flatness (pixel std $<8.0$), decodability, and chlorophyll/lesion color balance in $<0.1$ms. Instantly intercepts grey tiles with code `NOT_A_LEAF_COLOR`.
  - **Layer 2 Helmholtz Free Energy Score**: $E(x) = -T \cdot \log \sum \exp(z_i/T)$. Out-of-distribution inputs yield high energy.
  - Achieves **0.9548 AUROC** with only **10.0% FPR95** (an 84% reduction in false positives).
- **Embedded Visual**: OOD ROC Curves (`reports/figures/ood_roc_curves.png`).

---

### Slide 9: Dual Explainability: Grad-CAM, SHAP & Deletion Curves
- **Headline**: Proving Saliency Faithfulness via Attribution Mass & Greedy Pixel Deletion
- **Talking Points**:
  - Implemented dual explainability: Hook-based Grad-CAM (layer4) and game-theoretic SHAP GradientExplainer.
  - Attribution Mass: 70.4% of Grad-CAM mass and 64.5% of SHAP mass falls strictly on the segmented leaf lamina.
  - Faithfulness Verification: Hooker et al. pixel deletion benchmark shows progressive Salient pixel masking collapses accuracy from 99.3% to 13.9% at 50% deletion, proving functional feature relevance rather than border edge detection.
- **Embedded Visual**: Deletion Curve (`reports/figures/explain_deletion_curve.png`) and side-by-side SHAP visual (`reports/figures/shap/`).

---

### Slide 10: Agronomic Advisory Engine v2 & Weather-Driven Risk
- **Headline**: Traceable Knowledge Base with Epidemiological Risk Rules
- **Talking Points**:
  - Schema v2 advisory engine citing verified extension guides (UC IPM, Cornell, Penn State, ICAR-CPRI, ICAR-IIHR, TNAU).
  - 6 foliar epidemiological risk rules in `kb/weather_rules.yaml` elevating clinical urgency based on temperature and humidity thresholds.
  - Strict Negative Constraints: Fungicides are barred for viral, greening, and mite conditions.
  - Regulatory Safety: Dosages are deliberately omitted and deferred to physical registered container labels.
- **Embedded Visual**: UI Advisory Card & Source Traceability Matrix (`docs/SOURCE_AUDIT.md`).

---

### Slide 11: Production Serving, Batch Latency & Automated Drift Triggers
- **Headline**: High-Throughput Serving & Closed-Loop Retraining Governance
- **Talking Points**:
  - Asynchronous FastAPI serving with Prometheus instrumentation and JSON lines audit logging.
  - Batch inference throughput: 110.5 fps single, **317.4 fps batch-64** on RTX 3050 GPU.
  - Two-sample Kolmogorov-Smirnov test and PSI monitoring on image blur, brightness, and confidence.
  - Automated retraining trigger: generates `reports/retraining_trigger.json` when statistical distribution shift occurs.
- **Embedded Visual**: Throughput scaling chart from `reports/INFERENCE_BENCHMARK.md`.

---

### Slide 12: Responsible AI, Structural Gap Audit & Viva Defense Summary
- **Headline**: Complete Transparency: A Robust, Defensible Engineering Pipeline
- **Talking Points**:
  - Transparent structural gap auditing in `kb/coverage.yaml`: Orange/Squash (no healthy class), Blueberry/Soybean (no disease class).
  - 4-Criterion promotion gate enforcing holistic pipeline health.
  - Zero-difference deterministic reproducibility check (0.00 logit diff).
  - All 24 automated unit/integration tests passing.
- **Embedded Visual**: 5-Tab Streamlit dashboard overview and coverage matrix.
