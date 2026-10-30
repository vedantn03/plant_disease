# University MLOps Course Syllabus Mapping

**Course**: B.Tech Computer Science & Engineering (Specialization in AI & Machine Learning)  
**Subject**: Machine Learning Operations (MLOps)  
**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline  
**Evaluation Partner**: Syngenta Professional & Academic Faculty Review

---

## 1. Syllabus Topic-to-Artifact Traceability Matrix

This document maps the core academic learning objectives of an advanced university MLOps curriculum directly to the production-grade implementations and verifiable artifacts delivered in this repository.

| Module | Core MLOps Competency | Implemented Component in Pipeline | Concrete Code / Script Path | Verifiable Deliverable & Evidence |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Data Engineering & Versioning** | SHA-256 deduplication, multi-shot capture grouping `<crop>::<id>`, stratified train/val/test splitting without leakage. | `scripts/audit_and_split.py`<br>`dvc.yaml` | `reports/audit_report.md`<br>`tests/test_splits.py` (Asserts zero group overlap). |
| **2** | **Experiment Management & Tracking** | Automatic Mixed Precision (AMP FP16) training, cosine annealing learning rate scheduling, class-weighted cross-entropy for 36:1 imbalance. | `scripts/train.py`<br>`configs/config.yaml` | MLflow tracking (`mlruns.db`), per-epoch train/val loss, Macro-F1 curves, and system metrics. |
| **3** | **Rigorous Model Evaluation & Bias** | Non-parametric bootstrap confidence intervals (1,000 iterations), Expected Calibration Error (ECE), temperature scaling, Noyan background capture-bias probe. | `scripts/evaluate.py`<br>`src/segmentation.py` | `reports/metrics.json`<br>`reports/figures/bias_probe_accuracy.png`<br>`reports/calibration_curve.png`. |
| **4** | **Capture Bias Mitigation & Robustness** | Green-index leaf segmentation ($ExG$), foreground masking, background-randomized training (E0 vs E1 vs E2), background-swapped stress tests. | `src/segmentation.py`<br>`scripts/run_bias_experiments.py` | `reports/BIAS_STUDY.md`<br>`reports/figures/segmentation_40_grid.png`<br>`reports/bias_study_metrics.json` (91.0% retention). |
| **5** | **Selective Prediction & OOD Detection** | Multi-layered rejection engine: Layer 1 optical heuristics (flatness/color) + Layer 2 Helmholtz free energy score ($E(x)$) rejecting non-leaf noise and grey tiles. | `src/ood_detector.py`<br>`scripts/evaluate_ood.py` | `reports/ood_report.md`<br>`reports/figures/ood_roc_curves.png` (AUROC: 0.9548, FPR95: 10.0%). |
| **6** | **Hierarchical Domain-Aware Inference** | Solanaceae genus confusion resolution; crop-guided masked softmax inference; structural gap detection for single-condition crops. | `src/api.py`<br>`kb/coverage.yaml`<br>`scripts/evaluate_hierarchy.py` | `reports/hierarchy_report.md`<br>`reports/figures/potato_tomato_confusion.png`. |
| **7** | **Explainability & Attribution Faithfulness** | Hook-based Grad-CAM (layer4), SHAP GradientExplainer, attribution mass quantification on leaf mask, greedy pixel deletion curves. | `src/model.py`<br>`scripts/run_explainability.py` | `reports/EXPLAINABILITY.md`<br>`reports/figures/explain_deletion_curve.png`<br>`reports/figures/shap/`. |
| **8** | **Knowledge Engineering & Regulatory Safety** | Traceable Schema v2 advisory engine citing university extensions (UC IPM, Cornell, Penn State, ICAR, TNAU), epidemiological weather rules, zero-dosage invention policy. | `src/advisory_engine.py`<br>`kb/advisories.yaml`<br>`kb/weather_rules.yaml` | `docs/SOURCE_AUDIT.md`<br>`tests/test_advisory_kb.py` (Negative constraints for viral/greening/mites). |
| **9** | **Continuous Deployment & Model Registry** | 4-criterion mathematical promotion gate (Macro-F1, recall floor, OOD AUROC, background retention), model registry alias tagging. | `scripts/promote.py` | `reports/promotion_log.json`<br>MLflow Model Registry `champion` tag. |
| **10** | **Serving, Edge Latency & Batch Infer** | High-throughput FastAPI asynchronous serving, tensorized mini-batch inference engine (110.5 fps single, 317.4 fps batch-64). | `src/api.py`<br>`scripts/batch_infer.py` | `reports/INFERENCE_BENCHMARK.md`<br>`reports/batch_predictions.csv`<br>`docker-compose.yml`. |
| **11** | **Production Telemetry & Drift Monitoring** | Online JSON lines query logging, two-sample Kolmogorov-Smirnov test, Population Stability Index (PSI) shift detection, automated retraining trigger. | `scripts/simulate_drift.py`<br>`scripts/check_drift_and_trigger.py` | `reports/drift_report.md`<br>`reports/retraining_trigger.json`<br>`logs/requests.jsonl`. |
| **12** | **MLOps Governance, Ethics & Reproducibility** | Gebru datasheet, Risk register, deterministic reproducibility check with zero logit difference, 5-tab Streamlit governance UI. | `scripts/reproduce_check.py`<br>`app/ui.py`<br>`docs/MODEL_CARD.md` | `docs/DATASHEET.md`<br>`docs/RISK_REGISTER.md`<br>`tests/` (24 passing unit/integration tests). |

---

## 2. Examination Viva Talking Points by Syllabus Module

1. **Why not use standard K-fold cross-validation?**
   - *Defense*: In medical and agricultural datasets with multi-shot burst photography, standard K-fold or random splitting causes massive data leakage. The model memorizes leaf surface idiosyncrasies rather than pathology. We designed a group-aware split (`<crop>::<id>`) guaranteeing zero patient/leaf overlap across train and test.
2. **Why Helmholtz Free Energy over Softmax Thresholding for OOD Detection?**
   - *Defense*: Softmax probabilities suffer from the normalization constraint ($\sum p_i = 1$), which artificially forces high confidence on uniform noise or out-of-distribution inputs (e.g., a grey tile achieved 99.95% confidence under max-softmax). Free energy score $E(x) = -T \log \sum \exp(z_i/T)$ is mathematically aligned with the data log-likelihood without normalization distortion, reducing False Positive Rate at 95% TPR from 63.3% down to 10.0%.
3. **How do you verify explainability maps are faithful and not edge detectors?**
   - *Defense*: We implemented the Hooker et al. pixel deletion benchmark. Progressively masking out the top 50% most salient pixels identified by Grad-CAM and SHAP collapsed model prediction confidence from 99.3% down to 13.9%, proving the heatmaps trace true functional decision features rather than passive high-frequency edges.
4. **Why is your model promotion gate multi-objective?**
   - *Defense*: In real-world MLOps, deploying a model based solely on clean validation accuracy leads to catastrophic field failures. Our promotion gate enforces 4 strict gates: overall Macro-F1 delta ($\ge -0.005$), per-class recall floor ($\ge 0.85$), OOD discrimination capability ($\text{AUROC} \ge 0.85$), and background swap retention ($\ge 75\%$).
