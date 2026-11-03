# Comprehensive Viva & Evaluator Defense Guide (30 Questions)

**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline
**Role**: Lead Machine Learning Engineer / Student Presenter
**Audience**: Syngenta Evaluators & University Examination Committee

---

### Category 1: Data Engineering & Split Integrity

#### Q1: What is data leakage, and how did you mathematically guarantee zero leakage in this dataset?
**Answer**: Data leakage occurs when test samples share identity or capture artifacts with the training set. In PlantVillage, individual leaves were photographed multiple times from varying angles, sharing identical UUID stems (e.g. `GH_HL Leaf 173.1`, `173.2`). A naive random split places views of the identical leaf into both train and test splits, causing massive metric inflation. We extracted the root capture token, scoped it by crop species (`<crop>::<token>`), and performed a **group-aware stratified split** (Train: 38,002, Val: 8,148, Test: 8,134). Our unit tests mathematically verify that `len(set(train_groups) & set(test_groups)) == 0`.

#### Q2: Why did you have to prefix the capture token with the crop name (`<crop>::<id>`)?
**Answer**: During our initial SHA-256 and regex audit, we discovered that independent laboratory photographers reused generic sequence identifiers across distinct crops—for instance, `RS_HL 5672` existed in both `Apple___healthy` and `Soybean___healthy`. Grouping globally by token alone created false cross-crop collisions. Prefixing with `<crop>::` accurately isolates physical plant capture batches while allowing clean stratification.

#### Q3: How did you handle the severe class imbalance (36:1 ratio across 38 classes)?
**Answer**: PlantVillage varies from 5,507 images in the majority class (`Orange___Haunglongbing`) down to 152 images in the minority class (`Potato___healthy`). Rather than using a weighted random sampler—which distorts epoch lengths and repeatedly exposes the model to identical minority images—we utilized **class-weighted cross-entropy loss**, weighting each class inversely proportional to its training frequency: $w_c = \frac{N}{C \cdot N_c}$.

---

### Category 2: Model Architecture & Training Regime

#### Q4: Why select ResNet-18 over heavier architectures like ResNet-50 or Vision Transformers?
**Answer**: ResNet-18 contains 11.2 million parameters (128 MB checkpoint) and provides strong visual inductive bias through residual skip connections. In our inference latency benchmarks on host hardware, ResNet-18 achieves a median latency of **2.52 ms on an RTX 3050 GPU** and **26.67 ms on CPU**. For an agricultural edge device (e.g., a tractor boom or handheld scout device), a 26 ms CPU runtime permits real-time edge processing without expensive field cloud infrastructure.

#### Q5: What was the exact training regime used for the champion model?
**Answer**: The champion baseline was trained on the full 38,002-image training split using **Automatic Mixed Precision (AMP FP16)** on an NVIDIA RTX 3050 Laptop GPU for 8 epochs with an Adam optimizer ($lr=0.001$, weight decay $10^{-4}$), Cosine Annealing learning rate schedule, and class-weighted cross-entropy loss.

#### Q6: What are the exact champion metrics and their confidence intervals?
**Answer**: Evaluated across 8,134 held-out test images with 1,000 non-parametric bootstrap iterations:
- Test accuracy and Macro-F1: use the saved `reports/metrics.json` values; cite confidence intervals only from that report.
- **Expected Calibration Error (ECE)**: use the reproducible unscaled VAL/TEST artifact values; temperature scaling is diagnostic only and is not applied at serving unless a current VAL refit improves VAL NLL.

---

### Category 3: Capture Bias & Leaf Segmentation

#### Q7: What is the Noyan (2022) capture-bias probe, and what did our initial baseline show?
**Answer**: Noyan demonstrated that classifiers trained on PlantVillage can learn the uniform indoor laboratory background paper instead of foliar pathology. When we trained a Random Forest probe exclusively on **4 corner background pixels (12 RGB values)**, it achieved **46.1% test accuracy** (chance is 2.6%; majority baseline is 10.2%). This confirmed that laboratory background artifacts contain substantial predictive signal.

#### Q8: How did you implement automated leaf segmentation?
**Answer**: In `src/segmentation.py`, we implemented the Woebbecke Excess Green Index:
$$ExG = 2G - R - B$$
Combined with saturation thresholding in HSV color space, binary morphological opening (`scipy.ndimage`), and largest connected component extraction, it isolates the green/chlorotic leaf lamina. Benchmark result: **unverified here**; check the current segmentation report.

#### Q9: What were the results of the background-randomization experiments (E0, E1, E2)?
**Answer**: See the saved `reports/BIAS_STUDY.md` for experiment values and conditions. Quote results
with the report's split and setup; do not generalize them beyond that experiment.

---

### Category 4: Selective Prediction & Out-of-Distribution Detection

#### Q10: Why is softmax thresholding flawed for rejecting non-leaf images?
**Answer**: Softmax forces probabilities to sum to 1, so confidence alone is not a reliable OOD check. Quote specific grey-tile confidence only when supported by a current saved report; otherwise mark it unverified.

#### Q11: How does your Layer 1 Heuristic Pre-Check work?
**Answer**: Layer 1 executes in $<0.1$ ms before any neural inference:
1. Validates minimum dimensions ($64 \times 64$).
2. Checks pixel standard deviation ($\ge 8.0$) to reject flat or blank tiles.
3. Computes chlorophyll ($G > B$) and necrotic/lesion ($R > B$) vegetative pixel ratios.
Inputs failing these checks are rejected immediately with status `rejected_not_a_leaf` and reason codes `FLAT_IMAGE` or `NOT_A_LEAF_COLOR`.

#### Q12: What is the Helmholtz Free Energy Score, and why is it superior?
**Answer**: Free Energy Score is defined as:
$$E(x; T) = -T \cdot \log \sum_{i=1}^C \exp\left(\frac{z_i}{T}\right)$$
Unlike softmax, Energy does not constrain outputs to sum to 1. It is mathematically proportional to the negative log marginal likelihood of the input. In-distribution leaf images produce large negative energy scores ($E \le -12.44$), whereas non-leaf noise and out-of-distribution inputs yield higher energy values.

#### Q13: What are the empirical OOD detection benchmark numbers?
**Answer**: The saved `reports/ood_metrics.json` reports max-softmax AUROC 0.8369/FPR95 0.6333,
energy AUROC 0.8831/FPR95 0.6167, and layered prefilter+energy AUROC 0.9548/FPR95 0.1. These
are report-specific benchmark results, not a claim of field performance.

---

### Category 5: Hierarchical Inference & Structural Gaps

#### Q14: How does the pipeline resolve the Potato vs Tomato confusion?
**Answer**: Potato (*Solanum tuberosum*) and Tomato (*Solanum lycopersicum*) belong to the same genus and share identical pathogens (e.g. *Alternaria solani*, *Phytophthora infestans*). Under global 38-class inference, visual similarity creates cross-genus confusion. Our API supports an optional `crop` parameter that applies **masked softmax renormalization**, restricting prediction strictly to candidate conditions within the specified crop.

#### Q15: What are the structural gaps in PlantVillage, and how does the system handle them?
**Answer**:
- **Disease-Only Gaps**: Orange and Squash contain only diseased classes; healthy leaves of these crops will be falsely diagnosed as diseased.
- **Healthy-Only Gaps**: Blueberry, Raspberry, and Soybean contain only `healthy` classes; diseased leaves will be falsely classified as healthy.
We codified these in `kb/coverage.yaml`. The API and UI explicitly display a **Structural Gap Notice**, warning the user of conditions the model cannot diagnose.

---

### Category 6: Explainability & Faithfulness

#### Q16: How does Grad-CAM work, and where is it hooked?
**Answer**: Grad-CAM computes the gradient of winning class score $y^c$ with respect to feature activation maps $A^k$ of `model.layer4`:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
The resulting $7 \times 7$ activation map is bilinearly upsampled to $224 \times 224$ and overlaid as a heatmap.

#### Q17: What does SHAP provide beyond Grad-CAM?
**Answer**: Grad-CAM captures coarse activation localization from the final convolutional layer. SHAP (`shap.GradientExplainer`) calculates game-theoretic Shapley values, distributing marginal feature importance directly across input RGB pixels. It highlights exact lesion boundaries and petiole structures.

#### Q18: What is Attribution Mass, and what did you measure?
**Answer**: Attribution mass quantifies the percentage of saliency falling within the segmented leaf mask vs the background:
$$\text{Attribution Mass} = \frac{\sum_{(x,y) \in \text{Leaf}} M_{x,y}}{\sum_{(x,y) \in \text{Image}} M_{x,y}}$$
Across 100 test images:
- **Grad-CAM**: **70.4%** of attribution falls on the leaf (29.6% on background).
- **SHAP**: **64.5%** of attribution falls on the leaf (35.5% on background).

#### Q19: How did you prove your explainers are faithful and not edge detectors?
**Answer**: We executed the Hooker et al. **greedy pixel deletion benchmark** on 100 test leaves. Progressively replacing the top salient pixels with black pixels:
- **0% deleted**: 99.3% accuracy.
- **10% deleted**: drops to 90.6%.
- **50% deleted**: collapses to **13.9%**.
This steep collapse proves the explainers identify true functional decision features rather than decorative edges.

---

### Category 7: Advisory Engine & Agronomic Safety

#### Q20: What is Schema v2 in `kb/advisories.yaml`?
**Answer**: Schema v2 structures agronomic advice into `region`, `recommendation_scope`, `cultural_controls`, `chemical_options` (with mode of action and PPE), and verifiable `sources`. It incorporates both global extension guides (UC IPM, Cornell, Penn State) and Indian national agricultural institutes (ICAR-CPRI, ICAR-IIHR, ICAR-NRCG, TNAU).

#### Q21: How do weather decision rules function?
**Answer**: `kb/weather_rules.yaml` encodes 6 epidemiological forecasting models (e.g., Wallin Late Blight units, FAST Early Blight criteria, Mills Apple Scab periods). When ambient temperature and relative humidity exceed sporulation thresholds, the advisory engine elevates the urgency level and appends an epidemiological warning note.

#### Q22: Why does the system strictly prohibit dose modification via weather?
**Answer**: Changing chemical application rates (e.g. increasing fungicide concentration during high humidity) violates federal pesticide regulations (FIFRA in the US, Insecticides Act in India). Off-label dose modification causes crop phytotoxicity, toxic environmental runoff, and rapid fungicide resistance.

#### Q23: Why are fungicides forbidden for viral, greening, and mite conditions?
**Answer**:
1. **Viruses**: Intracellular RNA pathogens transmitted by insects; fungicides have zero effect.
2. **Citrus Greening**: Phloem-limited fastidious bacteria; require psyllid vector control and quarantine rogueing.
3. **Spider Mites**: Arachnid pests; broad-spectrum fungicides kill entomopathogenic fungi that naturally keep mite populations in check.

#### Q24: Why does the software omit specific dosage numbers?
**Answer**: Legal and agronomic safety. Dosage depends on crop canopy volume, spray equipment calibration, formulation concentration (EC, WP, SC), and regional label registrations. The interface explicitly directs the applicator to the physical registered container label.

---

### Category 8: Production MLOps, Drift & Governance

#### Q25: How does the automated 4-criterion promotion gate work?
**Answer**: `scripts/promote.py` evaluates VAL within-crop macro-F1, minimum per-class recall,
champion-relative masked-background drop, and ECE. Thresholds remain provisional; dry-run is the
current workflow and `--apply` is refused while criteria are provisional. TEST metrics are for
reporting only and do not drive candidate selection.

#### Q26: What throughput does the batch inference engine achieve?
**Answer**: The saved `reports/INFERENCE_BENCHMARK.md` reports 110.5, 263.3, and 317.4 images/sec
for batch sizes 1, 16, and 64 respectively, under its stated benchmark setup.

#### Q27: How is data drift detected in production?
**Answer**: The real-log path is `scripts/drift_exporter.py`: it compares a sliding window of API request-log features with a TRAIN-split baseline. `scripts/simulate_drift.py` remains SIMULATION ONLY. Replay is staged traffic, not real users, and threshold choices are provisional.

#### Q28: How does the automated retraining trigger work?
**Answer**: `scripts/check_drift_and_trigger.py` is SIMULATION ONLY and will not create synthetic inputs or trigger retraining. Real-log monitoring exposes a Prometheus alert; no automated retraining is connected.

#### Q29: How did you verify deterministic reproducibility?
**Answer**: In `scripts/reproduce_check.py`, we run dual forward passes with identical seeds and weights across a batch of images. The script asserts:
- Class prediction agreement and maximum logit difference: **unverified here**; check the current reproducibility report.

#### Q30: What is the commercial positioning of this project?
**Answer**: This project is an **academic prototype of the engineering pipeline behind a crop diagnostic advisory**, built to demonstrate end-to-end MLOps best practices (data grouping, bias mitigation, layered OOD, dual explainability, audited advisories, drift triggers). It is **not field-ready software** and does not claim commercial equivalence.
