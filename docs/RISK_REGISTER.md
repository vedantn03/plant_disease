# MLOps Risk Register & Mitigation Audit

**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline
**Course**: B.Tech CSE-AI (Sponsored by Syngenta MLOps Evaluation Pilot)
**Governance Standard**: ISO/IEC 23894 (AI Risk Management) & Good MLOps Practice
**Date**: October 2026

---

## 1. Executive Summary

In agricultural AI, deploying computer vision directly into agronomic spray decisions carries severe risks: crop phytotoxicity, unnecessary chemical exposure, pathogen resistance, and financial loss. This Risk Register documents the six primary technical and regulatory failure modes identified during the development of this pipeline, the quantitative baseline vulnerability, the implemented architectural mitigations, and the residual risks.

---

## 2. Risk Traceability Matrix

| Risk ID | Category | Risk Description | Pre-Mitigation Likelihood / Severity | Baseline Vulnerability | Implemented Technical Mitigation | Post-Mitigation Risk | Verification Artifact |
| :---: | :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **R-01** | **Data / Inductive** | **Spurious Laboratory Background Correlation**:<br>Model learns uniform laboratory backdrop paper rather than foliar disease lesions. | **High / High** (Critical) | Noyan (2022) 4-corner pixel probe reached **46.1% accuracy** (chance: 2.6%). Baseline E0 collapsed from 56.5% to 31.5% under background shifts. | Automated $ExG$ leaf segmentation + E2 background-randomized composite training. | **Low / Med** (Managed) | `reports/BIAS_STUDY.md`<br>91.0% retention under background swaps. |
| **R-02** | **Inference / Safety** | **Overconfidence on Out-of-Distribution (OOD) Non-Leaf Inputs**:<br>Model generates high-confidence disease diagnoses for grey tiles, random noise, or human faces. | **High / High** (Critical) | Solid grey tile scored **99.95% confidence** under max-softmax thresholding. FPR95 was 63.3%. | Multi-layer rejection engine: Layer 1 optical heuristics (flatness/color std) + Layer 2 Helmholtz Free Energy score. | **Low / Low** (Resolved) | `reports/ood_report.md`<br>Grey tile rejected; AUROC: 0.9548, FPR95: 10.0%. |
| **R-03** | **Data / Domain** | **Structural Coverage Gaps in Training Data**:<br>Single-condition crops (Orange & Squash have no healthy class; Blueberry & Soybean have no disease class) cause false forced diagnoses. | **Med / High** (Significant) | A healthy Orange leaf was guaranteed to be misdiagnosed as diseased (Huanglongbing). | Machine-readable `kb/coverage.yaml` and UI Structural Gap Warning banners informing users of single-condition blindspots. | **Low / Med** (Documented) | `kb/coverage.yaml`<br>Tab 4 Coverage audit table. |
| **R-04** | **Model / Taxonomy** | **Cross-Genus Solanaceae Confusion**:<br>Potato and Tomato share visual morphology and common pathogens (*Alternaria solani*, *Phytophthora infestans*), causing misattribution. | **Med / Med** (Moderate) | Cross-genus error rate observed under unguided global 38-class softmax. | Hierarchical crop-guided inference mode (`crop: Optional[str]`) using masked softmax renormalization. | **Low / Low** (Resolved) | `reports/hierarchy_report.md`<br>Zero cross-genus error in crop-guided mode. |
| **R-05** | **Agronomic / Regulatory** | **Chemical Hallucination & Off-Label Recommendation**:<br>Software generates fabricated active ingredient dosages or recommends fungicides for viral / non-fungal conditions. | **Low / Critical** (Catastrophic) | Unvalidated LLM or heuristic generation could advise fungicides on viral mosaic, causing wasted expenditure and environmental toxicity. | Schema v2 audited knowledge base (`kb/advisories.yaml`) citing university extensions; negative constraints strictly barring fungicides for viral/mites; zero-dosage invention policy. | **Low / Low** (Enforced) | `docs/SOURCE_AUDIT.md`<br>`tests/test_advisory_kb.py` passing all constraints. |
| **R-06** | **Ops / Drift** | **In-Field Camera Degradation & Silent Model Failure**:<br>Dirty lenses, camera sensor blur, and low-light field conditions can shift input features. | **High / Med** (Significant) | Live production accuracy is not measured by this local demo. | API request logs plus a provisional KS/PSI exporter and Prometheus alert; replay is staged traffic. Legacy synthetic drift/trigger outputs are SIMULATION ONLY; no automated retraining. | **Open** | `reports/drift_latest.json`; legacy `reports/drift_report.md` and `reports/retraining_trigger.json` are SIMULATION outputs. |

---

## 3. Residual Risk Disclosures for Viva Defense

1. **Residual Field Gap (R-01)**: While background randomization (E2) preserved 91.0% accuracy under laboratory background shifts, PlantVillage leaves were detached and photographed flat indoors. Real-world outdoor images introduce variable leaf angle, overlapping foliage, direct sunlight glare, and wind motion blur.
2. **Hard Negative Plant Leaves (R-02)**: Layer 1 and Layer 2 reject non-vegetative objects and noise reliably. However, a healthy leaf of an unrepresented crop species (e.g. Mango or Rice) may still produce a plausible green color ratio and register moderate energy scores. Field deployment requires training on an explicitly collected negative flora dataset.
3. **Regulatory Disclaimer (R-05)**: This pipeline is an engineering research prototype. Under no circumstances should agricultural chemical spray volumes be determined by automated software without physical inspection of the EPA / Central Insecticides Board product label and consultation with a certified crop advisor.
