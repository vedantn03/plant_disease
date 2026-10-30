# 10-Minute Viva & Evaluation Presentation Script

**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline  
**Role**: Lead Machine Learning Engineer / Student Presenter  
**Audience**: Syngenta Evaluators & University Examination Committee  
**Total Allocated Time**: 10 Minutes (8 min presentation & demo + 2 min defense Q&A)

---

## 1. Pre-Demo Setup Checklist (1 Minute Before Start)

Open two terminal windows in `plant_disease/`:

1. **Terminal 1 (Backend Serving)**:
   ```powershell
   cd "d:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
   ..\.venv\Scripts\uvicorn.exe src.api:app --host 127.0.0.1 --port 8000
   ```
2. **Terminal 2 (Frontend Dashboard)**:
   ```powershell
   cd "d:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
   ..\.venv\Scripts\streamlit.exe run app/ui.py --server.port 8501
   ```
3. Open browser at: `http://localhost:8501`

---

## 2. Minute-by-Minute Demo Script

| Timestamp | Screen / Visual | Action / Command | Key Talking Points & Exact Empirical Metrics |
| :---: | :--- | :--- | :--- |
| **00:00 – 01:15**<br>(75s) | **Terminal 2** | `..\.venv\Scripts\pytest.exe tests/ -v` | **Data Integrity & Zero Group Leakage**:<br>*"We begin by verifying system test integrity: all 24 unit and integration tests pass in 0.8s. In PlantVillage (54,305 raw images), leaves share capture stems across angles. A naive random split creates massive data leakage. We scoped tokens as `<crop>::<id>` and partitioned a 70/15/15 group-aware split (Train: 38,002, Val: 8,148, Test: 8,134) with zero group overlap mathematically asserted."* |
| **01:15 – 02:45**<br>(90s) | **Streamlit UI**:<br>*Tab 1 (Diagnostic Screening)* | Select: **`Tomato___Early_blight.jpg`**<br>Crop Selector: `Tomato`<br>Temp: `26.0°C`, RH: `80.0%`<br>Click: **Run Disease Diagnosis** | **Inference, Grad-CAM & Audited Advisory**:<br>*"The model diagnoses Tomato Early Blight with 99.8% confidence in 2.5ms on our host RTX 3050 GPU (26.7ms on CPU). Notice the Grad-CAM activation heatmap—it precisely localizes the concentric target-like necrotic lesions on the leaf lamina. Below, our Schema v2 advisory displays verified cultural practices from UC IPM and ICAR-IIHR, notes that weather conditions exceed the sporulation threshold (24-30°C, RH >75%), and complies with regulatory policy by directing the farmer to the registered container label rather than fabricating chemical doses."* |
| **02:45 – 03:45**<br>(60s) | **Streamlit UI**:<br>*Tab 1 (Diagnostic Screening)* | Click: **Explain with SHAP** button | **Deep Explainability & Deletion Faithfulness**:<br>*"Unlike simple edge detectors, our pipeline provides dual explainability via SHAP GradientExplainer. Quantitative analysis across 100 test leaves shows that 70.4% of Grad-CAM mass and 64.5% of SHAP mass falls strictly inside the segmented leaf lamina. Furthermore, our pixel deletion benchmark demonstrates that progressively removing the top 50% salient pixels collapses model confidence from 99.3% to 13.9%, proving functional feature faithfulness."* |
| **03:45 – 04:45**<br>(60s) | **Streamlit UI**:<br>*Tab 1 (Diagnostic Screening)* | Check: **Test Out-of-Distribution Rejection (Grey Tile)**<br>Click: **Run Disease Diagnosis** | **Layered Selective Prediction (Attacking Overconfidence)**:<br>*"Standard deep classifiers are dangerously overconfident on non-leaf images—under max-softmax, this flat grey tile received 99.95% confidence. Here, our multi-layered rejection engine instantly intercepts it at Layer 1 (`rejected_not_a_leaf`, code: `NOT_A_LEAF_COLOR`). Together with our Layer 2 Free Energy detector ($E(x) = -T \log \sum e^{z/T}$), we achieve an OOD AUROC of 0.9548 and reduce False Positive Rate at 95% TPR from 63.3% down to 10.0%."* |
| **04:45 – 05:45**<br>(60s) | **Streamlit UI**:<br>*Tab 1 (Diagnostic Screening)* | Select: **`Potato___Early_blight.jpg`**<br>Toggle Crop: `Auto` vs `Potato` | **Hierarchical Crop-Aware Inference**:<br>*"Potato and Tomato belong to the Solanaceae family and share common pathogens like Alternaria solani. Under Auto mode, minor cross-genus confusion occurs. By selecting Potato, the API applies masked softmax renormalization, restricting probabilities to valid species conditions and boosting prediction confidence to 99.9%."* |
| **05:45 – 06:45**<br>(60s) | **Streamlit UI**:<br>*Tab 3 (Model Evidence & Bias Study)* | Scroll to **Bias Study Table & Charts** | **Capture Bias Mitigation & Background Robustness**:<br>*"Honesty in AI is paramount. Noyan (2022) revealed that models can memorize laboratory paper backgrounds. Our E0 baseline collapsed from 56.5% clean accuracy to 31.5% when backgrounds were swapped. By introducing background-randomized composite training in E2, our model maintains 60.5% accuracy under complete background swaps—achieving a 91.0% retention rate while reducing spurious border pixel reliance from 37% down to 25%."* |
| **06:45 – 07:30**<br>(45s) | **Streamlit UI**:<br>*Tab 4 & Tab 5* | Switch to **Tab 4 (Coverage)** then **Tab 5 (Governance)** | **Structural Gaps, Drift & Retraining Triggers**:<br>*"Tab 4 transparently flags structural dataset gaps: Orange and Squash have no healthy classes in PlantVillage, while Blueberry and Soybean have no disease classes. Tab 5 displays our production governance: live request telemetry, automated drift detection using two-sample KS and PSI tests, and an automated retraining trigger manifest generated when drift exceeds statistical thresholds."* |
| **07:30 – 08:00**<br>(30s) | **Concluding Slide** | Review Summary Slide | **Conclusion**:<br>*"In conclusion: we have delivered not a naive metric-chaser, but an evaluation-grade MLOps engineering pipeline featuring zero-leakage splits, sub-30ms edge latency, layered OOD rejection (0.9548 AUROC), background robustness (91% retention), dual explainability, audited agronomic guidance, and automated drift triggers."* |

---

## 3. Evaluator Defense Strategy: 3 Golden Rules

1. **When asked if this model is ready for commercial farm release**:
   - *Direct Response*: *"No. PlantVillage was collected under controlled indoor lighting with uniform backdrop paper. While our E2 model demonstrates 91% retention against background shifts, true field deployment requires outdoor training data with complex canopy occlusion, insect damage, and variable solar zenith angles. This pipeline is an engineering prototype designed to evaluate the MLOps lifecycle."*
2. **When asked how chemical dosages were calculated**:
   - *Direct Response*: *"Dosages were deliberately not calculated. Fabricating chemical doses in software creates severe legal liability and risk of crop phytotoxicity. All recommendations derive from our audited knowledge base citing university extension publications (UC IPM, Cornell, ICAR), and farmers are strictly instructed to consult the physical container label for approved dosage rates and pre-harvest intervals."*
3. **When asked why ResNet-18 was chosen over a Vision Transformer**:
   - *Direct Response*: *"ResNet-18 delivers the optimal trade-off between architectural inductive bias, gradient stability, parameter size (11.2M parameters, 128 MB checkpoint), and edge inference speed (26.7 ms on CPU). A ViT requires order-of-magnitude more compute and memory without providing superior localized spatial features on single-leaf images."*
