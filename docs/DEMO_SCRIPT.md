# 10-Minute Viva & Evaluation Presentation Script

**Project**: Automated Crop Leaf Disease Identification and Audited Advisory Pipeline
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
| **00:00 – 01:15**<br>(75s) | **Terminal 2** | `..\.venv\Scripts\pytest.exe tests/ -v` | **Data Integrity**:<br>*"Use the current local pytest output and saved audit artifacts for numerical claims. The split uses crop-scoped group IDs; report exact counts only when the current audit report confirms them."* |
| **01:15 – 02:45**<br>(90s) | **Streamlit UI**:<br>*Diagnostic Screening* | Select a committed demo image and run diagnosis | **Inference and Advisory**:<br>*"Describe the prediction and advisory shown by the running application. Quote numerical confidence or latency only when supported by the current saved report."* |
| **02:45 – 03:45**<br>(60s) | **Streamlit UI**:<br>*Diagnostic Screening* | Click: **Explain with SHAP** when enabled | **Explainability**:<br>*"Show the explanation outputs. Treat quantitative claims as unverified unless the corresponding current report file is available."* |
| **03:45 – 04:45**<br>(60s) | **Streamlit UI**:<br>*Diagnostic Screening* | Use the OOD example only if locally available | **Input checks**:<br>*"Demonstrate the visible rejection behavior. Numeric benchmark claims remain unverified unless the current OOD report is checked."* |
| **04:45 – 05:45**<br>(60s) | **Streamlit UI**:<br>*Diagnostic Screening* | Compare automatic and crop-guided inference | **Crop guidance**:<br>*"Explain the behavior visible in the UI without quoting unverified accuracy or confidence figures."* |
| **05:45 – 06:45**<br>(60s) | **Streamlit UI**:<br>*MLOps System* | Review available evidence | **Model evidence**:<br>*"Describe the displayed evidence and identify the report used for each number. Treat figures without a saved source report as unverified."* |
| **06:45 – 07:30**<br>(45s) | **Streamlit UI** | Use the three available tabs: Diagnostic Screening, Production Telemetry, MLOps System | **Monitoring and governance**:<br>*"Airflow and drift-trigger outputs are SIMULATION. The dashboard has three tabs; distinguish local simulation from live production services."* |
| **07:30 – 08:00**<br>(30s) | **Concluding Slide** | Review Summary Slide | **Conclusion**:<br>*"Summarize implemented components and clearly label simulations. Include metrics only when verified in the current report artifacts."* |

---

## 3. Evaluator Defense Strategy: 3 Golden Rules

1. **When asked if this model is ready for commercial farm release**:
   - *Direct Response*: *"No. PlantVillage was collected under controlled indoor lighting with uniform backdrop paper. While our E2 model demonstrates 91% retention against background shifts, true field deployment requires outdoor training data with complex canopy occlusion, insect damage, and variable solar zenith angles. This pipeline is an engineering prototype designed to evaluate the MLOps lifecycle."*
2. **When asked how chemical dosages were calculated**:
   - *Direct Response*: *"Dosages were deliberately not calculated. Fabricating chemical doses in software creates severe legal liability and risk of crop phytotoxicity. All recommendations derive from our audited knowledge base citing university extension publications (UC IPM, Cornell, ICAR), and farmers are strictly instructed to consult the physical container label for approved dosage rates and pre-harvest intervals."*
3. **When asked why ResNet-18 was chosen over a Vision Transformer**:
   - *Direct Response*: *"ResNet-18 delivers the optimal trade-off between architectural inductive bias, gradient stability, parameter size (11.2M parameters, 128 MB checkpoint), and edge inference speed (26.7 ms on CPU). A ViT requires order-of-magnitude more compute and memory without providing superior localized spatial features on single-leaf images."*
