# Architectural Decisions

All decisions are recorded here. Format: `[YYYY-MM-DD] Decision | Reason`

---

[2026-10-14] Build inside `plant_disease/` subfolder of existing repo | The workspace already contains an airfare project; isolating in a subfolder avoids conflicts without creating a new git repo.

[2026-10-14] Use Python 3.14 + torch 2.11.0+cu128 | Only Python 3.14 is installed; PyTorch 2.11.0+cu128 (cp314 wheel) is available from pytorch.org/whl/cu128. CUDA 13.0 driver supports CUDA 12.8 runtime backward-compat.

[2026-10-14] Use existing `.venv` in project root | A venv already exists with pip 26.x; reusing avoids redundant env setup and PATH confusion.

[2026-10-14] Use `weighted_loss` (not weighted sampler) for imbalance | Simpler to implement, same theoretical justification, and avoids epoch-length distortion from over-sampling minority classes. Noted in config as a swap-able option.

[2026-10-14] Load model from local exported file in FastAPI (not MLflow registry client) | MLflow registry `pyfunc.load_model` has version-specific API surface that can break between releases; exporting to a .pth and loading with `torch.load` is simpler, more reliable for a demo, and easier to explain in a viva.

[2026-10-14] Skip Evidently for drift demo; use scipy + PSI | Evidently's API has changed significantly across versions and install can be flaky. scipy is already a dependency; PSI is < 20 lines of code.

[2026-10-14] MLflow SQLite backend (not file backend) | SQLite gives a persistent, queryable store and works identically across restarts. File backend is fragile on Windows due to path encoding.

[2026-10-14] PlantVillage dataset at `D:\VIT\SEM_5\MLOps\CP\archive\plantvillage dataset\color` | Found by filesystem search; stored only in .env (DATA_ROOT). Never hardcoded.

[2026-10-14] Scope group_id by crop (`<crop>::<source_code> <id>`) | Empirical audit revealed capture sequence IDs like 'RS_HL 5672' are reused across independent crop folders (e.g. Apple and Soybean); scoping to crop reflects true biological/capture grouping and enables 0 cross-split leakage.

[2026-10-14] Full-data retraining (38,002 images, 8 epochs) replacing subsample baseline | User confirmed full PlantVillage dataset accessible; subsample (300/class, ~10,967 images) gave 95.63% macro-F1 which was replaced by 99.03% full-data model. All docs updated to reflect full-data training.

[2026-10-28] Inference preprocessing changed from Resize(224,224) to Resize(256)+CenterCrop(224) | Direct square resize distorts aspect ratio of non-square real-world photos; standard ImageNet pipeline uses Resize(256)+CenterCrop(224). Applied to val/test and inference consistently.

[2026-10-28] OOD rejection rate corrected from "~2%" to "0%" in STATE_VERIFICATION.md | Measured value from reports/metrics.json is rejection_rate_at_0_50 = 0.0 (zero rejections in 200 OOD samples). Earlier "2%" was misread. Honest reporting of the more severe result.

[2026-10-28] Centre-masked CNN result corrected from "61.0%" to "68.67%" | Measured value from reports/metrics.json capture_bias_probe.center_masked_cnn_acc = 0.6867. The more severe (68.67%) result is correct and reported honestly.

[2026-10-28] Docker Compose updated with UI service and multi-target Dockerfile | docker-compose.yml previously only had api service; ui service added with depends_on api health check. Dockerfile rewritten with base/api/ui multi-stage targets.

[2026-10-28] make up target added | Makefile previously had separate 'serve' and 'ui' targets but no single-command stack. 'make up' now calls docker compose up --build -d.

[2026-10-28] UI backend-offline state improved | Previously showed generic error string and continued rendering tabs. Now shows actionable restart commands and calls st.stop() to halt rendering, preventing misleading empty tabs.

[2026-10-28] Automated leaf foreground extraction via Woebbecke Excess Green Index ($ExG$) | ExG = 2G - R - B combined with HSV saturation thresholding and largest connected component analysis cleanly segments leaves from background paper with 0% failure across 40 benchmark samples.

[2026-10-28] Background randomization ablation study (E0, E1, E2) | Evaluated on RTX 3050. Baseline E0 collapses from 56.5% to 31.5% under background shifts; E2 (mixed 40/60) maintains 60.5% accuracy (91.0% retention) and reduces center-masked spurious reliance from 37% down to 25%.

[2026-10-28] Multi-layer selective prediction with Helmholtz Free Energy Score | Max-softmax fails on flat grey tiles (99.95% confidence). Layer 1 optical heuristics immediately intercept non-leaf tiles, and Layer 2 Helmholtz Energy score ($E(x) = -T \log \sum e^{z_i/T}$) boosts OOD AUROC to 0.9548 with FPR95 of 10.0%.

[2026-10-28] Hierarchical crop-guided masked softmax inference | Solanaceae crops (Potato and Tomato) share visual morphology and common pathogens. Optional crop parameter applies masked softmax renormalization to restrict inference to species candidates, eliminating cross-genus confusion.

[2026-10-28] Dual explainability (Grad-CAM + SHAP) with greedy pixel deletion faithfulness | Hook-based Grad-CAM and SHAP GradientExplainer reveal 70.4% and 64.5% attribution mass on leaf foreground. Hooker et al. pixel deletion curve demonstrates confidence collapse from 99.3% to 13.9% at 50% deletion, proving saliency faithfulness.

[2026-10-28] Upgraded Schema v2 advisories and foliar epidemiological weather rules | Integrated verified university extension guidelines (UC IPM, Cornell, Penn State) and Indian national institutes (ICAR-CPRI, ICAR-IIHR, ICAR-NRCG, TNAU) in kb/advisories.yaml with 6 weather risk rules in kb/weather_rules.yaml; prohibited dose modification.

[2026-10-28] 4-Criterion promotion gate, batch inference benchmark, and automated drift retraining trigger | Champion promotion requires Macro-F1 delta >= -0.005, recall floor >= 0.85, OOD AUROC >= 0.85, and background retention >= 75%. Batch inference achieves 317.4 fps on batch-64. Drift monitor writes retraining_trigger.json upon KS/PSI breach.

[2026-10-28] 5-Tab Streamlit governance dashboard | UI enhanced with tabs for Diagnostic Screening, Service Logs & Telemetry, Model Evidence & Bias Study, Coverage & Structural Gaps, and MLOps Lifecycle & Governance.

[2026-10-28] Comprehensive Part 9 Documentation Pack delivered | Created SYLLABUS_MAP.md, RISK_REGISTER.md, DATASHEET.md, CLOUD_MAPPING.md, and dvc.yaml; refreshed ARCHITECTURE.md, SOURCE_AUDIT.md, DEMO_SCRIPT.md, VIVA_QA.md, and SLIDE_OUTLINE.md with all verified empirical metrics.

[2026-10-28] Added Automated Data Validation Suite (Unit-II) | src/data_validation.py and scripts/validate_dataset.py execute 7 automated assertions: split manifest existence, column schema, zero nulls, zero group leakage, class balance bounds (38 classes), and PIL tensor integrity.

[2026-10-28] Built Comparative XAI Benchmark with LIME & SHAP (Unit-V) | scripts/benchmark_xai.py benchmarks Grad-CAM, SHAP GradientExplainer, and LIME Superpixel ImageExplainer side-by-side, measuring latency, attribution sparsity, and leaf foreground concentration (Grad-CAM: 84.7%, SHAP: 77.2%, LIME: 92.3%). Saved reports/figures/xai_trio_comparison.png.

[2026-10-28] Built Apache Airflow Simulation Runner & SageMaker Cloud Compiler (Unit-III & Unit-VI) | scripts/run_airflow_simulation.py executes all DAG tasks with live logging (SUCCESS); scripts/compile_sagemaker_pipeline.py generates verified AWS SageMaker Pipeline Workflow JSON in reports/sagemaker_pipeline_definition.json.

[2026-10-28] Upgraded UI Tab 5 into VIT MLOps Curriculum & Governance Hub | Tab 5 now features interactive accordions for Units I to VI with one-click buttons to execute reproducibility checks, data validation, Airflow retraining simulations, and view SageMaker definitions.
