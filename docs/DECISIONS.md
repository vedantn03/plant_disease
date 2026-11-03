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

[2026-10-28] Initial promotion-gate proposal superseded | The earlier gate proposal used an
absolute background-retention ceiling and an OOD criterion. The current provisional gate uses
within-crop VAL macro-F1, per-class recall, champion-relative masked-background drop under the
three-fill protocol, and ECE. Batch inference and drift artifacts remain separate evaluations.

[2026-10-28] Historical dashboard expansion note (superseded) | The current UI has three tabs: Diagnostic Screening, Production Telemetry, and MLOps System.

[2026-10-28] Comprehensive Part 9 Documentation Pack delivered | Created SYLLABUS_MAP.md, RISK_REGISTER.md, DATASHEET.md, CLOUD_MAPPING.md, and dvc.yaml; refreshed ARCHITECTURE.md, SOURCE_AUDIT.md, DEMO_SCRIPT.md, VIVA_QA.md, and SLIDE_OUTLINE.md with all verified empirical metrics.

[2026-10-28] Added Automated Data Validation Suite (Unit-II) | src/data_validation.py and scripts/validate_dataset.py execute 7 automated assertions: split manifest existence, column schema, zero nulls, zero group leakage, class balance bounds (38 classes), and PIL tensor integrity.

[2026-10-28] Built Comparative XAI Benchmark with LIME & SHAP (Unit-V) | scripts/benchmark_xai.py benchmarks Grad-CAM, SHAP GradientExplainer, and LIME Superpixel ImageExplainer side-by-side, measuring latency, attribution sparsity, and leaf foreground concentration (Grad-CAM: 84.7%, SHAP: 77.2%, LIME: 92.3%). Saved reports/figures/xai_trio_comparison.png.

[2026-10-28] Airflow runner is SIMULATION; SageMaker compiler is local | `scripts/run_airflow_simulation.py` output is SIMULATION and does not indicate live Airflow execution. The pipeline compiler writes a local definition artifact.

[2026-10-28] Historical UI hub note (superseded) | The current UI contains three tabs; curriculum and governance content is under MLOps System.

[2026-10-31] Untrack scratch/ and large figure files | `scratch/` directory and `reports/figures/segmentation_40_grid.png` (~15 MB, exceeding the 5 MB repository limit) are untracked and added to `.gitignore`. The figure can be regenerated deterministically via `python scripts/benchmark_segmentation.py`.

[2026-10-31] Retain demo_samples and SHAP figures with licence review disclaimer | `demo_samples/*.jpg` (~300 KB each) and SHAP figures in `reports/figures/shap/` (~1 MB each) are kept in the repository for offline test suite & demonstration capability. PlantVillage licence must be checked before public redistribution.

[2026-10-31] Use FastAPI lifespan for serving lifecycle | T9 replaces deprecated
`@app.on_event("startup")` and `"shutdown"` handlers with one lifespan context.
The lifecycle change is independent of model source selection.

[2026-10-31] Make registry-backed serving opt-in | Local `models/best_model.pth` remains the default
serving source. Setting `serving.model_source: registry` resolves the configured MLflow
`@champion` alias and downloads exactly one `.pth` artifact without modifying the local champion.

[2026-10-31] T4 (MinIO object store) deferred | MinIO requires either a separate Docker service or a running binary. The local DVC remote at `D:\dvc_remote` is fully sufficient for offline demo and viva. MinIO would add complexity (two extra containers, bucket lifecycle config) with no measurable benefit given the project's local-first constraint. Re-evaluate if the demo moves to a multi-machine setup.

[2026-10-31] Shared venv (.venv) with airfare project retained for now | `D:\VIT\Hackathons\SIH'26\Airafare-proj\.venv` is shared between the plant_disease and airfare sub-projects. This is technically fine but risks version conflicts when one project upgrades a package. Proposed clean-up: create `plant_disease\.venv_pd` (plant-disease-only venv) after the demo so each project pins its own dependency tree. This is a post-demo housekeeping task and has no impact on current reproducibility.

[2026-10-31] manifest_hashes.json stays git-tracked, not DVC-tracked | The file is a tiny (<1 KB) hash-of-hashes manifest for the raw dataset. Git-tracking it gives a single-command diff (`git diff`) to detect dataset drift between sessions. DVC would add indirection with no space saving. Ref: https://dvc.org/doc/user-guide/data-management/remote-storage

[2026-10-31] Temperature scaling not applied at serving time | The current champion refit on the saved VAL logits found optimal T=0.9759, with VAL NLL 0.01920 at T=1 and 0.01919 at the optimum. The improvement is negligible, so serving remains unscaled (T=1). The sign is divide: `scaled_logits = logits / T`.

[2026-10-31] Fix per-crop macro-F1 computation via scoped labels | Naive sklearn `f1_score(ct, cp, average='macro')` on per-crop subsets treated out-of-crop false positives as extra classes with 0 support, artificially collapsing Potato macro-F1 to 73.71% and Tomato to 82.10%. Scoping evaluation via `labels=np.unique(ct)` yields true in-domain macro-F1 of 98.27% for Potato and 98.52% for Tomato, matching the actual 99.03% model performance.

[2026-10-31] A1-A7 artifact audit and serving decision | The refit uses the current champion
checkpoint, the same VAL preprocessing, and the same saved VAL logits used for evaluation.
The earlier diagnostic fit came from the old evaluator's non-unit initialization; it was not a
valid serving value. The current implementation starts at T=1, includes T=1 in its search, and
selects the lowest VAL NLL. Serving uses `temperature=1.0` in `src/api.py:269` and `:408`.

[2026-10-31] A4 bootstrap ECE note | The percentile bootstrap mean/interval for ECE is not a
confidence interval for the point estimate and can be shifted upward because ECE is a nonlinear,
absolute-value, data-adaptive statistic. The project keeps the existing percentile bootstrap as a
descriptive uncertainty summary and records this upward-bias caveat; it does not treat the interval
as evidence that must contain the point estimate.

[2026-10-31] A5 speed benchmark scope | The old step benchmark used pre-generated tensors and
excluded DataLoader/JPEG work. The measured champion run took 287.6 seconds per epoch and
2321.9 seconds wall-clock for 8 epochs (about 39 minutes).

[2026-10-31] A7 champion-version wording | The checkout contains no MLflow registry database or
registered-version artifacts, so the identity and safety of “Champion v1” cannot be verified.
Documentation must not claim v1 is registered or use it as a rollback target. Rollback now selects
only older versions tagged `gate_verdict=PASS`; otherwise it refuses the move.

[2026-10-31] Secret scanning claim corrected | The pinned pre-commit hooks are trailing-whitespace,
end-of-file-fixer, YAML/JSON validation, large-file checking, and Ruff. They do not scan secrets.
The claim was corrected in `.pre-commit-config.yaml` and `docs/MANUAL_STEPS.md` rather than adding
an unverified third-party hook. GitHub secret scanning and the requested history checks remain
separate manual checks.

[2026-10-31] Plain train command is candidate-only | `scripts/train.py` writes the best checkpoint
to `models/candidates/<run_id>.pth` even when invoked without `--candidate`; the flag is retained
as documentation. This safer default prevents accidental overwrite of `models/best_model.pth`.
Promotion remains a separate dry-run gate. The DVC train output declaration is currently
inconsistent and must be resolved before any real `dvc repro`.

[2026-10-31] DVC champion-output mismatch options | Before enabling real reproduction, choose one:
(a) declare a candidate output path; (b) remove the train output declaration; or (c) add a
separate explicitly approved champion-producing stage. No option is implemented here.

[2026-11-02] DVC train output follows candidate-only training | The train stage now declares
`models/candidates/` with `persist: true` because `scripts/train.py` writes only candidate
checkpoints and must not overwrite `models/best_model.pth`. The champion remains a separate DVC
tracked file and stays a dependency of evaluation stages. The rejected alternatives were removing
the train output entirely (which would make candidate artifacts invisible to the pipeline) and
adding a champion-producing stage (which would create an unsafe promotion path before gate approval).
The `persist` syntax is documented by DVC at
https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#outs and in the DVC pipeline
running guide at https://github.com/treeverse/dvc.org/blob/main/content/docs/user-guide/pipelines/running-pipelines.md.

After pulling this change, verify the existing champion object without running reproduction:

```powershell
(Get-FileHash models\best_model.pth -Algorithm MD5).Hash
..\.venv\Scripts\dvc.exe status
..\.venv\Scripts\dvc.exe push
(Get-FileHash models\best_model.pth -Algorithm MD5).Hash
..\.venv\Scripts\dvc.exe status -c
```

The two MD5 outputs must both be
`7203474FB41FEDAA504A0033FC13A6D3`. `dvc status` may report the changed train
stage until a candidate run updates its lock entry; this is expected and is not
a reason to run `dvc repro`. If any MD5 differs or the champion is missing,
stop immediately and restore only from the existing `D:\dvc_remote` backup; do
not overwrite the file from another checkpoint.

[2026-11-03] Candidate selection uses VAL metrics; TEST is reporting only | Promotion-gate
comparisons and candidate selection use saved VAL metrics only. TEST metrics are for final
reporting and must not drive model selection; seed 44's higher TEST macro-F1 is not a basis for
promotion. Gate thresholds remain provisional pending review of the seed-variance report.

[2026-11-03] Registry-verified serving design (WP2) | The registry alias is authoritative for
champion identity, resolved read-only at API startup. The API loads checkpoint bytes only from the
read-only mounted `models/` volume and compares their MD5 plus manifest version/run ID with the
alias. The committed `models/champion_manifest.json` is initialized for v2 from the existing DVC
pointer. On a successful explicitly applied promotion, `scripts/promote.py` writes the new
manifest only after alias movement. `/health` reports mismatch and WP6 refuses to load a mismatched
checkpoint. Alternatives considered: loading the
registry artifact directly (may reference host-only Windows paths and would break Linux Docker),
or making local checkpoint identity authoritative (does not meet the registry-source-of-truth
requirement). Neither alternative is implemented. Host inspection cannot establish whether saved
artifact URIs are readable inside the Linux container; run `scripts/inspect_registry.py` and test
container-side access before making that claim.

[2026-11-03] Dataset-free serving config (WP3) | `load_config()` continues to require
`DATA_ROOT` by default, so training, audit, evaluation, and other existing callers retain their
validation. The FastAPI startup path opts out explicitly because inference uses the mounted
checkpoint and serving configuration, not the training dataset. The checkpoint remains required
for successful predictions. The prior Compose `DATA_ROOT=/app/data` value was an unmounted
placeholder; it satisfied the string-only loader check but did not make dataset access necessary.

[2026-11-03] Local Compose monitoring (WP3) | Compose profiles are `core` (API and UI) and
`monitoring` (Prometheus and Grafana). Prometheus scrapes the API service name on port 8000 at a
15-second interval, and rule files start empty for the next work package. Prometheus and Grafana
use named storage volumes. The API model mount remains read-only. Pinned images: Prometheus
`prom/prometheus:v3.13.2` ([official tags](https://hub.docker.com/r/prom/prometheus/tags)) and
Grafana `grafana/grafana:13.2.1` ([official tags](https://hub.docker.com/r/grafana/grafana/tags));
their Docker install references are [Prometheus](https://prometheus.io/docs/prometheus/latest/installation/)
and [Grafana](https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/).
The scrape interval is declared in `monitoring/prometheus.yml`. When API runs on the host,
Prometheus can instead target `host.docker.internal:8000`.

[2026-11-03] MLflow-in-Docker investigation (WP3) | The host MLflow tracking store is
`sqlite:///mlflow.db`, and the registered model artifact sources may be Windows paths. No
containerized MLflow service or artifact proxy is implemented. Options: (1) run the MLflow UI on
the host through `tasks.ps1 mlflow-ui` (recommended for local inspection; uses the host database
and host filesystem, so do not use concurrently with host training that writes the same SQLite
store); (2) mount a copy of the SQLite database into a container (risks stale metadata, and copied
Windows artifact URIs may still not resolve in Linux); or (3) build an artifact proxy (additional
service, access control, and path mapping concerns). The host UI task is provided; the user must
decide whether either container option should be pursued later.

[2026-11-03] Windows Compose path quoting (WP3) | Compose uses relative bind-mount paths in YAML,
which avoids embedding the apostrophe-containing project path in shell command arguments. In
PowerShell, use `Set-Location -LiteralPath` with a double-quoted path. If Docker Desktop cannot
translate a bind mount from that location, use a checkout under an apostrophe-free path that
Docker Desktop can access.

[2026-11-03] Docker base matches pinned scientific dependencies (WP3) | The supplied Docker build
output showed the pinned NumPy 2.5.3 and SciPy 1.18.1 releases require Python 3.12, while the
Dockerfile used Python 3.11. The API/UI base is therefore `python:3.12-slim`, an available Docker
Official Image tag ([tag listing](https://hub.docker.com/_/python?tab=description)); the existing
NumPy/SciPy pins remain unchanged.

[2026-11-03] Real-log drift monitor proposal (WP4) | The baseline samples TRAIN records with the
configured seed and sample count, and derives brightness/blur through `src.image_features`. It
also runs read-only champion inference on that same sample to establish a confidence baseline;
the checkpoint is never written. The exporter evaluates the most recent configured request-log
window and labels replay share separately from other logged traffic. The proposed rule is
PSI greater than the conventional 0.2 value AND KS p-value below the existing configured
`ks_alpha`; this is provisional and awaits user approval. `alert_for` is in config and mirrored in
the Prometheus rule file. The legacy scripts `simulate_drift.py` and `check_drift_and_trigger.py`
remain SIMULATION ONLY; the latter no longer creates synthetic observations or a real retraining
action. No real baseline, exporter result, or alert was run in this environment.

[2026-11-03] Background sensitivity in serving (WP5) | The optional response comparison re-runs
inference after replacing pixels outside the ExG mask with that image's mean RGB colour. A
segmentation fallback/failure or mask area outside the configured bounds yields `unknown`; it never
counts as consistent. The mask bounds and VAL sample size in `configs/config.yaml` are provisional
and remain subject to review. `abstain_on_bg_inconsistent` defaults to false. This measures
sensitivity to this background replacement and is not proof of shortcut learning. Added latency is
reported only by a run on actual VAL images or real API requests; no such measurement was run here.
Field-probe output is aggregate and unlabelled, with no persisted per-image responses.

[2026-11-03] WP6 promotion lifecycle | Candidate selection is VAL-only; TEST metrics remain for
reporting. User-selected thresholds are within-crop macro-F1 at least champion minus 0.005,
minimum class recall at least 0.85 with a Wilson 95% lower bound when support is below 30,
per-image-mean background drop at most champion plus 0.03, and ECE at most 0.01. Black and
mid-grey fill drops are informational. These are still provisional: `--apply` requires
`configs/gates/approval.yaml` to set `approved: true` and include the SHA-256 of the exact
threshold file. Any change to that file invalidates the approval hash. Observed seed ranges are
generated from `reports/seed_variance.json` by `scripts/sync_gate_spreads.py`; the threshold
file is not populated by hand. The supplied seed report states that three seeds are a small,
descriptive sample, not a stable variance estimate.

The registry alias identifies the serving version. `models/champion_manifest.json` maps versions
to run IDs, repository-relative checkpoint paths, MD5 values, and gate verdicts. The API loads
weights only from the mounted local checkpoint after matching registry version, run ID, and
checkpoint MD5. A candidate must have a matching DVC pointer, and that pointer must be committed
before an applied promotion. Alias movement is followed by an atomic manifest replacement; if
manifest writing fails, promotion attempts to restore the prior alias and manifest.
`models/best_model.pth` remains the v2 checkpoint and is never overwritten by promotion.

`POST /admin/reload` requires `X-Admin-Token` to match `ADMIN_RELOAD_TOKEN` from ignored `.env` and
requires a local Host plus a loopback or private-network client address. This supports Docker
Desktop's private bridge forwarding while the Compose API port is bound to host loopback. The
Compose service reads the mounted MLflow SQLite file as read-only to resolve the alias; do not run
container serving concurrently with a host process that writes the same database. Registry writes
occur only when the user runs
`scripts/promote.py --apply` or `tasks.ps1 rollback`.
