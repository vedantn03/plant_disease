# Final presentation outline

Designed for a ten-minute faculty and Syngenta evaluation. Snapshot metrics below are from the
committed files in `reports/snapshot/`. No runtime screenshot is implied where only a source or
JSON report is listed.

## Slide 1 — The problem: trustworthy plant disease screening

- Plant disease classification is the target task.
- A useful demonstration needs traceable data, model, evaluation, serving, and monitoring evidence.
- This is a decision-support prototype, not a field diagnosis service.

**Show:** `docs/ARCHITECTURE.md` and the Diagnostic Screening screenshot, if captured at
`docs/evidence/ui/`.

**Say:** “I built an end-to-end MLOps prototype and will show what its evidence supports, including
where it does not yet support field claims.”

## Slide 2 — Why a PlantVillage score is not field validation

- The saved test headline is 0.993 accuracy and 0.9903 macro-F1.
- The dataset consists of lab-style leaf images; field representativeness is not established.
- Background probes show sensitivity, but do not prove shortcut learning.
- The dataset licence and redistribution terms still need verification.

**Show:** `reports/snapshot/eval_summary.json` and `reports/snapshot/robustness_metrics.json`.

**Say:** “The headline describes this held-out dataset split; it does not establish performance on
field photographs.”

## Slide 3 — Honest evaluation and a metric bug caught

- Headline test macro-F1 is 0.9903; within-crop macro-F1 excluding single-class crops is 0.9888.
- Out-of-crop predicted labels once depressed single-class crop macro-F1; the fix scopes labels to the crop’s true classes, and old per-crop outputs remain stale.
- Saved leading confusions include Corn Cercospora / Northern Leaf Blight and Tomato Early / Late blight.

**Show:** `reports/snapshot/eval_summary.json`, `scripts/train.py` function
`compute_per_crop_metrics`, and `tests/test_wp1_close_gate.py`.

**Say:** “We caught and regression-tested a reporting bug, so I separate the headline score from
the corrected within-crop calculation and mark old per-crop files stale.”

## Slide 4 — Architecture: from image to observable service

- Streamlit provides the user interface.
- FastAPI serves predictions and trust metadata.
- Prometheus scrapes service metrics; Grafana displays them.
- Drift monitoring reads request logs and exports window statistics.

**Show:** `docs/ARCHITECTURE.md` and `docker-compose.yml`.

**Say:** “The UI, inference API, and monitoring components are separate so prediction behavior and
operational signals can be inspected independently.”

## Slide 5 — Data and model versioning

- DVC declares data and pipeline artifacts in `dvc.yaml`.
- The configured local DVC remote is `D:\dvc_remote` on the owner’s machine.
- MLflow tracks runs and registry versions; the saved registry export identifies champion version 2.
- The committed manifest maps version 2 to the mounted checkpoint and its MD5.

**Show:** `dvc.yaml`, `reports/snapshot/registry_export.json`, and
`models/champion_manifest.json`.

**Say:** “DVC carries data and checkpoint pointers, while MLflow records experiments and the
registry alias; this repository snapshot does not prove remote availability.”

## Slide 6 — Promotion gate and lifecycle evidence

- The provisional VAL gate criteria are within-crop macro-F1 margin 0.005, recall floor 0.85, per-image-mean background-drop margin 0.03, and ECE ceiling 0.01; compare them with the committed three-seed summary.
- The seed report calls three seeds a small descriptive sample.
- The approval file remains unapproved, and the committed promotion log contains dry-runs only; every recorded entry says the alias did not move.
- There is no real promotion or rollback event in the saved evidence, so I will not claim one.

**Show:** `reports/snapshot/seed_variance.json`, `configs/gates/gate_thresholds.yaml`,
`configs/gates/approval.yaml`, and `reports/snapshot/promotion_log.jsonl`.

**Say:** “The gate and rollback path are implemented, but the evidence pack records rehearsal only;
no registry change is being presented as a completed event.”

## Slide 7 — Serving and monitoring

- The API exposes `/health`, `/predict`, and `/metrics`.
- Prometheus scrapes API metrics; Grafana provisions service and model-trust panels.
- The dashboard is repository evidence; a screenshot should be captured from the local running stack.
- CI’s fixture model checks plumbing, not classifier accuracy.

**Show:** `src/api.py`, `monitoring/prometheus.yml`,
`monitoring/grafana/dashboards/system-health-model-trust.json`, and a local screenshot under
`docs/evidence/monitoring/` if captured.

**Say:** “These panels make observed request and model signals visible; they do not substitute for
an accuracy evaluation.”

## Slide 8 — Drift detection with staged traffic

- The saved latest window has 200 records and replay share 0.765.
- Its drift flag is 1 for the recorded window.
- Replay is explicitly marked staged traffic.
- A clean-control run and a captured Prometheus alert firing are not present in this snapshot.

**Show:** `reports/snapshot/drift_latest.json`; add a Prometheus alert screenshot only if captured
under `docs/evidence/drift/`.

**Say:** “This is a drift result on a staged, replay-heavy request window; I do not claim a clean
control or field drift event from this evidence.”

## Slide 9 — Background sensitivity and masking artifacts

- Across the three saved seeds, black-fill drop ranges from 0.1524 to 0.1843.
- Mid-grey drop ranges from 0.0563 to 0.0997.
- Per-image-mean drop ranges from 0.0163 to 0.0570.
- Fill choice and segmentation affect the result; this is not proof of shortcut learning.

**Show:** `reports/snapshot/seed_variance.json` and `reports/snapshot/bg_check_rates.json`.

**Say:** “The defensible interpretation is that replacing background pixels with the image’s own
mean colour costs a few macro-F1 points in these measurements; black-fill drops include masking
artifacts.”

## Slide 10 — Advice with provenance and graded tiers

- The active knowledge base has one source-backed entry and thirty-seven placeholders.
- The retained entry is not yet reviewed, so it is not grade A.
- Placeholders direct users to consult a local agronomist.
- Chemical options require the highest advice tier and cited source text.

**Show:** `reports/snapshot/advisory_audit.json`, `docs/ADVISORY_AUDIT.md`, and an Advice passport
screenshot under `docs/evidence/advisory/` if captured.

**Say:** “The advice feature reports its evidence status; most entries remain placeholders rather
than unsupported treatment recommendations.”

## Slide 11 — CI/CD and release controls

- CI separates lint, tests, secret scan, Compose checks, image builds, and tracked-file checks.
- Fixture-model tests verify API wiring only.
- GHCR publishing is manually dispatched and requires an explicit publish input on a `v` tag.
- Pushing a tag by itself does not publish an image.

**Show:** `.github/workflows/ci.yml`, `.github/workflows/publish-api.yml`, and the PR Checks page
if captured.

**Say:** “The release workflow is opt-in; ordinary pushes and tag pushes alone do not publish the
API image.”

## Slide 12 — Limitations and next work

- Field validation and dataset-licence verification remain open.
- Drift replay and the legacy Airflow script are simulations or staged traffic.
- SageMaker support is documentation/local compilation, not managed execution.
- Next: independently review advice sources, capture clean-control monitoring evidence, and evaluate
  on consented field images.

**Show:** `docs/STATUS.md`, `docs/PROGRESS.md`, and `docs/DECISIONS.md`.

**Say:** “The next step is stronger external evidence: reviewed sources, controlled monitoring, and
field evaluation with appropriate data rights.”
