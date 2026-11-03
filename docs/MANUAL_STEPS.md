# Manual Action Steps Log (Master Plan v2)

This document mirrors all `🛑 MANUAL ACTION REQUIRED` blocks raised during the lifecycle implementation.

---

## Task T2 — Preflight, Runner, Hooks & Resource Budget

### Step M2.1: Pre-commit Installation & Preflight Execution
- **Why**: Verifies preflight diagnostics directly on your active PowerShell console and activates git pre-commit hooks to block large files (>5MB) and run heuristic secret checks. Secret detection can miss some secret formats; it is not a guarantee.
- **Do**:
  1. Open PowerShell and navigate to the project directory:
     ```powershell
     d:
     cd "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
     ```
  2. Install `pre-commit` into the venv and register the git hook:
     ```powershell
     ..\.venv\Scripts\pip.exe install pre-commit
     ..\.venv\Scripts\pre-commit.exe install
     ```
  3. Execute the preflight script:
     ```powershell
     powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1
     ```
- **Expect**:
  - `pre-commit installed at .git/hooks/pre-commit`
  - All items in preflight show `[PASS]` (Docker CLI will show `[WARN]` until Docker Desktop is started; ports 5000-9000 show free).
- **If it fails**:
  - `pre-commit: command not found` -> Ensure you are invoking `..\.venv\Scripts\pre-commit.exe`.
  - `ExecutionPolicy error` -> Use the exact `-ExecutionPolicy Bypass` flag provided.
- **Send me**:
  - Paste back the terminal output of `pre-commit install` and `scripts\preflight.ps1`.
  - Save the console output to `docs/evidence/preflight/preflight_output.txt`.

### Step M2.2: Start Docker Desktop
- **Why**: Docker Engine is required for subsequent platform tasks (MLflow, MinIO, Prometheus, Grafana, Airflow).
- **Do**:
  1. Launch **Docker Desktop** from the Windows Start menu.
  2. Wait until the whale icon in the Windows taskbar stays solid (Docker engine initialized).
  3. Verify from PowerShell:
     ```powershell
     docker info
     ```
- **Expect**: `Server:` section outputs details without `failed to connect to docker API`.
- **Send me**: Confirmation that `docker info` runs without daemon errors.

---

## Task T3 — DVC Data Pipeline & Versioning

### Step M3.1: DVC Push, Status Verification & Restore Proof
- **Why**: Pushes versioned data assets (splits, reports, model weights) to local storage remote `D:\dvc_remote` and proves data reproducibility and pull restoration.
- **Do**:
  1. Navigate to `plant_disease`:
     ```powershell
     d:
     cd "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
     ```
  2. Push tracked assets to DVC remote:
     ```powershell
     ..\.venv\Scripts\dvc.exe push
     ```
  3. Verify status with remote:
     ```powershell
     ..\.venv\Scripts\dvc.exe status -c
     ```
  4. Perform restore proof on a small regenerable file (`reports/reproduce_check.json`):
     ```powershell
     Remove-Item reports\reproduce_check.json
     ..\.venv\Scripts\dvc.exe pull reports/reproduce_check.json.dvc
     # Or run general pull if individual pull isn't mapped:
     ..\.venv\Scripts\dvc.exe pull
     ```
- **Expect**:
  - `dvc push` transfers files to `D:\dvc_remote`
  - `dvc status -c` reports `Data and pipelines are up to date`
  - `reports\reproduce_check.json` is restored
- **If it fails**:
  - Check whether `D:\dvc_remote` exists (`New-Item -ItemType Directory -Force D:\dvc_remote`).
- **Send me**:
  - Output of `dvc push` and `dvc status -c`. Save them into `docs/evidence/dvc/dvc_push_output.txt` and `docs/evidence/dvc/dvc_status_c.txt`.

---

## Task T7 — Registry Lifecycle, Trust Gate & Rollback

### Step M7.1: Candidate Promotion Check & Alias Verification
- **Why**: Proves that the MLflow model registry strictly enforces the four-criterion trust gate: a candidate must retain within-crop VAL macro-F1, stay within the champion-relative bg_drop margin under the agreed fill protocol, meet the ECE ceiling, and meet the recall floor.
- **Do**:
  1. Open PowerShell in `plant_disease`:
     ```powershell
     d:
     cd "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
     ```
  2. Run the promotion script in dry-run mode to evaluate the candidate:
     ```powershell
     ..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_48ecf14d.json --candidate-name candidate_seed_43 --champion-metrics reports\candidate_gate_comparison.json --champion-name champion_v2 --champion-version 2
     ```
  3. Verify the logged decisions in `reports/promotion_log.jsonl`; no alias moves while thresholds are provisional.
- **Expect**:
  - Model is evaluated against gate criteria.
  - Decision is logged with full transparency.
- **Send me**: Output of `scripts\promote.py`.

For the current WP1 review, dry-run candidate-only reports can be passed with
`--candidate-metrics reports\candidate_gate_<run8>.json`. Champion metrics must come from the
   refreshed comparison report; no `--version` is needed, so dry-run does not attempt registry tagging.

## WP1 — Close Promotion Gate (manual machine steps)

Run from PowerShell in the repository after checking out `feat/wp1-close-gate`:

1. Pull the branch: `git pull origin feat/wp1-close-gate`.
   Branch setup from `main` (required because the sandbox could not write `.git`):
   `git switch main`, `git pull origin main`, then `git switch -c feat/wp1-close-gate`.
2. Run the full test suite: `..\.venv\Scripts\pytest.exe -q`.
3. Refresh the seed-43 comparison and seed-44 candidate-only report. The evaluator compares each
   model against the same VAL split and writes per-class recall counts:
   ```powershell
   ..\.venv\Scripts\python.exe scripts\evaluate_candidate_gate.py --candidate models\candidates\48ecf14da2d84803a36d6fd108bd6b61.pth --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --seed 43
   ..\.venv\Scripts\python.exe scripts\evaluate_candidate_gate.py --candidate models\candidates\0e2c51abf0834c248e03c5aef774e291.pth --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --seed 44 --preserve-comparison
   ```
   Seed 43 took about 723 seconds on the supplied machine; allow roughly 12–15 minutes for seed
   44. This is an estimate, not a measured seed-44 runtime.
   If you already have the seed-44 output from the preceding version, rerun both evaluator
   commands after pulling this fix because its within-crop calculation changed to match
   `scripts/evaluate.py`.
4. Summarize without loading models:
   `..\.venv\Scripts\python.exe scripts\summarize_seed_variance.py`
5. Dry-run v3 and v4 against v2 (do not pass `--version`):
   ```powershell
   ..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2
   ..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_comparison.json --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2
   ```
   Each stdout JSON should contain the action, `dry_run: true`, candidate gate verdict/criteria,
   `champion_version: "2"`, and `alias_moved: false`. Two appended JSONL records should have the
   same fields; dry-run does not move the alias.
6. Stage only tracked code/docs and the new regression test, commit conventionally, then push:
   ```powershell
   git add .gitignore docs scripts\train.py scripts\promote.py scripts\evaluate_candidate_gate.py scripts\summarize_seed_variance.py tests\test_wp1_close_gate.py
   git commit -m "fix: close candidate promotion gate"
   git push -u origin feat/wp1-close-gate
   ```
   Do not stage generated files under `reports\`.
7. Open a PR from `feat/wp1-close-gate` into `main` in the repository host UI.

If a checkpoint, validation manifest, dependency, or config is missing, stop and paste the exact
error. Paste back exactly `reports\seed_variance.md` and both dry-run stdout JSON outputs.

---

## WP2 — Registry-verified serving, metrics, and replay

Install the metrics library and report the installed version:

1. Install: `..\.venv\Scripts\pip.exe install prometheus-client`.
   Expected: pip reports successful installation; paste the installed `prometheus-client` version.
2. From the repo directory, start the API in another PowerShell window: `.\tasks.ps1 serve`.
3. Check health: `Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5`.
   Expected: fields include `model_version`, `model_source`, `checkpoint_md5_prefix`, and
   `registry_match`. `unknown` means registry identity is not configured/verified.
4. Check metrics: `Invoke-WebRequest http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content`.
   Expected: Prometheus text including metric families; observations appear after requests.
5. Replay normal TEST traffic:
   `..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1`.
6. Replay degraded traffic:
   `..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade blur`.
   Expected: each request output includes an HTTP status/body and concludes with
   `traffic_source=replay`; logs include `traffic_source: replay` and metrics carry that label.
7. Read-only registry inspection (when local MLflow DB is available):
   `..\.venv\Scripts\python.exe scripts\inspect_registry.py`.
   Paste its version `source_uri` fields and `container_compatibility`. Host retrieval alone does
   not prove Linux-container access; test URI access inside the running API container if desired.
8. Before committing WP2, resolve the existing WP1 work. Do not discard its modifications or
   generated reports. Create `feat/wp2-serving-metrics` from updated `main` in a clean worktree,
   apply only the WP2 files, then run:
   `git add .gitignore Dockerfile configs\config.yaml docs\DECISIONS.md models\champion_manifest.json models\champion_manifest.json.dvc requirements.txt scripts\inspect_registry.py scripts\promote.py scripts\replay_traffic.py src\api.py src\model_loader.py tests\test_api.py tests\test_model_loader.py tests\test_replay_traffic.py conftest.py`
   `git commit -m "feat: add registry-verified serving metrics and replay"`
   `git push -u origin feat/wp2-serving-metrics`
9. Open a PR from `feat/wp2-serving-metrics` into `main`.

Paste back the `/health` JSON, `/metrics` output before and after replay, both replay summaries,
the installed library version, registry inspection output, and test output. If a request fails,
paste the exact error and stop before PR creation.

## WP3 — Docker Compose platform and monitoring

Run from PowerShell. Preserve the existing WP1/WP2 working tree and apply this work package
on a clean `feat/wp3-compose-monitoring` branch from updated `main` before staging it.

1. Change to the repository using a literal path (the apostrophe is valid inside this
   double-quoted PowerShell argument):
   ```powershell
   Set-Location -LiteralPath "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
   ```
   If Docker Desktop cannot translate bind mounts from this path, use a separate checkout in
   an apostrophe-free path that Docker Desktop can access; do not copy or replace model files.
2. Create `.env` from placeholders if needed, edit the Grafana credentials, then validate the
   fully interpolated Compose configuration:
   ```powershell
   if (!(Test-Path .env)) { Copy-Item .env.example .env }
   notepad .env
   docker compose config
   ```
   Expected: configuration shows API/UI in `core`, Prometheus/Grafana in `monitoring`, and the
   configured local credentials. Do not paste the resolved config because it contains the
   password. If Compose reports missing Grafana environment variables, ensure both credential
   keys are set in `.env`. If a port is already bound, identify and stop only the known process
   using that port before retrying.
3. Build and start the core profile, then check API health, metrics, and the UI:
   ```powershell
   .\tasks.ps1 up-core
   Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/_stcore/health | Select-Object -ExpandProperty Content
   ```
   Expected: containers reach healthy state; `/health` reports `model_loaded: true` when
   `models\best_model.pth` is present, `/metrics` returns Prometheus text, and the UI health
   endpoint returns successfully. The dataset is not mounted or required for serving. If health
   reports `model_loaded: false`, stop and verify the expected checkpoint is already present;
   do not replace it with a candidate. On a first run, image/dependency downloads can make the
   build take longer. If Docker reports memory pressure, close other heavy workloads or adjust
   Docker Desktop's memory allocation, then retry.
4. Start monitoring and check the Prometheus target:
   ```powershell
   .\tasks.ps1 up-monitoring
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/targets | Select-Object -ExpandProperty StatusCode
   ```
   Open `http://127.0.0.1:9090/targets` and confirm `plant-disease-api` is `UP`. If it is
   `DOWN`, inspect `docker compose --profile core --profile monitoring logs prometheus api`;
   confirm core is running and the API healthcheck passes. The scrape interval is `15s`, declared
   in `monitoring\prometheus.yml`. For an API running on the host, use
   `host.docker.internal:8000` as the Prometheus scrape target instead of `api:8000`.
5. Open `http://127.0.0.1:3000`, sign in with the credentials in `.env`, and open the provisioned
   **System health and model trust** dashboard. Empty panels before requests are expected; no
   sample series are fabricated. Save dashboard screenshots under `docs\evidence\monitoring\`:
   ```powershell
   New-Item -ItemType Directory -Force docs\evidence\monitoring
   ```
6. From the repository directory, with the API still running, replay TEST images normally and
   then with blur degradation:
   ```powershell
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade blur
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
   ```
   Expected: each completed run ends with `traffic_source=replay`; the dashboard gains observed
   request, latency, and confidence samples. A replay request prints its response body, which
   includes a base64 heatmap and can be long. At `--rate 1`, the script waits about a second
   between sends; allow it to finish rather than interrupting the sleep. If the TEST manifest or
   sample files are missing, paste the error and stop.
7. Capture current resource usage and tear down the services:
   ```powershell
   docker stats --no-stream
   .\tasks.ps1 down
   ```
   Expected: `docker stats` lists active containers; Compose reports the project stopped. The
   named Prometheus/Grafana volumes are retained because teardown does not request volume removal.
   If a host port is occupied, stop the identified conflicting service and retry.
8. Host MLflow UI (optional local inspection, no Docker service):
   ```powershell
   .\tasks.ps1 mlflow-ui
   ```
   Open `http://127.0.0.1:5000`. This points at the live host `mlflow.db`; do not run host
   training concurrently with another process writing that SQLite store. The database's recorded
   Windows artifact URIs are not established as readable inside Linux containers.

Prometheus and Grafana image tags are pinned in `docker-compose.yml` and linked to their official
tag/installation documentation in `README.md`. If checking tags later, query Docker Hub from
PowerShell with:
`Invoke-RestMethod 'https://hub.docker.com/v2/repositories/prom/prometheus/tags?page_size=100'`
and
`Invoke-RestMethod 'https://hub.docker.com/v2/repositories/grafana/grafana/tags?page_size=100'`.

## WP4 — Real request-log drift monitoring

These steps use the host TRAIN split and checkpoint only to write a derived baseline. No training
or checkpoint write occurs. The baseline command records the actual elapsed time; a trustworthy
runtime estimate cannot be produced in this environment without your data and machine. The
configured sample size is in `configs\config.yaml` and can be changed there before the baseline
is created. The PSI threshold proposal and KS alpha remain provisional. Replay is staged traffic,
not real users.

1. In PowerShell, go to the project and run the focused synthetic tests:
   ```powershell
   Set-Location -LiteralPath "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
   ..\.venv\Scripts\pytest.exe tests\test_drift_monitor.py tests\test_compose_monitoring.py -q
   ```
   Expected: both test modules pass. If a module is missing, confirm this branch's WP4 files are
   present; if an import/dependency error appears, paste the full error.
2. Create the real baseline:
   ```powershell
   ..\.venv\Scripts\python.exe scripts\compute_baseline_features.py
   ```
   Expected: JSON naming `reports\baseline_features.json`, the actual sampled record count, and
   measured `elapsed_seconds`. It reads TRAIN images and the current champion checkpoint. Likely
   failures: missing `data\splits\train.csv`, `data\splits\manifest_hashes.json`, image files,
   `DATA_ROOT` for relative image paths, or `models\best_model.pth`. Fix the missing path/input;
   never replace the champion checkpoint. Paste the command output and the resulting baseline JSON.
3. Start core plus monitoring and check service state:
   ```powershell
   .\tasks.ps1 up-monitoring
   docker compose --profile core --profile monitoring ps -a
   ```
   Expected: API, UI, drift exporter, Prometheus, and Grafana become healthy. If a service is not
   healthy, run `docker compose --profile core --profile monitoring logs --tail 100 api drift-exporter prometheus`
   and paste the relevant error. On a first run, image builds/downloads may take time. If Docker
   cannot mount from the apostrophe-containing project path, use an apostrophe-free checkout.
4. Check the target, current report, and alert endpoint:
   ```powershell
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/targets | Select-Object -ExpandProperty StatusCode
   docker compose --profile core --profile monitoring exec -T drift-exporter python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:9101/metrics').read().decode())"
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/api/v1/alerts | Select-Object -ExpandProperty Content
   Get-Content reports\drift_latest.json
   ```
   Expected: targets page returns HTTP 200; before enough records the JSON says
   `insufficient_data`, `drift_flag` is null, and drift verdict metrics are absent. The scrape
   interval is configured in `monitoring\prometheus.yml`. If port 9101 is unavailable, inspect
   `docker compose ... logs drift-exporter` and resolve the port conflict.
5. Send normal staged TEST replay, then blur replay, then dark replay. Each run sends at least the
   configured minimum sample count; allow the process to finish:
   ```powershell
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 30 --rate 1
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 30 --rate 1 --degrade blur
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 30 --rate 1 --degrade dark
   ```
   Expected: request output/log records identify `traffic_source=replay`, exporter JSON shows a
   sliding window and replay share, and Grafana charts observed series. Normal replay is expected
   to be less shifted than degraded replay; no flag is guaranteed because this depends on the real
   baseline, successful requests, and provisional PSI AND KS rule. An alert appears only if the
   flag remains active for configured `alert_for`. If records are insufficient, use additional
   replay requests and wait for the exporter polling interval. If there is no flag, paste the
   baseline summary, `reports\drift_latest.json`, and replay summaries; do not change thresholds.
6. Open the dashboard at `http://127.0.0.1:3000`, then Prometheus alerts at
   `http://127.0.0.1:9090/alerts`. Save screenshots:
   ```powershell
   New-Item -ItemType Directory -Force docs\evidence\drift
   ```
   Take screenshots of the drift panels and alert state and save them in that directory. If the
   panels show no data, inspect Prometheus `/targets`, the exporter `/metrics`, and the JSON report.
7. Stop the stack:
   ```powershell
   .\tasks.ps1 down
   ```
   Expected: Compose stops the services while named monitoring data volumes remain. Paste back
   the baseline command output, `reports\drift_latest.json`, Prometheus target/alert output, the
   normal/blur/dark replay summaries, and the dashboard/alert screenshots. Do not paste raw image
   data or secret-containing `.env` contents.

## Utility: Large Figure Regeneration

### Regenerating `reports/figures/segmentation_40_grid.png`
- **File**: `reports/figures/segmentation_40_grid.png` (~15.2 MB, untracked by `.gitignore` due to the 5 MB repository limit).
- **Command to regenerate deterministically**:
  ```powershell
  d:
  cd "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
  ..\.venv\Scripts\python.exe scripts/benchmark_segmentation.py
  ```

## T10/T11 — Docker and Monitoring Evidence

The Compose definitions now include the `core` and `monitoring` profiles, Prometheus scrape
configuration, Grafana provisioning, and a dashboard. They have not been built or started in this
environment. Use the WP3 checklist above to validate container startup, target health, dashboard
provisioning, replay traffic, and resource usage on the machine with Docker Desktop and the mounted
champion checkpoint.

  ## DataLoader Worker Probe

  To compare Windows worker counts without training, run:

  ```powershell
  ..\.venv\Scripts\python.exe scripts\benchmark_dataloader_workers.py --workers 0 1 2 --batches 8
  ```

  The output reports elapsed time and whether the first labels are deterministic
  across two runs. Keep `training.num_workers: 0` for the champion until this
  probe shows a stable improvement; worker processes add Windows spawn overhead and
  can complicate reproducibility.
- **Output**: Writes `reports/figures/segmentation_40_grid.png` and `reports/segmentation_quality.json`. Verified reproducible with 0% failure across 40 benchmark samples.
