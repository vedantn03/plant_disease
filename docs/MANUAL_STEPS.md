# Manual Action Steps Log (Master Plan v2)

This document mirrors all `🛑 MANUAL ACTION REQUIRED` blocks raised during the lifecycle implementation.

---

## Task T2 — Preflight, Runner, Hooks & Resource Budget

### Step M2.1: Pre-commit Installation & Preflight Execution
- **Why**: Verifies preflight diagnostics directly on your active PowerShell console and activates git pre-commit hooks to block large files (>5MB) and secrets from ever entering git history.
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

## Utility: Large Figure Regeneration

### Regenerating `reports/figures/segmentation_40_grid.png`
- **File**: `reports/figures/segmentation_40_grid.png` (~15.2 MB, untracked by `.gitignore` due to the 5 MB repository limit).
- **Command to regenerate deterministically**:
  ```powershell
  d:
  cd "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
  ..\.venv\Scripts\python.exe scripts/benchmark_segmentation.py
  ```
- **Output**: Writes `reports/figures/segmentation_40_grid.png` and `reports/segmentation_quality.json`. Verified reproducible with 0% failure across 40 benchmark samples.
