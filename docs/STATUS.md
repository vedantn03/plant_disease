# Project Implementation Status Log (Phase 3)

## Current state (2026-11-03)

| Implemented | Partial | Simulated | Planned |
|---|---|---|---|
| Candidate-only training, VAL gate tooling, three-tab dashboard, Compose monitoring, WP4 baseline/exporter/rule code | WP1/WP4 data-dependent checks and full local Docker verification awaiting user-machine execution; drift thresholds provisional | `scripts/run_airflow_simulation.py`, `scripts/simulate_drift.py`, and `scripts/check_drift_and_trigger.py` outputs are SIMULATION | User approval of proposed PSI threshold; any alias movement; live production deployment |

WP2 and WP3 implementation is in the working tree; WP4 baseline and real-log monitor code is
implemented, but baseline generation, alert behavior, and full Docker verification remain NOT RUN
here. Replay traffic is staged and must not be described as real users.

Candidate selection uses VAL metrics. TEST metrics are for reporting only; seed 44's higher TEST
macro-F1 must not drive selection. The current saved comparison has champion and seed 43 only.
The previously reported per-crop JSON values are affected by the unscoped-label F1 bug; regenerate
from source runs or mark stale. Do not hand-edit those generated files. The historical
`reports/per_crop_metrics_<run8>.json` files are ignored generated artifacts and should not be
committed as part of this package.

## Task T0 — Repository Audit & Baseline Truth Table
- **Status**: COMPLETED
- **What changed**:
  - Full audit conducted without modifying feature code.
  - Measured live test suite results: 23 passed, 1 failed (negative phrasing in HLB advisory triggering substring test).
  - Verified API endpoints `/health`, `/stats`, and `/predict` with test payload.
  - Documented ground truth on serving disconnect (local `.pth` vs MLflow Registry), drift simulation (25 vs 25), near-duplicate sampling, spurious `plantdisease.net` domain, and background shortcut bias prior art (Noyan 2022).
  - Created `docs/AUDIT.md`.
- **Verification Commands & Output**:
  - `..\.venv\Scripts\pytest.exe -q` -> 23 passed, 1 failed in 19.58s
  - `python scripts/train.py --help` -> exit 0, flags documented
  - `GET /health` -> `200 OK`, `GET /stats` -> `200 OK`, `POST /predict` -> `200 OK`
- **Manual / Pending Confirmation**:
  - Awaiting user confirmation of `docs/AUDIT.md` before starting Task T1.
