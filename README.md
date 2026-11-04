# Plant Disease MLOps Pipeline

Academic prototype for leaf-image classification, evidence-aware advice, model governance, and
local monitoring. It is not validated for field diagnosis or treatment decisions. See
[`docs/STATUS.md`](docs/STATUS.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), and
[`docs/ADVISORY_AUDIT.md`](docs/ADVISORY_AUDIT.md).

## Local quickstart

Run PowerShell commands from the repository root. The local Python environment and model files
are user-managed; no dataset, checkpoint, `.env`, MLflow database, or DVC cache belongs in Git.

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
docker compose --profile core config --quiet
.\tasks.ps1 up-core
```

Set local Grafana credentials in `.env` before monitoring. Then:

```powershell
docker compose --profile core --profile monitoring config --quiet
.\tasks.ps1 up-monitoring
```

Open API health at `http://localhost:8000/health`, Streamlit at `http://localhost:8501`,
Prometheus targets at `http://localhost:9090/targets`, and Grafana at `http://localhost:3000`.
The API needs the mounted model checkpoint and registry configuration to serve predictions. If
Docker Desktop is stopped, start its Linux engine before running these commands.

To stop the stack without deleting persistent monitoring data:

```powershell
.\tasks.ps1 down
```

## Development checks

```powershell
..\.venv\Scripts\python.exe -m pytest tests/ -q -rs
..\.venv\Scripts\pre-commit.exe run --all-files
..\.venv\Scripts\python.exe scripts\gen_model_card.py --check
```

Some tests skip when local data or model artifacts are not present. CI exercises API plumbing
with a temporary fixture model; that is not an accuracy evaluation.

## Current implementation status

| Area | Status | Scope |
|---|---|---|
| Data and DVC workflow | Implemented | Pipeline code exists; local data/remote execution is machine-specific. |
| Training and evaluation | Partial | Code exists; no training is performed by the quickstart. |
| Promotion governance | Implemented | Dry-run and approval controls are present; alias changes require operator action. |
| API, UI, and local Compose | Implemented | Runtime requires the local checkpoint and environment configuration. |
| Drift monitoring | Partial | Real request logs are supported; replay is staged traffic and field drift is not established. |
| Airflow | Simulated | The script output is labelled SIMULATION. |
| Managed SageMaker | Planned | Local pipeline compilation only; no managed execution is claimed. |
| Advice provenance | Partial | Most KB entries are placeholders; see the source audit. |

## Known limitations

- Marathi and Hindi selectors translate generic explanation text. These translations are drafts
  requiring fluent review; cited advisory text remains in its source language.

- Evaluation images are lab-style; field performance has not been validated.
- Background-fill results vary with the fill and segmentation. Black-fill drops include masking
  artifacts and are not proof of shortcut learning.
- Corn Cercospora / Gray leaf spot is a known weaker class; use saved evaluation artifacts for
  current numeric claims.
- The API serves unscaled softmax confidence at T=1. See `docs/DECISIONS.md` for the rationale.
- Advisory content is provenance-gated. Do not treat placeholders as agronomic guidance.
- Verify the dataset license and redistribution rights before public release.

## Documentation and evidence

- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Rehearsal commands: [`docs/REHEARSAL.md`](docs/REHEARSAL.md)
- Demo script: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)
- Viva Q&A: [`docs/VIVA_QA.md`](docs/VIVA_QA.md)
- Evidence inventory: [`docs/evidence/INDEX.md`](docs/evidence/INDEX.md)
- Model card and datasheet inputs: committed, sanitized files in `reports/snapshot/`. Refresh them
  with the snapshot commands below when local reports change; review generated files before commit.

To prepare local evidence snapshots, use the read-only sanitized registry export, collect the
available reports, then regenerate and check the card:

```powershell
..\.venv\Scripts\python.exe scripts\inspect_registry.py --snapshot reports\registry_export.json
..\.venv\Scripts\python.exe scripts\snapshot_reports.py
..\.venv\Scripts\python.exe scripts\gen_model_card.py
..\.venv\Scripts\python.exe scripts\gen_model_card.py --check
```

The snapshot command omits unavailable inputs from its copy list, redacts local paths/usernames,
and rejects files above the pre-commit configured limit. Review generated snapshots before
committing them; do not stage raw model files, databases, logs, or local data.
