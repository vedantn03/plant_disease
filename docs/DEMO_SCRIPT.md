# Demo Script

This is a run sheet, not evidence that a local runtime was exercised. Fill the timing column only
with times measured during your rehearsal. Replay is staged traffic; do not describe it as real
farmer or field traffic.

| Step | What to show | Command or action | Evidence to report | My time |
|---|---|---|---|---|
| 1 | Check the Compose configuration | `docker compose --profile core config` | Successful configuration output from this machine. | |
| 2 | Build and start the UI and API | `docker compose --profile core build ui` then `docker compose --profile core up -d` | Container status from `docker compose --profile core ps`. | |
| 3 | API readiness | Open `http://127.0.0.1:8000/health` | Actual health response, including model source/version and registry match. | |
| 4 | Farmer-facing screening and explanation | Open `http://127.0.0.1:8501`, upload an available leaf image | Returned label/confidence, trust panel, Advice passport, API Grad-CAM if returned; SHAP/LIME saved figures are offline comparisons, not generated for this upload. | |
| 5 | Language selector | Select English, Marathi, then Hindi | Marathi/Hindi currently fall back to English until reviewed translations are added. | |
| 6 | Monitoring | `docker compose --profile core --profile monitoring up -d`; open Prometheus `/targets` and Grafana `http://127.0.0.1:3000` | Actual target health and dashboard values. | |
| 7 | Clean replay control | `..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1` | Confirm `traffic_source=replay`; report observed drift state without assuming no alert. | |
| 8 | Staged degraded replay | Repeat with `--degrade blur` and `--degrade dark` | Report observed metrics/alert state; this is staged traffic, not field validation. | |
| 9 | Candidate gate dry-run | Run the v3 and v4 dry-run commands in `docs/REHEARSAL.md` | Actual gate verdict/reasons. Dry-run does not move the registry alias. | |
| 10 | Saved promotion evidence | Inspect `reports/promotion_log.jsonl` if present | Show only existing entries. If no apply/rollback entry exists, state that it was not captured. | |
| 11 | Stop the stack | `docker compose --profile core --profile monitoring down` | Containers stopped; named monitoring volumes retained. | |

The previous Docker image-export attempt failed with an I/O error in Docker Desktop storage. If the
same error occurs, report it as an environment failure and follow the non-destructive checks in
[`REHEARSAL.md`](REHEARSAL.md). The supplied “no matching manifest for ...” line was explicitly an
example and is not recorded as an observed failure.
