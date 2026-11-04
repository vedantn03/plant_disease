# Demo Script

This is a rehearsal guide, not a claim that the local machine or runtime was exercised in this
checkout. Only quote values visible in current report files or live responses.

| Step | What to show | Command or action | Expected visible evidence | My time |
|---|---|---|---|---|
| 1 | Clean Docker start | `docker compose --profile core --profile monitoring down -v` then `docker compose --profile core --profile monitoring build` | Services are rebuilt; `down -v` deletes named Prometheus/Grafana data volumes. | |
| 2 | Start core UI/API | `.\tasks.ps1 up-core` | API health and Streamlit UI respond. | |
| 3 | Trust panel and advice passport | Open `http://localhost:8501`; upload one available leaf image | Show returned label, confidence, quality/OOD fields, background sensitivity and advice trace; say “not available” when response lacks a field. | |
| 4 | Background sensitivity comparison | Use the same leaf under original and mean-fill display in Diagnostic Screening | Compare the two returned predictions and confidence changes; do not claim this proves shortcut learning. | |
| 5 | Monitoring | `.\tasks.ps1 up-monitoring`; open `http://localhost:9090/targets` and `http://localhost:3000` | Prometheus scrape targets and the provisioned dashboard. | |
| 6 | Clean replay control | `..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1` | Requests are marked `traffic_source=replay`; inspect current drift report and alert state rather than promising no drift. | |
| 7 | Staged drift example | Repeat replay with `--degrade blur`, then `--degrade dark` | Show measured window/alert response; replay is staged traffic, not real users or field data. | |
| 8 | Promotion dry-run | Run the v3/v4 commands in `docs/REHEARSAL.md` | Gate verdict and reasons from the current reports; dry-run does not move the alias. | |
| 9 | Promotion/rollback evidence | Open `reports/promotion_log.jsonl` if present | Show only existing entries. If no apply/rollback entry exists, say “not captured”; do not perform a registry write during this demo. | |
| 10 | DVC and CI evidence | `..\.venv\Scripts\dvc.exe status -c`; open the GitHub PR checks | Show command output and actual CI state. | |

## Timing checklist

Fill in the “My time” column during rehearsal. Do not publish estimated timings as measured
runtime. If Docker is cold or the checkpoint is unavailable, use the fallback steps in
[`REHEARSAL.md`](REHEARSAL.md).
