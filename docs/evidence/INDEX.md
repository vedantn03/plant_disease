# Evidence Inventory

This inventory records files committed with the repository and evidence still needed from the
local machine. A script, dashboard definition, or test proves implementation structure; it is not
runtime evidence by itself.

| Evidence | File / location | What it supports | State |
|---|---|---|---|
| DVC command guidance | `docs/evidence/dvc/README.md` | Local DVC commands and handling notes | Committed documentation; no claim of current remote sync |
| Gate diagnostics transcript | `docs/evidence/gate/val_diagnostics.txt` | Saved command output present in the repository | Committed; re-check before quoting as current |
| Preflight output | `docs/evidence/preflight/preflight_output.txt` | Saved preflight text | Committed; re-check before quoting as current |
| Preflight collection directory | `docs/evidence/preflight/.gitkeep` | Directory placeholder only | No evidence collected |
| Report snapshots | `reports/snapshot/` | Model-card evidence inputs | Sanitized snapshot files are committed; refresh when local source reports change |
| Docker runtime screenshots | `docs/evidence/monitoring/` | Compose, Prometheus and Grafana runtime | Missing from repository |
| Drift screenshots | `docs/evidence/drift/` | Observed replay window and alert state | Missing from repository |
| UI screenshots | `docs/evidence/ui/` | Prediction trust panel and Advice passport | Missing from repository |
| Advisory screenshots | `docs/evidence/advisory/` | Advice tiers and source trace | Missing from repository |
| Promotion lifecycle screenshots/log | `docs/evidence/lifecycle/`, `reports/promotion_log.jsonl` | Existing apply/rollback evidence | Missing unless a real saved entry is available; do not fabricate |
| CI run | GitHub Actions PR checks | Lint, tests, config, build and artifact checks | Per-commit browser evidence not stored here |

Before adding evidence, review each file for `.env` values, absolute local paths, usernames,
weights, raw data, and file-size policy. Keep runtime logs, models, databases, data, and monitoring
volumes out of Git.
