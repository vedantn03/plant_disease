# Viva cheatsheet

## Fifteen facts to know

| # | Fact | Source |
|---|---|---|
| 1 | Test accuracy: **0.993** | `reports/snapshot/eval_summary.json` |
| 2 | Test macro-F1: **0.9903** | `reports/snapshot/eval_summary.json` |
| 3 | Test weighted F1: **0.993** | `reports/snapshot/eval_summary.json` |
| 4 | Test set: **8,134** images | `reports/snapshot/eval_summary.json` |
| 5 | Within-crop macro-F1, single-class crops excluded: **0.9888** | `reports/snapshot/eval_summary.json` |
| 6 | Crop F1 now scopes labels to that crop’s true classes | `scripts/train.py`; `tests/test_wp1_close_gate.py` |
| 7 | Lowest-recall saved test class: Corn Cercospora; recall **0.8701**, support **77** | `reports/snapshot/eval_summary.json` |
| 8 | Three-seed within-crop macro-F1 range: **0.002124** | `reports/snapshot/seed_variance.json` |
| 9 | Three-seed minimum-recall range: **0.025974** | `reports/snapshot/seed_variance.json` |
| 10 | Per-image-mean bg-drop, champion / seed 43 / seed 44: **0.0364 / 0.0163 / 0.0570** | `reports/snapshot/seed_variance.json` |
| 11 | Black-fill bg-drop, same model order: **0.1843 / 0.1797 / 0.1524** | `reports/snapshot/seed_variance.json` |
| 12 | Mid-grey bg-drop, same model order: **0.0938 / 0.0563 / 0.0997** | `reports/snapshot/seed_variance.json` |
| 13 | Background-sensitivity VAL sample: **100**; consistent **87**, inconsistent **9**, unknown **4** | `reports/snapshot/bg_check_rates.json` |
| 14 | Saved drift window: **200**; replay share **0.765**; flag **1** | `reports/snapshot/drift_latest.json` |
| 15 | Advice entries: **1 VERIFIED**, **37 PLACEHOLDER** | `reports/snapshot/advisory_audit.json` |

## Ten claims not to make

1. No field validation — `docs/DATASHEET.md`, `docs/STATUS.md`.
2. PlantVillage score does not prove field performance — `reports/snapshot/eval_summary.json`, `docs/STATUS.md`.
3. Background sensitivity does not prove shortcut learning — `docs/DECISIONS.md`.
4. Drift was not measured on real users; saved window is staged replay — `reports/snapshot/drift_latest.json`.
5. No clean-control or captured alert evidence — `reports/snapshot/drift_latest.json`, `docs/evidence/INDEX.md`.
6. No live Airflow DAG execution; script is a simulation — `scripts/run_airflow_simulation.py`.
7. No managed SageMaker execution; status is planned/local compilation — `docs/STATUS.md`.
8. Dataset licence is unverified — `docs/DATASHEET.md`, `docs/STATUS.md`.
9. Old per-crop reports are not corrected reports — `scripts/train.py`, `tests/test_wp1_close_gate.py`.
10. No real promotion or rollback; saved entries have `alias_moved: false` — `reports/snapshot/promotion_log.jsonl`.
