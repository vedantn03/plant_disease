# WP8 advisory verification checklist

Run these steps from the project root in Windows PowerShell. Do not run training or `dvc repro`.

## Generate the VAL-only confidence candidate table

1. Run the diagnostics script to save the champion's per-image VAL predictions. It also runs the existing diagnostics, including TEST reporting; the confidence table itself is constructed only from VAL labels and VAL softmax confidence.

   ```powershell
   ..\.venv\Scripts\python.exe scripts\compute_val_diagnostics.py
   ```

   Expected: the script completes and creates `reports\val_predictions.csv`. If it cannot find DATA_ROOT, the split manifests, or the champion checkpoint, check `.env`, `data\splits\val.csv`, and `models\best_model.pth`; do not substitute TEST values.

2. Derive the table from that saved VAL file.

   ```powershell
   ..\.venv\Scripts\python.exe scripts\derive_confidence_tiers.py
   ```

   Expected: rows by predicted crop, candidate quantile, observed confidence cutoff, precision, coverage, and accepted/total; also `input_sha256`, `table_sha256`, and output path. These are candidate operating points only. Keep `configs\config.yaml` cutoffs empty until you choose values. Paste the full table and both hashes for review.

   If it says the input CSV is missing, confirm step 1 produced `reports\val_predictions.csv`; do not hand-create values. If required-column validation fails, paste the CSV header and error.

## Create confusion hints and check sources

3. Convert only the saved evaluation summary's actual confusion pairs into a CSV for the API to read.

   ```powershell
   ..\.venv\Scripts\python.exe scripts\export_confused_pairs.py
   ```

   Expected: a count of copied pairs and a message that no distinguishing text was generated. If `reports\eval_summary.json` is absent or lacks `top_10_confused_pairs`, paste the error and do not invent pairs. The hints state only “May be confused with …”; no symptom comparison is shown without a cited passage.

4. Open the source links in `docs\ADVISORY_AUDIT.md`. Check five candidate records against their actual page title and cited passage. Only one entry currently remains VERIFIED; four additional entries must not be promoted until their source text is confirmed. For each confirmed entry, record publisher, access date, passage location, and your reviewer identity. Leave failed or mismatched sources as placeholders.

## Run the API and inspect advice

5. Rebuild and start the core stack after the branch changes and reports are ready.

   ```powershell
   .\tasks.ps1 up-core
   ```

   Expected: API and Streamlit start. A Docker/model startup failure should be checked with `docker compose --profile core ps -a` and `docker compose --profile core logs --tail 80 api`; confirm the mounted `models\` folder contains the configured checkpoint.

6. Check health and metrics after an actual prediction.

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
   ```

   ```powershell
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
   ```

   Expected after requests: real observed `advice_total`, `advice_withheld_total`, and `advice_gap_total` series. Before a matching request, a counter may have no series; no placeholder series are emitted. Paste `/health` and the three advice metric lines.

7. Open `http://127.0.0.1:8501`, then in **Diagnostic Screening** upload and diagnose one clear diseased-leaf image, one blurry image, and one non-leaf image. Inspect the **Advice passport**: tier, evidence grade, source URL/date, KB version, rules fired, and background-sensitivity result must match the API response. A placeholder or missing source should not show chemical options. Save screenshots in `docs\evidence\advisory\`.

8. Send marked TEST replay traffic, first unchanged and then blurred.

   ```powershell
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1
   ```

   ```powershell
   ..\.venv\Scripts\python.exe scripts\replay_traffic.py --limit 20 --rate 1 --degrade blur
   ```

   Expected: each response is marked `traffic_source=replay`; the request log preserves the source and `kb_version`. These are staged replay requests, not real-user traffic.

9. Open `http://127.0.0.1:3000/d/system-health-model-trust/system-health-and-model-trust` and inspect the **Advice coverage** panels after replay. For class frequency, run:

   ```powershell
   ..\.venv\Scripts\python.exe scripts\advisory_backlog.py
   ```

   Expected: a report path and the number of observed classes in the backlog. The report separates replay request counts from other logged requests.

10. Save dashboard screenshots in `docs\evidence\advisory\` and stop the stack when finished.

   ```powershell
   .\tasks.ps1 down
   ```
