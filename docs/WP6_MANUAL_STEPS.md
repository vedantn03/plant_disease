# WP6 — Promotion, serving reload, and rollback


Run these steps from the project directory. `--dry-run` evaluates VAL reports and may append a
local `reports\promotion_log.jsonl` record; it never calls the registry alias setter. Do not run
`--apply` until you have reviewed the verdict, explicitly approved the threshold file, committed
the approval hash, and committed the candidate DVC pointer. No candidate is assumed to pass.

1. Sync the observed seed ranges from the saved WP1 report, then run focused tests:

   ```powershell
   ..\.venv\Scripts\python.exe scripts\sync_gate_spreads.py
   ..\.venv\Scripts\pytest.exe tests\test_gates.py tests\test_promote_dry_run.py tests\test_promotion_safety.py tests\test_model_loader.py tests\test_admin_reload.py tests\test_wp6_lifecycle.py -v
   ```

   Expected sync output includes `seed_spreads` for all four criteria and `written: true`. If the
   report or a metric range is missing, stop; do not type spread values into the YAML.

2. Run the two VAL dry-runs:

   ```powershell
   ..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --version 3
   ..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --version 4
   ```

   Expected output is JSON with `dry_run: true`, candidate/champion run IDs, `split: VAL`,
   `approval_valid`, the four criterion results, a `PASS` or `REJECT` verdict, and
   `alias_moved: false`. A nonzero exit for a rejected candidate is expected. `Observed seed
   spreads are not populated` means step 1 did not complete; a missing report or metric is an
   input problem, not a reason to loosen the gate. Paste both complete JSON outputs.

3. Inspect the registry and manifest without writing them:

   ```powershell
   ..\.venv\Scripts\python.exe scripts\inspect_registry.py
   Get-Content models\champion_manifest.json
   ```

   Expected: inspection shows current alias/version and read-only artifact source URIs; the
   manifest has an `active_version` and `versions` map. Do not paste `.env` contents.

4. `models\candidates` is already declared as the `train` stage's directory output in `dvc.yaml`.
   That is why per-checkpoint `dvc add` reports an overlap. Capture the existing stage output
   without executing training:

   ```powershell
   ..\.venv\Scripts\dvc.exe commit train
   ..\.venv\Scripts\dvc.exe push
   ..\.venv\Scripts\dvc.exe status -c
   git add dvc.lock
   git commit --only -m "chore: record existing candidate checkpoints in DVC" -- dvc.lock
   ```

   `dvc commit train` records the current `models\candidates` output and updates `dvc.lock`; it
   does not run the stage command. Expected: the candidate directory output no longer has a
   missing-version warning; `dvc push` uploads its cache entries and `dvc status -c` reports the
   cache/remote state. Warnings about other pipeline stages can remain. Then the apply guard checks
   that the selected checkpoint is present in the locked directory manifest and matches its local
   DVC cache object. If `dvc commit train` or `dvc push` fails, stop and paste the full output.
   Do not run `dvc repro` or `dvc repro --force`.

5. Before candidate DVC operations, capture the champion checkpoint hash:

   ```powershell
   $bestModelMd5Before = (Get-FileHash models\best_model.pth -Algorithm MD5).Hash
   Write-Host $bestModelMd5Before
   ```

   After DVC operations, compare it again:

   ```powershell
   $bestModelMd5After = (Get-FileHash models\best_model.pth -Algorithm MD5).Hash
   Write-Host $bestModelMd5After
   if ($bestModelMd5Before -ne $bestModelMd5After) { throw "Champion checkpoint changed; stop." }
   ```

   Both outputs must match the user-supplied champion MD5 `7203474FB41FEDAA504A0033FC13A6D3`.
   If either differs, stop; do not overwrite or restore it from a candidate.

6. Threshold approval remains your decision. To prepare an approval, first review the current
   threshold file. Compute its SHA-256 only after the final review, edit `approval.yaml` with your
   own approver and date, then commit both files. Do not set `approved: true` before that decision:

   ```powershell
   Get-Content configs\gates\gate_thresholds.yaml
   ..\.venv\Scripts\python.exe -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('configs/gates/gate_thresholds.yaml').read_bytes()).hexdigest())"
   ```

   Copy that command's output to `thresholds_sha256` in `configs\gates\approval.yaml` only if you
   approve the thresholds. Any later threshold edit invalidates that hash and `--apply` refuses.
   Commit the approval only after consciously setting `approved: true`, `approved_by`,
   `approved_on`, and the hash:

   ```powershell
   git add configs\gates\gate_thresholds.yaml configs\gates\approval.yaml
   git commit --only -m "chore: approve promotion gate thresholds" -- configs\gates\gate_thresholds.yaml configs\gates\approval.yaml
   ```

7. Only after a candidate has a dry-run `PASS`, the threshold approval is committed, and its DVC
   pointer is committed, choose exactly one matching apply command:

   ```powershell
   ..\.venv\Scripts\python.exe scripts\promote.py --apply --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 3 --candidate-checkpoint models\candidates\48ecf14da2d84803a36d6fd108bd6b61.pth
   ..\.venv\Scripts\python.exe scripts\promote.py --apply --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 4 --candidate-checkpoint models\candidates\0e2c51abf0834c248e03c5aef774e291.pth
   ```

   Expected JSON records `alias_moved: true`, previous and new versions, checkpoint MD5, and gate
   verdict. A gate reject, missing approval, hash mismatch, missing registry version, run-ID
   mismatch, missing DVC directory entry/cache object, or manifest/alias mismatch refuses before alias movement. This
   step writes the MLflow alias and manifest; do not run it without an explicit decision.

8. Start the Compose API and monitoring stack so the same serving process is visible in Grafana:

   ```powershell
   .\tasks.ps1 up-monitoring
   ```

   In another PowerShell window, read the token from ignored `.env` and send the request to the
   loopback-bound API port:

   ```powershell
   $adminToken = (Get-Content .env | Where-Object { $_ -match '^ADMIN_RELOAD_TOKEN=' } | Select-Object -First 1) -replace '^ADMIN_RELOAD_TOKEN=', ''
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/admin/reload -Headers @{ 'X-Admin-Token' = $adminToken }
   Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
   Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
   ```

   Expected reload response reports the new `model_version`, `model_source`, MD5 prefix, and
   `registry_match: true`; `/health` agrees. After the configured Prometheus scrape interval,
   Grafana's `model_info` panel reports the new version label. A 401 means the token is missing or
   incorrect; a 503 means alias/manifest/checkpoint verification failed. The API's published port
   is bound to loopback; Docker Desktop can forward that request through a private bridge address.

9. `.\tasks.ps1 rollback` is an alias and manifest write. It prints the before/after versions
   and refuses when there is no earlier version marked `PASS` with a valid DVC pointer. The current
   v2 baseline is not automatically labelled `PASS`; with no earlier passing promotion, refusal is
   the expected safe outcome. If rollback succeeds, repeat the host-run reload request and check
   `/health` plus the Grafana `model_info` panel. Save lifecycle screenshots in
   `docs\evidence\lifecycle\`. Paste the two dry-run JSON results, any apply/rollback result you
   chose to run, and the resulting health JSON; never paste the token.
