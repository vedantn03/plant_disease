# Promotion lifecycle commands

```powershell
Set-Location -LiteralPath "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
..\.venv\Scripts\python.exe scripts\sync_gate_spreads.py
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --version 3
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --version 4
..\.venv\Scripts\python.exe scripts\inspect_registry.py
Get-Content models\champion_manifest.json
Get-Content configs\gates\gate_thresholds.yaml
..\.venv\Scripts\python.exe -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('configs/gates/gate_thresholds.yaml').read_bytes()).hexdigest())"
notepad configs\gates\approval.yaml
..\.venv\Scripts\dvc.exe commit train
..\.venv\Scripts\dvc.exe push
..\.venv\Scripts\dvc.exe status -c
git add dvc.lock
git commit --only -m "chore: record existing candidate checkpoints in DVC" -- dvc.lock
git add configs\gates\gate_thresholds.yaml configs\gates\approval.yaml
git commit --only -m "chore: approve gate thresholds" -- configs\gates\gate_thresholds.yaml configs\gates\approval.yaml
..\.venv\Scripts\python.exe scripts\promote.py --apply --candidate-metrics reports\candidate_gate_48ecf14d.json --champion-metrics reports\candidate_gate_comparison.json --candidate-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-run-id fb731412d82544e0abc231b8d7c62f95 --champion-version 2 --version 3 --candidate-checkpoint models\candidates\48ecf14da2d84803a36d6fd108bd6b61.pth
.\tasks.ps1 up-monitoring
$adminToken = (Get-Content .env | Where-Object { $_ -match '^ADMIN_RELOAD_TOKEN=' } | Select-Object -First 1) -replace '^ADMIN_RELOAD_TOKEN=', ''
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/admin/reload -Headers @{ 'X-Admin-Token' = $adminToken }
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
Start-Process "http://127.0.0.1:3000/d/system-health-model-trust/system-health-and-model-trust"
..\.venv\Scripts\python.exe scripts\promote.py --dry-run --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_48ecf14d.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-version 3 --version 4
..\.venv\Scripts\python.exe scripts\promote.py --apply --candidate-metrics reports\candidate_gate_0e2c51ab.json --champion-metrics reports\candidate_gate_48ecf14d.json --candidate-run-id 0e2c51abf0834c248e03c5aef774e291 --champion-run-id 48ecf14da2d84803a36d6fd108bd6b61 --champion-version 3 --version 4 --candidate-checkpoint models\candidates\0e2c51abf0834c248e03c5aef774e291.pth
..\.venv\Scripts\python.exe -c "import json; from src.registry import rollback_champion; result=rollback_champion(dry_run=False); print(json.dumps(result, indent=2)); raise SystemExit(0 if result.get('ok') else 1)"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/admin/reload -Headers @{ 'X-Admin-Token' = $adminToken }
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
Start-Process "http://127.0.0.1:3000/d/system-health-model-trust/system-health-and-model-trust"
Get-Content reports\promotion_log.jsonl
```

# Run an apply command only after its matching dry-run passes and approval/DVC pointer are committed.
# The v4 dry-run and apply require v3 to be the active champion and its report as champion metrics.
# Rollback can select v3 only when its manifest verdict is PASS and its committed DVC pointer verifies.
# If neither dry-run passes, do not approve or apply to force a result; retain the current champion.
