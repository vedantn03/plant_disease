# Demo freeze checklist

Run these commands from Windows PowerShell 5.1 in the repository root. They are a manual release
checklist; they were **not run in this cloud workspace**. Inspect command output and stop on the
first failure. Do not run training or `dvc repro` as part of the freeze.

## Verify source and tests

```powershell
Set-Location -LiteralPath "D:\VIT\Hackathons\SIH'26\Airafare-proj\plant_disease"
git status --short --branch
..\.venv\Scripts\python.exe -m pytest tests\ -q -rs
..\.venv\Scripts\pre-commit.exe run --all-files
..\.venv\Scripts\python.exe scripts\gen_model_card.py --check
```

Expected: the worktree is clean after the intended freeze commit, pytest exits successfully, all
pre-commit hooks pass, and the model-card check reports no stale generated content. If pytest skips
data/model-dependent checks, retain and report the `-rs` skip reasons; do not describe those checks
as executed.

## Review DVC state and upload tracked cache

```powershell
..\.venv\Scripts\dvc.exe status -c
..\.venv\Scripts\dvc.exe push
..\.venv\Scripts\dvc.exe status -c
```

Expected: the first and last commands report the configured cache/remote state; `dvc push` uploads
missing tracked objects. Review warnings and remote identity before pushing. The repository
configuration names `localremote` with the owner-machine path `D:\dvc_remote`; that machine-local
remote is separate from GitHub. Do not run `dvc repro` for this freeze.

## Verify the champion checkpoint without modifying it

```powershell
$manifest = Get-Content models\champion_manifest.json -Raw | ConvertFrom-Json
$expected = $manifest.versions.'2'.md5.ToLowerInvariant()
$actual = (Get-FileHash models\best_model.pth -Algorithm MD5).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Champion MD5 mismatch: expected $expected, got $actual" }
$actual
```

Expected: the command prints the matching MD5 from the committed version-2 manifest. This reads the
checkpoint only; it does not rewrite it. If the checkpoint is unavailable, stop and restore it
through the owner’s established DVC procedure before continuing.

## Scan tracked files for secrets and absolute paths

```powershell
..\.venv\Scripts\pre-commit.exe run detect-secrets --all-files
git grep -n -I -E '[A-Za-z]:[/\\]|/home/[^/]+/|/Users/[^/]+/' -- .
git status --short
```

Expected: secret scanning passes, and the path scan has no unintended machine-specific paths.
Review every match: documentation may intentionally mention the configured DVC path, and tests may
contain path examples. Do not add broad allowlists to silence a match. Ensure `.env`, `mlflow.db`,
model weights, local data, logs, and DVC cache are not staged.

## Create the source backup

The archive below copies tracked source files only and excludes `models\`, `mlruns\`, `data\`,
`logs\`, `.dvc\cache\`, `.env`, and `mlflow.db`. It does not back up the separate DVC remote or
the MLflow database. The database configured by this repository is `sqlite:///mlflow.db` at the
project root; back it up separately while the owning application is stopped, using the owner’s
approved local backup procedure.

```powershell
$backupRoot = Join-Path $env:TEMP ("plant_disease_freeze_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$zipPath = Join-Path (Get-Location) "plant_disease_source_freeze.zip"
New-Item -ItemType Directory -Path $backupRoot | Out-Null
git ls-files | Where-Object { $_ -notmatch '^(models|mlruns|data|logs|\.dvc/cache)(/|$)' -and $_ -notmatch '(^|/)\.env($|\.)' -and $_ -ne 'mlflow.db' } | ForEach-Object { $source = Join-Path (Get-Location) $_; $target = Join-Path $backupRoot $_; New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null; Copy-Item -LiteralPath $source -Destination $target }
Compress-Archive -Path (Join-Path $backupRoot '*') -DestinationPath $zipPath
Get-Item -LiteralPath $zipPath | Select-Object FullName,Length
```

Expected: `plant_disease_source_freeze.zip` is created in the repository root. Keep it outside
Git. The archive excludes all model files, including the committed manifest and DVC pointers,
because the entire `models\` directory is excluded; DVC remains the artifact backup mechanism.

## Tag and push the demo freeze

Create the tag only after review and merge of the final PR, and only after the checks above pass.

```powershell
git switch main
git pull --ff-only origin main
git tag -a v1.0-demo -m "demo freeze"
git push origin v1.0-demo
```

Expected: Git creates and pushes the annotated tag. This does **not** publish an image. The
repository’s `.github/workflows/publish-api.yml` has exactly the `workflow_dispatch` trigger. Its
job additionally requires the manually selected ref to be a tag beginning with `v` and the
`publish` input to be `true`; pushing `v1.0-demo` alone cannot start the publish job. To publish,
manually open GitHub **Actions → Publish API image → Run workflow**, choose the `v1.0-demo` tag, set
`publish` to `true`, and confirm. The workflow publishes only the API image to
`ghcr.io/vedantn03/plant_disease`; it does not include model weights or dataset files.
