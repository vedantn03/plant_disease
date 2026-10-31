<#
.SYNOPSIS
    PowerShell Task Runner for Plant Disease MLOps Pipeline.
    Mirrors all Makefile targets on Windows environments where GNU make is absent.

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 verify-gpu
    .\tasks.ps1 train
    .\tasks.ps1 serve
    .\tasks.ps1 ui
    .\tasks.ps1 evaluate
    .\tasks.ps1 promote
    .\tasks.ps1 test
    .\tasks.ps1 drift
    .\tasks.ps1 up
    .\tasks.ps1 down
    .\tasks.ps1 clean
#>

param(
    [Parameter(Position=0, Mandatory=$false)]
    [string]$Target = "help",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$PYTHON    = "..\.venv\Scripts\python.exe"
$PIP       = "..\.venv\Scripts\pip.exe"
$PYTEST    = "..\.venv\Scripts\pytest.exe"
$UVICORN   = "..\.venv\Scripts\uvicorn.exe"
$STREAMLIT = "..\.venv\Scripts\streamlit.exe"

function Show-Help {
    Write-Host "Plant Disease Pipeline Task Runner (tasks.ps1)" -ForegroundColor Cyan
    Write-Host "Available targets:" -ForegroundColor Yellow
    Write-Host "  preflight   - Run system environment and readiness check"
    Write-Host "  setup       - Install PyTorch with CUDA 12.8 and pip requirements"
    Write-Host "  verify-gpu  - Verify CUDA GPU availability via PyTorch"
    Write-Host "  audit       - Audit dataset, check hashes, and generate group splits"
    Write-Host "  train       - Train ResNet-18 classifier (8 epochs, AMP)"
    Write-Host "  candidate   - Train a candidate with an explicit seed; never promotes"
    Write-Host "  serve       - Start FastAPI backend directly on port 8000"
    Write-Host "  ui          - Start Streamlit dashboard on port 8501"
    Write-Host "  evaluate    - Run comprehensive evaluation suite and generate reports"
    Write-Host "  promote     - Execute MLflow Model Registry promotion gate"
    Write-Host "  rollback    - Rollback champion alias to previous model version in registry"
    Write-Host "  test        - Run pytest test suite"
    Write-Host "  drift       - Execute dataset drift simulation"
    Write-Host "  up          - Start core stack using Docker Compose"
    Write-Host "  down        - Stop Docker Compose stack"
    Write-Host "  clean       - Clean local run caches, logs, and temp report files"
}

switch ($Target.ToLower()) {
    "preflight" {
        & powershell.exe -ExecutionPolicy Bypass -File scripts\preflight.ps1
    }
    "setup" {
        Write-Host "Installing dependencies..." -ForegroundColor Cyan
        & $PIP install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
        & $PIP install -r requirements.txt
    }
    "verify-gpu" {
        Write-Host "Verifying CUDA device..." -ForegroundColor Cyan
        & $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU OK:', torch.cuda.get_device_name(0))"
    }
    "audit" {
        Write-Host "Running audit & group-aware split..." -ForegroundColor Cyan
        & $PYTHON scripts/audit_and_split.py
    }
    "train" {
        Write-Host "Starting candidate training; models\best_model.pth is protected..." -ForegroundColor Cyan
        & $PYTHON scripts/train.py
    }
    "candidate" {
        $champion = "models\best_model.pth"
        $before = if (Test-Path $champion) { (Get-FileHash $champion -Algorithm MD5).Hash } else { "MISSING" }
        Write-Host "Champion MD5 before: $before"
        & $PYTHON scripts/train.py --candidate --seed $Seed
        $after = if (Test-Path $champion) { (Get-FileHash $champion -Algorithm MD5).Hash } else { "MISSING" }
        Write-Host "Champion MD5 after:  $after"
        if ($before -ne $after) {
            throw "Safety check failed: models\best_model.pth changed."
        }
        Write-Host "Candidate completed; champion checkpoint unchanged." -ForegroundColor Green
    }
    "serve" {
        Write-Host "Starting FastAPI backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
        & $UVICORN src.api:app --host 127.0.0.1 --port 8000
    }
    "ui" {
        Write-Host "Starting Streamlit dashboard on http://localhost:8501 ..." -ForegroundColor Cyan
        & $STREAMLIT run app/ui.py --server.port 8501
    }
    "evaluate" {
        Write-Host "Running model evaluation suite..." -ForegroundColor Cyan
        & $PYTHON scripts/evaluate.py
    }
    "promote" {
        Write-Host "Evaluating promotion gate (dry-run; provisional thresholds)..." -ForegroundColor Cyan
        & $PYTHON scripts/promote.py --dry-run
    }
    "rollback" {
        Write-Host "Executing model rollback..." -ForegroundColor Cyan
        & $PYTHON -c "from src.registry import rollback_champion; rollback_champion()"
    }
    "test" {
        Write-Host "Running pytest test suite..." -ForegroundColor Cyan
        & $PYTEST tests/ -v
    }
    "drift" {
        Write-Host "Running drift simulation..." -ForegroundColor Cyan
        & $PYTHON scripts/simulate_drift.py
    }
    "up" {
        Write-Host "Starting Docker Compose stack..." -ForegroundColor Cyan
        docker compose up --build -d
        Write-Host "API: http://localhost:8000/health"
        Write-Host "UI:  http://localhost:8501"
    }
    "down" {
        Write-Host "Stopping Docker Compose stack..." -ForegroundColor Cyan
        docker compose down
    }
    "clean" {
        Write-Host "Cleaning temporary artifacts..." -ForegroundColor Cyan
        if (Test-Path "mlflow.db") { Remove-Item "mlflow.db" -Force }
        if (Test-Path "logs") { Remove-Item "logs" -Recurse -Force }
        Get-ChildItem -Path "reports" -Filter "*.png" | Remove-Item -Force
        Get-ChildItem -Path "reports" -Filter "*.html" | Remove-Item -Force
        Write-Host "Clean complete." -ForegroundColor Green
    }
    default {
        Show-Help
    }
}
