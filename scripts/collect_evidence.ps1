<#
.SYNOPSIS
    Collect read-only repository, environment, and process evidence.

.DESCRIPTION
    Writes command output below docs\evidence\collect\<timestamp>\. It does not
    edit source files, configuration, model weights, DVC state, or MLflow state.
    Run from the repository root in Windows PowerShell 5.1.
#>

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputRoot = Join-Path "docs\evidence\collect" $timestamp
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$summaryPath = Join-Path $outputRoot "summary.txt"

function Write-Summary([string]$Text) {
    Add-Content -Path $summaryPath -Value $Text
}

function Invoke-Captured([string]$Name, [scriptblock]$Command) {
    $path = Join-Path $outputRoot "$Name.txt"
    Write-Summary "`n===== $Name ====="
    try {
        & $Command *>&1 | Tee-Object -FilePath $path
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        Write-Summary "exit_code=$exitCode"
    }
    catch {
        $_ | Tee-Object -FilePath $path
        Write-Summary "exception=$($_.Exception.Message)"
    }
}

Write-Summary "Evidence collection timestamp: $timestamp"
Write-Summary "Repository: $(Get-Location)"

Invoke-Captured "01_git_branch_status_log" {
    git branch --show-current
    git status --short --branch
    git log --oneline --graph --decorate -10
}

Invoke-Captured "02_pytest" {
    & "..\.venv\Scripts\pytest.exe" tests\ -q -p no:cacheprovider
}

Invoke-Captured "03_history_sensitive_extensions" {
    git log --all --name-only --pretty=format: |
        Select-String -Pattern '(^|/)(\.env|.*\.pth|.*\.pt|.*\.db)$'
}

Invoke-Captured "04_local_paths" {
    git grep -n -I -E 'D:\\|C:\\Users' -- .
}

Invoke-Captured "05_secret_like_patterns" {
    git grep -n -I -E '(api[_-]?key|secret|password|token|BEGIN [A-Z ]+ PRIVATE KEY)' -- .
}

Invoke-Captured "06_dvc_status" {
    & "..\.venv\Scripts\dvc.exe" status -c
}

Invoke-Captured "07_champion_md5" {
    if (Test-Path "models\best_model.pth") {
        Get-FileHash "models\best_model.pth" -Algorithm MD5
    }
    else {
        Write-Output "models\best_model.pth is missing"
    }
}

Invoke-Captured "08_docker_version" {
    docker version
}

Invoke-Captured "09_preflight" {
    powershell.exe -ExecutionPolicy Bypass -File scripts\preflight.ps1
}

Invoke-Captured "10_ports_8000_8501" {
    $connections = Get-NetTCPConnection -LocalPort 8000,8501 -State Listen -ErrorAction SilentlyContinue
    if ($null -eq $connections) {
        Write-Output "No listeners found on ports 8000 or 8501"
    }
    else {
        $connections |
            Select-Object LocalAddress,LocalPort,OwningProcess |
            ForEach-Object {
                $process = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
                [PSCustomObject]@{
                    LocalAddress = $_.LocalAddress
                    LocalPort = $_.LocalPort
                    PID = $_.OwningProcess
                    ProcessName = if ($process) { $process.ProcessName } else { "unknown" }
                }
            } | Format-Table -AutoSize
    }
}

Write-Summary "`nFiles written under: $outputRoot"
Write-Output "Evidence collected under $outputRoot"
