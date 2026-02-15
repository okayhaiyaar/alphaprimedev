param(
    [ValidateSet("auto", "online", "offline")]
    [string]$Mode = "auto",
    [ValidateSet("core", "api", "full")]
    [string]$Tier = "core",
    [bool]$StartUI = $true,
    [bool]$StartAPI = $false,
    [int]$PortUI = 8501,
    [int]$PortAPI = 8000,
    [bool]$Mock = $true
)

function Set-EnvValueInFile {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $line = "$Key=$Value"
    if (!(Test-Path $Path)) {
        Set-Content -Path $Path -Value $line
        return
    }

    $content = Get-Content -Path $Path
    $pattern = "^$([Regex]::Escape($Key))="
    $updated = $false

    for ($i = 0; $i -lt $content.Length; $i++) {
        if ($content[$i] -match $pattern) {
            $content[$i] = $line
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $content += $line
    }

    Set-Content -Path $Path -Value $content
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
$ErrorActionPreference = "Stop"

if (!(Test-Path "app.py") -or !(Test-Path "requirements.txt") -or !(Test-Path "scripts/selfcheck.py")) {
    throw "Run this script from the repository root (expected app.py, requirements.txt, scripts/selfcheck.py)."
}

if ($StartAPI -and $Tier -eq "core") {
    Write-Host "StartAPI requested; upgrading tier from core to api." -ForegroundColor Yellow
    $Tier = "api"
}

$requirementsMap = @{
    core = @("requirements.txt")
    api  = @("requirements.txt", "requirements-api.txt")
    full = @("requirements-full.txt")
}
$checkModeMap = @{
    core = ""
    api  = "--api"
    full = "--full"
}

$selectedMode = $Mode
if ($Mode -eq "auto") {
    if (Test-Path "wheelhouse") {
        $selectedMode = "offline"
    } else {
        $selectedMode = "online"
    }
}

Write-Host "==> Bootstrap mode: $selectedMode (tier=$Tier)"

if (!(Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment (.venv)"
    python -m venv .venv
}

$python = Join-Path ".venv" "Scripts\python.exe"
if (!(Test-Path $python)) {
    throw "Virtual environment python not found at $python"
}

& $python -m pip install --upgrade pip setuptools wheel

foreach ($req in $requirementsMap[$Tier]) {
    if (!(Test-Path $req)) { throw "Requirements file not found: $req" }

    if ($selectedMode -eq "offline") {
        if (!(Test-Path "wheelhouse")) {
            throw "Offline mode requested but ./wheelhouse not found. Build/copy wheelhouse first (see docs/RUN_WINDOWS.md)."
        }
        Write-Host "==> Installing $req from local wheelhouse"
        & $python -m pip install --no-index --find-links .\wheelhouse -r $req
    } else {
        Write-Host "==> Installing $req from online index"
        & $python -m pip install -r $req
    }
}

Write-Host "==> Installing project in editable mode"
& $python -m pip install --no-build-isolation --no-deps -e .

if (!(Test-Path ".env")) {
    Write-Host "==> Creating .env from .env.example"
    Copy-Item .env.example .env
}

if ($Mock) {
    Write-Host "==> Enforcing MOCK mode defaults in .env"
    Set-EnvValueInFile -Path ".env" -Key "MOCK_API_CALLS" -Value "true"
    Set-EnvValueInFile -Path ".env" -Key "OPENAI_API_KEY" -Value "sk-test"
    Set-EnvValueInFile -Path ".env" -Key "PAPER_TRADING_ONLY" -Value "true"
}

Write-Host "==> Running self-check"
$checkMode = $checkModeMap[$Tier]
if ($checkMode) { & $python scripts/selfcheck.py $checkMode } else { & $python scripts/selfcheck.py }

Write-Host ""
Write-Host "Bootstrap successful." -ForegroundColor Green
Write-Host "UI URL:  http://127.0.0.1:$PortUI"
if ($StartAPI) { Write-Host "API URL: http://127.0.0.1:$PortAPI" }
Write-Host ""
Write-Host "Next commands:" -ForegroundColor Cyan
Write-Host "  .\scripts\run_ui.ps1 -Port $PortUI"
Write-Host "  .\scripts\run_api.ps1 -Port $PortAPI"
Write-Host "  .\scripts\run_scheduler.ps1 -Mode once"
Write-Host "  alphaprime doctor"
Write-Host ""

if ($StartAPI) {
    Write-Host "==> Starting API in background on port $PortAPI"
    Start-Process -FilePath $python -ArgumentList "-m uvicorn dashboard.app_v2:create_app --factory --reload --port $PortAPI" -NoNewWindow
}

if ($StartUI) {
    Write-Host "==> Starting Streamlit UI on port $PortUI"
    & $python -m streamlit run app.py --server.port $PortUI --server.address 127.0.0.1
}
}
finally {
    Pop-Location
}
