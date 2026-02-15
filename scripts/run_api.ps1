param([int]$Port = 8000)
$python = ".\.venv\Scripts\python.exe"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
if (!(Test-Path $python)) { throw "Missing .venv. Run .\scripts\bootstrap_windows.ps1 first." }
& $python scripts/selfcheck.py --api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m uvicorn dashboard.app_v2:create_app --factory --reload --port $Port
}
finally {
    Pop-Location
}
