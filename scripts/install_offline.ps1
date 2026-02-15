param(
    [ValidateSet("core", "api", "full")]
    [string]$Tier = "core",
    [string]$WheelhouseDir = "wheelhouse"
)

$ErrorActionPreference = "Stop"

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
$requirementsFiles = $requirementsMap[$Tier]
$checkMode = $checkModeMap[$Tier]

if (!(Test-Path $WheelhouseDir)) {
    throw "Wheelhouse directory not found: $WheelhouseDir"
}

foreach ($req in $requirementsFiles) {
    if (!(Test-Path $req)) {
        throw "Requirements file not found: $req"
    }
    Write-Host "==> Installing $req from local wheelhouse"
    python -m pip install --no-index --find-links $WheelhouseDir -r $req
}

Write-Host "==> Running self-check for tier '$Tier'"
if ($checkMode) {
    python scripts/selfcheck.py $checkMode
} else {
    python scripts/selfcheck.py
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Offline install verification failed for tier '$Tier'." -ForegroundColor Red
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1) Rebuild wheelhouse for this tier: .\scripts\build_wheelhouse.ps1 -Tier $Tier"
    Write-Host "  2) Verify wheelhouse path: .\$WheelhouseDir"
    Write-Host "  3) See docs/RUN_WINDOWS.md tier matrix and troubleshooting"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==> Offline install completed successfully (tier=$Tier)" -ForegroundColor Green
Write-Host "Run commands:" -ForegroundColor Cyan
Write-Host "  alphaprime-ui --port 8501"
Write-Host "  alphaprime-api --port 8000 --reload"
