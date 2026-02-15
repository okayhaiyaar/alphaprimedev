param(
    [ValidateSet("core", "api", "full")]
    [string]$Tier = "core",
    [string]$WheelhouseDir = "wheelhouse",
    [string]$PythonVersion = "3.12",
    [string]$Platform = "win_amd64"
)

$ErrorActionPreference = "Stop"

$requirementsMap = @{
    core = @("requirements.txt")
    api  = @("requirements.txt", "requirements-api.txt")
    full = @("requirements-full.txt")
}
$requirementsFiles = $requirementsMap[$Tier]

Write-Host "==> Building wheelhouse for tier: $Tier"
Write-Host "==> Target platform: $Platform, Python: $PythonVersion"

foreach ($req in $requirementsFiles) {
    if (!(Test-Path $req)) {
        throw "Requirements file not found: $req"
    }
}

if (Test-Path $WheelhouseDir) {
    Write-Host "==> Removing existing $WheelhouseDir"
    Remove-Item -Recurse -Force $WheelhouseDir
}

New-Item -ItemType Directory -Path $WheelhouseDir | Out-Null

$pyTag = $PythonVersion.Replace(".", "")
$abiTag = "cp$pyTag"
$downloadLog = Join-Path $WheelhouseDir "download.log"

python -m pip install --upgrade pip

$allOutput = @()
foreach ($req in $requirementsFiles) {
    Write-Host "==> Downloading wheels from $req"
    $cmdOutput = python -m pip download -r $req -d $WheelhouseDir --only-binary=:all: --platform $Platform --python-version $pyTag --implementation cp --abi $abiTag 2>&1
    $allOutput += $cmdOutput
}
$allOutput | Tee-Object -FilePath $downloadLog

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Wheelhouse download failed." -ForegroundColor Red
    $missingLines = $allOutput | Where-Object { $_ -match "Could not find a version that satisfies the requirement" -or $_ -match "No matching distribution found for" }
    if ($missingLines) {
        Write-Host "Packages without compatible wheels:" -ForegroundColor Yellow
        $missingLines | ForEach-Object { Write-Host "  $_" }
    }
    Write-Host ""
    Write-Host "Suggestions:" -ForegroundColor Yellow
    Write-Host "  1) Keep Python and wheel target aligned (e.g., -PythonVersion 3.12 -Platform win_amd64)."
    Write-Host "  2) Pin package versions that publish cp$pyTag/$Platform wheels."
    Write-Host "  3) Build separate tier wheelhouse first: core then api/full."
    exit 1
}

$manifest = Join-Path $WheelhouseDir "manifest.txt"
Write-Host "==> Writing wheel manifest: $manifest"
Get-ChildItem -Path $WheelhouseDir -File |
    Sort-Object Name |
    ForEach-Object {
        $hash = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash
        "$($_.Name)`t$hash"
    } | Out-File -FilePath $manifest -Encoding utf8

Write-Host "==> Wheelhouse build complete"
Write-Host "   Tier: $Tier"
Write-Host "   Files: $((Get-ChildItem -Path $WheelhouseDir -File | Measure-Object).Count)"
Write-Host "   Manifest: $manifest"
Write-Host "   Download log: $downloadLog"
