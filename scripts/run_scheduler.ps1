param(
    [ValidateSet("once", "scheduled", "test")]
    [string]$Mode = "once",
    [switch]$Armed
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
$python = ".\.venv\Scripts\python.exe"
if (!(Test-Path $python)) { throw "Missing .venv. Run .\scripts\bootstrap_windows.ps1 first." }
$args = @("scheduler.py", $Mode)
if ($Armed) { $args += "--armed"; $args += "--execute" }
& $python $args
}
finally {
    Pop-Location
}
