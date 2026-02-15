param([int]$Port = 8501)
$python = ".\.venv\Scripts\python.exe"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
if (!(Test-Path $python)) { throw "Missing .venv. Run .\scripts\bootstrap_windows.ps1 first." }
& $python -m streamlit run app.py --server.port $Port --server.address 127.0.0.1
}
finally {
    Pop-Location
}
