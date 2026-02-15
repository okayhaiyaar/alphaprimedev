param(
    [switch]$ClearCache,
    [switch]$ResetPortfolio,
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not $ClearCache -and -not $ResetPortfolio) {
        throw "Choose at least one action: -ClearCache and/or -ResetPortfolio"
    }

    if (-not $Force) {
        $summary = @()
        if ($ClearCache) { $summary += "clear data/cache" }
        if ($ResetPortfolio) { $summary += "reset data/portfolio.json and data/trade_history.csv" }
        $answer = Read-Host "About to $($summary -join '; '). Continue? (y/N)"
        if ($answer -notin @("y", "Y", "yes", "YES")) {
            Write-Host "Cancelled."
            exit 1
        }
    }

    if ($ClearCache -and (Test-Path "data/cache")) {
        Remove-Item -Path "data/cache/*" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Cleared data/cache"
    }

    if ($ResetPortfolio) {
        if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
        if (Test-Path "data/portfolio.json") { Remove-Item -Path "data/portfolio.json" -Force }
        if (Test-Path "data/trade_history.csv") { Remove-Item -Path "data/trade_history.csv" -Force }
        Write-Host "Removed data/portfolio.json and data/trade_history.csv (they will be recreated automatically)."
    }

    Write-Host "Local reset complete."
}
finally {
    Pop-Location
}
