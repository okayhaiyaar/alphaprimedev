# LOCAL OPS (Windows 11 + PowerShell)

Local-only cheat sheet for running ALPHA-PRIME on a single Windows machine.

## Daily commands (copy/paste)

### 1) First-time bootstrap (online)
```powershell
cd <path-to-repo>\alphaprimedev
.\scripts\bootstrap_windows.ps1 -Tier core -Mode online -Mock $true
```

### 2) First-time bootstrap (offline, wheelhouse already present)
```powershell
cd <path-to-repo>\alphaprimedev
.\scripts\bootstrap_windows.ps1 -Tier core -Mode offline -Mock $true
```

### 3) Start UI in MOCK mode (daily)
```powershell
cd <path-to-repo>\alphaprimedev
.\scripts\run_ui.ps1 -Port 8501
```
Open: http://127.0.0.1:8501

### 4) Optional API tier install + run
```powershell
cd <path-to-repo>\alphaprimedev
.\scripts\bootstrap_windows.ps1 -Tier api -StartUI $false -StartAPI $true -PortAPI 8000
# OR, if already bootstrapped:
.\scripts\run_api.ps1 -Port 8000
```
Open: http://127.0.0.1:8000/docs

### 5) Scheduler once (analysis-only, safe default)
```powershell
cd <path-to-repo>\alphaprimedev
.\scripts\run_scheduler.ps1 -Mode once
```

### 6) Scheduler once (armed execute mode, paper-only guard still applies)
```powershell
cd <path-to-repo>\alphaprimedev
.\scripts\run_scheduler.ps1 -Mode once -Armed
```

### 7) Diagnostics
```powershell
cd <path-to-repo>\alphaprimedev
.\.venv\Scripts\alphaprime doctor
python .\scripts\selfcheck.py
python .\scripts\selfcheck.py --api
python .\scripts\selfcheck.py --full
```

---

## Minimal `.env` for local UI MOCK mode

Use this minimum set:
```env
OPENAI_API_KEY=sk-test
MOCK_API_CALLS=true
PAPER_TRADING_ONLY=true
LOG_LEVEL=INFO
```

Notes:
- `bootstrap_windows.ps1 -Mock $true` now enforces these safe values in `.env`.
- API mode additionally needs API dependencies installed (`requirements-api.txt`).

---

## Where things are

- Logs: `logs\alpha_prime.log`
- Cache: `data\cache\`
- Portfolio state: `data\portfolio.json`
- Trade history: `data\trade_history.csv`
- Backups: `backups\`

---

## Common problems & fixes (Windows)

### 1) `pip` fails with 403 / restricted network
Use wheelhouse install:
```powershell
.\scripts\install_offline.ps1 -Tier core
# or api/full
.\scripts\install_offline.ps1 -Tier api
```
If needed, build wheelhouse on an internet-enabled machine first:
```powershell
.\scripts\build_wheelhouse.ps1 -Tier core
```

### 2) Port already in use (8501 or 8000)
Run on a different port:
```powershell
.\scripts\run_ui.ps1 -Port 8502
.\scripts\run_api.ps1 -Port 8001
```
Or check process using the port:
```powershell
Get-NetTCPConnection -LocalPort 8501 | Select-Object LocalAddress,LocalPort,OwningProcess,State
Get-Process -Id <PID>
```

### 3) Streamlit blank page / stale widgets
- Hard refresh browser (`Ctrl+F5`).
- Stop Streamlit and run again.
- Clear cache with `scripts\reset_local.ps1 -ClearCache -Force`.

### 4) yfinance/network failures
Expected behavior in local mode: analysis can degrade or skip symbols when market data fetch fails; logs should include fetch errors. Retry later or test network/DNS.

### 5) OpenAI key confusion in local mock testing
- For safe local testing: keep `OPENAI_API_KEY=sk-test` and `MOCK_API_CALLS=true`.
- If `MOCK_API_CALLS=false`, you must provide a real `OPENAI_API_KEY`.

---

## Reset (safe local cleanup)

Clear cache only:
```powershell
.\scripts\reset_local.ps1 -ClearCache -Force
```

Clear cache + reset portfolio/trade files:
```powershell
.\scripts\reset_local.ps1 -ClearCache -ResetPortfolio -Force
```

This does **not** delete the full repo.
