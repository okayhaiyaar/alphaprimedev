# Run ALPHA-PRIME Locally on Windows 11 (PowerShell)

This guide is Windows-first and includes **normal**, **proxy-restricted**, and **offline wheelhouse** install paths.


## Recommended: one-command bootstrap

Use the bootstrap script for fastest setup:

```powershell
# Auto mode (online if no wheelhouse, offline if wheelhouse exists)
.\scripts\bootstrap_windows.ps1

# Force online
.\scripts\bootstrap_windows.ps1 -Mode online

# Force offline (requires .\wheelhouse)
.\scripts\bootstrap_windows.ps1 -Mode offline -Tier core
```

Common options:

```powershell
.\scripts\bootstrap_windows.ps1 -Tier api -StartAPI $true -PortUI 8501 -PortAPI 8000
```

The bootstrap flow also installs the project in editable mode (`pip install -e .`) so console commands are available.

Use diagnostics anytime (includes masked secrets + RUN_ID):
```powershell
alphaprime doctor
# or
python -m alphaprime.cli doctor
```

The Streamlit UI also exposes a **Diagnostics** panel for quick local troubleshooting.


## Windows git long paths (recommended)

For repositories with deep paths, enable long path support once:

```powershell
git config --global core.longpaths true
```


## Tier matrix

| Feature target | Requirements file(s) | Self-check mode | Run command |
|---|---|---|---|
| UI only | `requirements.txt` | `python scripts/selfcheck.py` | `alphaprime-ui --port 8501` |
| UI + API | `requirements.txt` + `requirements-api.txt` | `python scripts/selfcheck.py --api` | `alphaprime-api --port 8000 --reload` |
| Full/dev | `requirements-full.txt` | `python scripts/selfcheck.py --full` | `alphaprime-ui` + optional tools |

## 0) Prerequisites

- Windows 11
- Python **3.12.x** (recommended)
- Git

Verify Python:

```powershell
python --version
```

Expected: `Python 3.12.x`.

---

## 1) Project setup (all install modes)

```powershell
git clone https://github.com/okayhaiyaar/alphaprimedev.git
cd alphaprimedev
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Create runtime config:

```powershell
Copy-Item .env.example .env
```

Minimal startup values in `.env`:

```env
OPENAI_API_KEY=sk-test
OPENAI_MODEL=gpt-4o
LOG_LEVEL=INFO
PAPER_TRADING_ONLY=true
MOCK_API_CALLS=true
```

---


## MOCK mode (safe first boot)

For first boot/testing without real credentials, use:

```env
OPENAI_API_KEY=sk-test
MOCK_API_CALLS=true
PAPER_TRADING_ONLY=true
```

Behavior in MOCK mode:
- Oracle uses a deterministic local stub decision.
- No external OpenAI API calls are made.
- Streamlit UI and scheduler analysis paths can run without real API credentials.

## 2) Normal install (internet available)

```powershell
pip install -r requirements.txt
python scripts/selfcheck.py
```

Expected: UI-required imports print `[OK]`. Missing API modules are shown as `[WARN]` in default mode and do not fail startup.

Self-check modes:

```powershell
python scripts/selfcheck.py          # UI mode (default)
python scripts/selfcheck.py --api    # require API deps
python scripts/selfcheck.py --full   # require full optional analytics stack
```

---

## 3) If `pip install` fails with **403 Forbidden**

A 403 during dependency resolution usually means network policy, not package syntax:

- outbound access to PyPI is blocked,
- corporate proxy requires auth,
- SSL inspection breaks TLS trust,
- direct `pypi.org/files.pythonhosted.org` egress is denied,
- internal mirror/index is required by policy.

### Quick checklist to confirm network/proxy root cause

```powershell
python -m pip --version
python -m pip config debug
python -m pip index versions streamlit
```

If multiple packages fail with 403 (especially early), treat this as a network/index issue.

---

## 4) Proxy / internal index install methods (secondary)

> Never commit proxy credentials or internal mirror URLs with tokens.

### Option A: Set proxy in current PowerShell session

```powershell
$env:HTTPS_PROXY="http://user:pass@proxy.company.local:8080"
$env:HTTP_PROXY="http://user:pass@proxy.company.local:8080"
pip install -r requirements.txt
```

### Option B: Configure pip.ini (persistent per-user)

Path: `%APPDATA%\pip\pip.ini`

Template:

```ini
[global]
index-url = https://pypi.org/simple
extra-index-url = https://your-internal-mirror/simple
trusted-host =
    pypi.org
    files.pythonhosted.org
    your-internal-mirror
proxy = http://user:pass@proxy.company.local:8080
```

Then run:

```powershell
pip install -r requirements.txt
```

### Option C: One-shot command flags

```powershell
pip install -r requirements.txt `
  --proxy http://user:pass@proxy.company.local:8080 `
  --trusted-host pypi.org `
  --trusted-host files.pythonhosted.org
```

---

## 5) Offline / Wheelhouse install (primary robust path)

Use this when the target machine cannot access package indexes.

### 5.1 Build wheelhouse on an internet-enabled Windows machine

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
.\scripts\build_wheelhouse.ps1 -Tier core
```

Optional cross-machine build (Linux/macOS with PowerShell Core), still targeting Windows wheels:

```powershell
pwsh ./scripts/build_wheelhouse.ps1 -Tier core -PythonVersion 3.12 -Platform win_amd64

# API tier (includes core + api requirements)
.\scripts\build_wheelhouse.ps1 -Tier api

# Full tier
.\scripts\build_wheelhouse.ps1 -Tier full
```

This script will:
- recreate `./wheelhouse`,
- download wheels for `requirements.txt` (`--only-binary=:all:`),
- write `wheelhouse/manifest.txt` with file hashes.

If binary-only download fails for a package, either pin wheel-available versions or build wheels on a matching Windows builder.

### 5.2 Transfer to restricted machine

Copy these items to the restricted machine (USB/share/artifact store):
- project source tree,
- `wheelhouse/` directory,
- `.env` file (or create from `.env.example`).

Example copy command (from online box):

```powershell
Copy-Item -Recurse .\wheelhouse \\restricted-host\drop\alphaprimedev\wheelhouse
```

### 5.3 Install from wheelhouse on restricted machine

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\install_offline.ps1 -Tier core
```

Equivalent raw install command:

```powershell
pip install --no-index --find-links .\wheelhouse -r requirements.txt
```

Verification:

```powershell
python scripts/selfcheck.py
python scripts/selfcheck.py --api   # API tier verification
python scripts/selfcheck.py --full  # strict full-stack verification
```

Expected:
- `selfcheck.py` passes for Streamlit UI readiness (core tier).
- `--api` passes after installing api tier.
- `--full` passes after installing full tier.

Offline API in 10 minutes:
```powershell
.\scripts\build_wheelhouse.ps1 -Tier api
# transfer wheelhouse
.\scripts\install_offline.ps1 -Tier api
python scripts/selfcheck.py --api
alphaprime-api --port 8000 --reload
```

---



## Dependency groups and reproducibility

- `requirements.txt`: core runtime (UI + core modules).
- `requirements-api.txt`: API runtime tier (FastAPI/uvicorn and related API deps).
- `requirements-full.txt`: optional heavy analytics + dev tooling (includes core + api).

Install full optional set when needed:

```powershell
pip install -r requirements-full.txt
```

Reproducible lock snapshot (on your target Python, e.g. 3.12):

```powershell
pip freeze > requirements.lock.txt
```

---

## Scheduler safety (armed mode)

Scheduler runs are analysis-only by default. Trade execution is blocked unless explicitly armed.

```powershell
.\scripts\run_scheduler.ps1 -Mode once                  # analysis-only
.\scripts\run_scheduler.ps1 -Mode once -Armed           # execute paper trades (still paper-only)
$env:SCHEDULER_ARMED="true"; python scheduler.py once --execute
```

## 6) Start services

### Streamlit UI (required)

```powershell
.\scripts\run_ui.ps1
# or: streamlit run app.py
```

Expected: local URL (typically `http://localhost:8501`) and dashboard loads.

### FastAPI (optional)

```powershell
.\scripts\run_api.ps1
# or: uvicorn dashboard.app_v2:create_app --factory --reload --port 8000
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content
```

---

## 7) Common Windows fixes

- **PowerShell execution policy blocks activate script**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```
- **Wrong Python selected**
  ```powershell
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python --version
  ```
- **Missing `.env`**: copy `.env.example` to `.env` and retry.
- **Playwright browser error**: not required for first Streamlit boot; only run `playwright install chromium` if/when that feature is used.
