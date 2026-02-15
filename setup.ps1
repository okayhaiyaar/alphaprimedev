<#
  ALPHA-PRIME v2.0 - Windows Setup (simplified)
  Usage:
    .\setup.ps1
    .\setup.ps1 -Environment production -SkipDb -SkipRedis
#>

param(
    [ValidateSet("development","production","ci")]
    [string]$Environment = "development",
    [switch]$SkipDb,
    [switch]$SkipRedis
)

$ErrorActionPreference = "Stop"

function Write-Info      { param($m) Write-Host "[INFO]  $m" -ForegroundColor Green }
function Write-Warn      { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-ErrorMsg  { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

# --- CONFIG (no ? operator, just simple ifs) ---
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
Set-Location $ProjectRoot

$DB_NAME     = "alpha_prime_dev"
$DB_USER     = "alpha_prime_user"
$DB_PASSWORD = "alpha_prime_pass_local"
$DB_HOST     = "localhost"
$DB_PORT     = "5432"
$REDIS_HOST  = "localhost"
$REDIS_PORT  = "6379"

if ($env:DB_NAME)     { $DB_NAME     = $env:DB_NAME }
if ($env:DB_USER)     { $DB_USER     = $env:DB_USER }
if ($env:DB_PASSWORD) { $DB_PASSWORD = $env:DB_PASSWORD }
if ($env:DB_HOST)     { $DB_HOST     = $env:DB_HOST }
if ($env:DB_PORT)     { $DB_PORT     = $env:DB_PORT }
if ($env:REDIS_HOST)  { $REDIS_HOST  = $env:REDIS_HOST }
if ($env:REDIS_PORT)  { $REDIS_PORT  = $env:REDIS_PORT }

# --- 1. PRE-FLIGHT CHECKS ---
function Check-Requirements {
    Write-Info "Running pre-flight checks..."

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-ErrorMsg "Python not found on PATH. Install Python 3.10+ and re-run."
        exit 1
    }

    $pyVerParts = (python --version 2>&1).Split()
    if ($pyVerParts.Count -lt 2) {
        Write-ErrorMsg "Could not parse python version."
        exit 1
    }
    $pyVer = $pyVerParts[1]
    if ([version]$pyVer -lt [version]"3.10") {
        Write-ErrorMsg "Python >= 3.10 required (found $pyVer)."
        exit 1
    }

    if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
        Write-Warn "pip not found; attempting python -m ensurepip --upgrade."
        python -m ensurepip --upgrade | Out-Null
    }

    Write-Info "Pre-flight checks passed ✓"
}

# --- 2. PYTHON ENVIRONMENT ---
function Setup-PythonEnvironment {
    Write-Info "Setting up Python environment..."

    $venvPath = Join-Path $ProjectRoot "venv"
    if (-not (Test-Path $venvPath)) {
        Write-Info "Creating virtual environment at venv..."
        python -m venv venv
    } else {
        Write-Info "Virtual environment already exists."
    }

    $activate = Join-Path $venvPath "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) {
        Write-ErrorMsg "Activation script not found at $activate"
        exit 1
    }

    Write-Info "Activating virtual environment..."
    . $activate

    Write-Info "Upgrading pip, setuptools, wheel..."
    python -m pip install --upgrade pip setuptools wheel

    if (Test-Path "requirements.txt") {
        Write-Info "Installing Python dependencies from requirements.txt..."
        pip install -r requirements.txt
    } else {
        Write-Warn "requirements.txt not found, skipping main dependencies."
    }

    if ($Environment -ne "production" -and (Test-Path "requirements-dev.txt")) {
        Write-Info "Installing dev dependencies from requirements-dev.txt..."
        pip install -r requirements-dev.txt
    }

    Write-Info "Python environment ready ✓"
}

# --- 3. DATABASE SETUP ---
function Setup-Database {
    if ($SkipDb) {
        Write-Warn "Skipping database setup (-SkipDb)."
        return
    }

    Write-Info "Setting up PostgreSQL database..."

    if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
        Write-Warn "psql not found on PATH. Ensure PostgreSQL is installed and psql is available."
        return
    }

    Write-Info "Ensuring role $DB_USER exists..."
    $roleCheck = & psql -U postgres -h $DB_HOST -p $DB_PORT -tAc "SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER';" 2>$null
    if (-not $roleCheck) {
        & psql -U postgres -h $DB_HOST -p $DB_PORT -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
        Write-Info "Created database user $DB_USER."
    } else {
        Write-Info "Database user $DB_USER already exists."
    }

    Write-Info "Ensuring database $DB_NAME exists..."
    $dbCheck = & psql -U postgres -h $DB_HOST -p $DB_PORT -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" 2>$null
    if (-not $dbCheck) {
        & psql -U postgres -h $DB_HOST -p $DB_PORT -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
        Write-Info "Created database $DB_NAME."
    } else {
        Write-Info "Database $DB_NAME already exists."
    }

    & psql -U postgres -h $DB_HOST -p $DB_PORT -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" | Out-Null

    if (Test-Path ".\alembic.ini") {
        if (Get-Command alembic -ErrorAction SilentlyContinue) {
            Write-Info "Running Alembic migrations..."
            alembic upgrade head
        } else {
            Write-Warn "alembic not found in PATH; skipping migrations."
        }
    }

    Write-Info "Database setup complete ✓"
}

# --- 4. REDIS CHECK ---
function Setup-Redis {
    if ($SkipRedis) {
        Write-Warn "Skipping Redis setup (-SkipRedis)."
        return
    }

    Write-Info "Checking Redis..."

    $redisCli = Get-Command redis-cli -ErrorAction SilentlyContinue
    if (-not $redisCli) {
        Write-Warn "redis-cli not found. Ensure Redis (or Memurai) is installed and running."
        return
    }

    $ping = & redis-cli -h $REDIS_HOST -p $REDIS_PORT ping 2>$null
    if ($ping -eq "PONG") {
        Write-Info "Redis running on $REDIS_HOST:$REDIS_PORT ✓"
    } else {
        Write-Warn "Could not ping Redis at $REDIS_HOST:$REDIS_PORT. Start the Redis service and re-run if needed."
    }
}

# --- 5. .env CONFIG ---
function Setup-Configuration {
    Write-Info "Generating configuration files..."

    $envPath = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $envPath)) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Info "Created .env from .env.example."
        } else {
            Write-Info "Creating default .env file..."

            $secret    = [Guid]::NewGuid().ToString("N")
            $jwtSecret = [Guid]::NewGuid().ToString("N")
            $debug = "True"
            if ($Environment -eq "production") { $debug = "False" }

@"
# ALPHA-PRIME v2.0 Configuration
ENVIRONMENT=$Environment
DEBUG=$debug

# Database
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME

# Redis
REDIS_URL=redis://$REDIS_HOST:$REDIS_PORT/0

# Broker API (Zerodha)
ZERODHA_API_KEY=your_api_key_here
ZERODHA_SECRET=your_secret_here

# Security
SECRET_KEY=$secret
JWT_SECRET=$jwtSecret

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/alpha_prime.log

# Features
ENABLE_PAPER_TRADING=True
ENABLE_BACKTESTING=True
"@ | Set-Content $envPath -Encoding UTF8

            Write-Info "Generated .env file."
        }
        Write-Warn "Update .env with your real API credentials and secrets."
    } else {
        Write-Info ".env already exists, skipping creation."
    }
}

# --- 6. DIRECTORIES ---
function Setup-Directories {
    Write-Info "Creating directory structure..."

    New-Item -ItemType Directory -Path "logs"             -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "data"             -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "data\cache"       -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "data\backtest"    -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "data\strategies"  -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "data\models"      -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "tests"            -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "tests\unit"       -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "tests\integration"-ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "tests\e2e"        -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "tests\fixtures"   -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "uploads"          -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "exports"          -ErrorAction SilentlyContinue | Out-Null

    Write-Info "Directory structure created ✓"
}

# --- 7. DEV TOOLS ---
function Setup-DevTools {
    if ($Environment -eq "production") {
        Write-Info "Skipping development tools in production."
        return
    }

    Write-Info "Setting up development tools..."

    if (Get-Command pre-commit -ErrorAction SilentlyContinue) {
        pre-commit install
        Write-Info "Pre-commit hooks installed."
    } else {
        Write-Warn "pre-commit not installed; install via 'pip install pre-commit' to enable git hooks."
    }

    if (-not (Test-Path "pytest.ini")) {
@"
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
"@ | Set-Content "pytest.ini" -Encoding UTF8

        Write-Info "Created pytest.ini."
    } else {
        Write-Info "pytest.ini already exists."
    }

    Write-Info "Development tools ready ✓"
}

# --- 8. TEST DB ---
function Setup-Testing {
    Write-Info "Setting up test database..."

    if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
        Write-Warn "psql not found; skipping test database setup."
        return
    }

    $TEST_DB_NAME = "${DB_NAME}_test"

    $dbCheck = & psql -U postgres -h $DB_HOST -p $DB_PORT -tAc "SELECT 1 FROM pg_database WHERE datname = '$TEST_DB_NAME';" 2>$null
    if (-not $dbCheck) {
        & psql -U postgres -h $DB_HOST -p $DB_PORT -c "CREATE DATABASE $TEST_DB_NAME OWNER $DB_USER;"
        Write-Info "Created test database $TEST_DB_NAME."
    } else {
        Write-Info "Test database $TEST_DB_NAME already exists."
    }

    if (Test-Path ".\tests\fixtures") {
        Write-Info "Test fixtures directory detected: tests/fixtures"
    }

    Write-Info "Testing setup complete ✓"
}

# --- 9. VERIFY ---
function Verify-Installation {
    Write-Info "Verifying installation..."

    $venvPath = Join-Path $ProjectRoot "venv"
    $activate = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activate) { . $activate }

    try {
        python -c "import core, integrations" 2>$null
        Write-Info "Core Python modules importable ✓"
    } catch {
        Write-Warn "Failed to import some modules (core/integrations). Check PYTHONPATH and installation."
    }

    try {
        python -c "from config import get_settings; get_settings()" 2>$null
        Write-Info "Configuration module loadable ✓"
    } catch {
        Write-Warn "Could not import config.get_settings. Ensure config module exists."
    }

    Write-Info "========================================"
    Write-Info "✓ ALPHA-PRIME v2.0 setup complete!"
    Write-Info "========================================"
}

# --- MAIN ---
Write-Info "ALPHA-PRIME v2.0 Setup (Windows)"
Write-Info "Environment: $Environment"

Check-Requirements
Setup-PythonEnvironment
Setup-Database
Setup-Redis
Setup-Configuration
Setup-Directories
Setup-DevTools
Setup-Testing
Verify-Installation
