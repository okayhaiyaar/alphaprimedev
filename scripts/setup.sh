#!/bin/bash
# ALPHA-PRIME v2.0 - Setup Script
# Description: Automated development environment setup
# Usage: bash scripts/setup.sh [--production|--dev|--ci|--skip-db|--skip-redis]

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# === CONFIGURATION ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
NODE_VERSION="${NODE_VERSION:-18}"
ENVIRONMENT="development"
SKIP_DB=""
SKIP_REDIS=""

# === COLOR OUTPUT ===
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# === HELPER FUNCTIONS ===
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}==>${NC} $1\n"; }

abort() {
    log_error "$1"
    exit 1
}

# === 1. PRE-FLIGHT CHECKS ===
check_requirements() {
    log_step "Running pre-flight checks..."

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        PKG_MANAGER="apt-get"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    else
        abort "Unsupported OS: $OSTYPE"
    fi

    # WSL detection (treated as linux)
    if grep -qi microsoft /proc/version 2>/dev/null; then
        OS="linux"
        log_info "WSL2 environment detected"
    fi

    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        abort "Do not run as root. Use sudo for individual commands."
    fi

    # Check required tools
    for cmd in git curl; do
        if ! command -v "$cmd" &>/dev/null; then
            abort "Required command not found: $cmd"
        fi
    done

    # Check Python
    if ! command -v python3 &> /dev/null; then
        abort "Python 3 not found (install python3 first)"
    fi

    PYTHON_VER=$(python3 --version | awk '{print $2}')
    if [[ $(echo "$PYTHON_VER 3.10" | awk '{print ($1 >= $2)}') -eq 0 ]]; then
        abort "Python >= 3.10 required (found $PYTHON_VER)"
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_warn "pip3 not found, will install via system package manager if possible"
    fi

    log_info "Pre-flight checks passed ✓"
}

# === 2. SYSTEM DEPENDENCIES ===
install_system_dependencies() {
    log_step "Installing system dependencies..."

    if [[ "$OS" == "linux" ]]; then
        if ! command -v sudo &>/dev/null; then
            abort "sudo is required on Linux"
        fi
        sudo apt-get update -y

        sudo apt-get install -y \
            postgresql postgresql-contrib postgresql-client \
            redis-server \
            python3-dev python3-pip python3-venv \
            build-essential libpq-dev \
            git curl wget \
            libssl-dev libffi-dev

    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &>/dev/null; then
            abort "Homebrew not found. Install from https://brew.sh and re-run."
        fi

        brew update
        brew install postgresql@14 redis python@${PYTHON_VERSION} || true

        # Ensure services are started
        brew services start postgresql@14 || true
        brew services start redis || true
    fi

    log_info "System dependencies installed ✓"
}

# === 3. PYTHON ENVIRONMENT ===
setup_python_environment() {
    log_step "Setting up Python environment..."

    cd "$PROJECT_ROOT"

    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_info "Virtual environment created at venv/"
    else
        log_info "Virtual environment already exists"
    fi

    # shellcheck disable=SC1091
    source venv/bin/activate

    # Upgrade tooling
    pip install --upgrade pip setuptools wheel

    # Install main dependencies
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python dependencies from requirements.txt..."
        pip install -r requirements.txt
    else
        log_warn "requirements.txt not found, skipping main dependencies"
    fi

    # Install dev dependencies
    if [[ -f "requirements-dev.txt" && "$ENVIRONMENT" != "production" ]]; then
        log_info "Installing dev dependencies from requirements-dev.txt..."
        pip install -r requirements-dev.txt
    fi

    log_info "Python environment ready ✓"
}

# === 4. DATABASE SETUP ===
setup_database() {
    log_step "Setting up PostgreSQL database..."

    DB_NAME="${DB_NAME:-alpha_prime_dev}"
    DB_USER="${DB_USER:-alpha_prime_user}"
    DB_PASSWORD="${DB_PASSWORD:-alpha_prime_pass_local}"
    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"

    # Ensure pg_isready exists
    if ! command -v pg_isready &>/dev/null; then
        log_warn "pg_isready not found, attempting to install client tools..."
        if [[ "$OS" == "linux" ]]; then
            sudo apt-get install -y postgresql-client
        fi
    fi

    # Wait for PostgreSQL
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -t 5 &>/dev/null; then
        log_warn "PostgreSQL not ready on $DB_HOST:$DB_PORT, attempting to start..."

        if [[ "$OS" == "linux" ]]; then
            if command -v systemctl &>/dev/null; then
                sudo systemctl start postgresql || true
            fi
        elif [[ "$OS" == "macos" ]]; then
            brew services start postgresql@14 || true
        fi

        sleep 3

        if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -t 10 &>/dev/null; then
            abort "PostgreSQL not running on $DB_HOST:$DB_PORT"
        fi
    fi

    # Create DB user
    if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER'" | grep -q 1; then
        log_info "Creating database user $DB_USER..."
        sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
    else
        log_info "Database user $DB_USER already exists"
    fi

    # Create DB
    if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1; then
        log_info "Creating database $DB_NAME..."
        sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
    else
        log_info "Database $DB_NAME already exists"
    fi

    # Grant privileges
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" >/dev/null

    # Run migrations if alembic available
    cd "$PROJECT_ROOT"
    if [[ -d "migrations" ]]; then
        if command -v alembic &>/dev/null; then
            log_info "Running database migrations..."
            alembic upgrade head
        else
            log_warn "alembic not found; skipping migrations (install in requirements-dev.txt)"
        fi
    fi

    log_info "Database setup complete ✓"
}

# === 5. REDIS SETUP ===
setup_redis() {
    log_step "Setting up Redis..."

    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"

    if ! command -v redis-cli &>/dev/null; then
        log_warn "redis-cli not found; attempting to install redis-tools/client..."
        if [[ "$OS" == "linux" ]]; then
            sudo apt-get install -y redis-tools
        fi
    fi

    # Initial ping
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &>/dev/null; then
        log_warn "Redis not running on $REDIS_HOST:$REDIS_PORT, attempting to start..."

        if [[ "$OS" == "linux" ]]; then
            if command -v systemctl &>/dev/null; then
                sudo systemctl start redis-server || true
                sudo systemctl enable redis-server || true
            else
                sudo service redis-server start || true
            fi
        elif [[ "$OS" == "macos" ]]; then
            brew services start redis || true
        fi

        sleep 2
    fi

    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q "PONG"; then
        log_info "Redis running on $REDIS_HOST:$REDIS_PORT ✓"
    else
        abort "Failed to connect to Redis at $REDIS_HOST:$REDIS_PORT"
    fi
}

# === 6. CONFIGURATION FILES ===
setup_configuration() {
    log_step "Generating configuration files..."

    cd "$PROJECT_ROOT"

    DB_NAME="${DB_NAME:-alpha_prime_dev}"
    DB_USER="${DB_USER:-alpha_prime_user}"
    DB_PASSWORD="${DB_PASSWORD:-alpha_prime_pass_local}"
    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"
    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"

    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            log_info "Created .env from .env.example"
        else
            log_info "Creating default .env file..."
            SECRET_KEY="$(openssl rand -hex 32 2>/dev/null || echo 'changeme')"
            JWT_SECRET="$(openssl rand -hex 32 2>/dev/null || echo 'changemejwt')"

            cat > .env << EOF
# ALPHA-PRIME v2.0 Configuration
ENVIRONMENT=${ENVIRONMENT}
DEBUG=$([[ "$ENVIRONMENT" == "production" ]] && echo "False" || echo "True")

# Database
DATABASE_URL=postgresql+asyncpg://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME

# Redis
REDIS_URL=redis://$REDIS_HOST:$REDIS_PORT/0

# Broker API (Zerodha)
ZERODHA_API_KEY=your_api_key_here
ZERODHA_SECRET=your_secret_here

# Security
SECRET_KEY=$SECRET_KEY
JWT_SECRET=$JWT_SECRET

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/alpha_prime.log

# Features
ENABLE_PAPER_TRADING=True
ENABLE_BACKTESTING=True
EOF
            log_info "Generated .env file"
        fi

        log_warn "Update .env with your real API credentials and secrets"
    else
        log_info ".env already exists, skipping creation"
    fi

    log_info "Configuration ready ✓"
}

# === 7. DIRECTORY STRUCTURE ===
create_directory_structure() {
    log_step "Creating directory structure..."

    cd "$PROJECT_ROOT"

    mkdir -p logs
    mkdir -p data/{cache,backtest,strategies,models}
    mkdir -p tests/{unit,integration,e2e,fixtures}
    mkdir -p uploads
    mkdir -p exports

    chmod 755 logs data uploads exports || true

    log_info "Directory structure created ✓"
}

# === 8. DEVELOPMENT TOOLS ===
setup_dev_tools() {
    log_step "Setting up development tools..."

    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Skipping development tools in production"
        return
    fi

    cd "$PROJECT_ROOT"

    # Install pre-commit in current venv if available
    if command -v pre-commit &>/dev/null; then
        pre-commit install
        log_info "Pre-commit hooks installed"
    else
        log_warn "pre-commit not installed; install via pip to enable git hooks"
    fi

    # Create pytest.ini if missing
    if [[ ! -f "pytest.ini" ]]; then
        cat > pytest.ini << 'EOF'
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
EOF
        log_info "Created pytest.ini"
    else
        log_info "pytest.ini already exists"
    fi

    log_info "Development tools ready ✓"
}

# === 9. TESTING SETUP ===
setup_testing() {
    log_step "Setting up test environment..."

    DB_NAME="${DB_NAME:-alpha_prime_dev}"
    DB_USER="${DB_USER:-alpha_prime_user}"
    TEST_DB_NAME="${TEST_DB_NAME:-${DB_NAME}_test}"

    # Create test database
    if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname = '$TEST_DB_NAME'" | grep -q 1; then
        log_info "Creating test database $TEST_DB_NAME..."
        sudo -u postgres psql -c "CREATE DATABASE $TEST_DB_NAME OWNER $DB_USER;"
    else
        log_info "Test database $TEST_DB_NAME already exists"
    fi

    if [[ -d "$PROJECT_ROOT/tests/fixtures" ]]; then
        log_info "Test fixtures directory detected: tests/fixtures"
    fi

    log_info "Testing setup complete ✓"
}

# === 10. VERIFICATION & SMOKE TESTS ===
verify_installation() {
    log_step "Verifying installation..."

    cd "$PROJECT_ROOT"
    # shellcheck disable=SC1091
    source venv/bin/activate

    # Basic imports
    if python3 -c "import core, integrations" 2>/dev/null; then
        log_info "Core Python modules importable ✓"
    else
        log_warn "Failed to import some modules (core/integrations). Check your PYTHONPATH and installation."
    fi

    # Configuration load
    if python3 -c "from config import get_settings; get_settings()" 2>/dev/null; then
        log_info "Configuration module loadable ✓"
    else
        log_warn "Could not import config.get_settings. Ensure config module exists."
    fi

    # Optional smoke test
    if command -v pytest &>/dev/null && [[ -f "tests/unit/test_config.py" ]]; then
        log_info "Running smoke tests (tests/unit/test_config.py)..."
        if pytest tests/unit/test_config.py -v --tb=short; then
            log_info "Smoke tests passed ✓"
        else
            log_warn "Some smoke tests failed (see output above)"
        fi
    else
        log_warn "Skipping smoke tests (pytest or tests/unit/test_config.py missing)"
    fi

    log_info "========================================"
    log_info "✓ ALPHA-PRIME v2.0 setup complete!"
    log_info "========================================"

    echo ""
    echo "Next steps:"
    echo "  1. Activate venv:  source venv/bin/activate"
    echo "  2. Update .env with API credentials and secrets"
    echo "  3. Run tests:      pytest tests/"
    echo "  4. Start app:      python -m dashboard.app_v2"
    echo ""
}

# === ARGUMENT PARSING ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --production)
            ENVIRONMENT="production"
            shift
            ;;
        --dev)
            ENVIRONMENT="development"
            shift
            ;;
        --ci)
            ENVIRONMENT="ci"
            shift
            ;;
        --skip-db)
            SKIP_DB="true"
            shift
            ;;
        --skip-redis)
            SKIP_REDIS="true"
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --production    Production setup"
            echo "  --dev           Development setup (default)"
            echo "  --ci            CI/CD setup (non-interactive)"
            echo "  --skip-db       Skip PostgreSQL setup"
            echo "  --skip-redis    Skip Redis setup"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            abort "Unknown option: $1"
            ;;
    esac
done

# === MAIN EXECUTION FLOW ===
main() {
    log_info "ALPHA-PRIME v2.0 Setup"
    log_info "Environment: $ENVIRONMENT"
    log_info "Platform: $OSTYPE"

    check_requirements
    install_system_dependencies
    setup_python_environment

    if [[ -z "${SKIP_DB}" ]]; then
        setup_database
    else
        log_warn "Skipping database setup (--skip-db)"
    fi

    if [[ -z "${SKIP_REDIS}" ]]; then
        setup_redis
    else
        log_warn "Skipping Redis setup (--skip-redis)"
    fi

    setup_configuration
    create_directory_structure
    setup_dev_tools
    setup_testing
    verify_installation
}

main "$@"
