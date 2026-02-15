# ALPHA-PRIME v2.0 – Additional Notes & Technical Reference  
**Everything else you need to know**

Last updated: February 11, 2026

***

## Table of Contents

1. [System Architecture](#system-architecture)  
2. [Directory Structure](#directory-structure)  
3. [Common Errors & Solutions](#common-errors--solutions)  
4. [Configuration Files Explained](#configuration-files-explained)  
5. [Database Schema](#database-schema)  
6. [Logs & Debugging](#logs--debugging)  
7. [Pre-Live Checklist](#pre-live-checklist)  
8. [Safe Experimentation Guide](#safe-experimentation-guide)  
9. [Maintenance Tasks](#maintenance-tasks)  
10. [API Reference](#api-reference)  
11. [Performance Tuning](#performance-tuning)  
12. [Backup & Recovery](#backup--recovery)  
13. [Security & API Keys](#security--api-keys)  
14. [Broker-Specific Notes](#broker-specific-notes)  
15. [Personal Notes & Journal](#personal-notes--journal)

***

## System Architecture

### High-Level Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    ALPHA-PRIME v2.0                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                     ┌──────────────┐     │
│  │  Dashboard   │◄──── WebSocket ───► │   Trading    │     │
│  │ (Streamlit)  │       / HTTP        │   Engine     │     │
│  │  Port: 8000  │                     │   (Core)     │     │
│  └──────────────┘                     └───────┬──────┘     │
│                                              │             │
│                                   ┌──────────▼─────────┐   │
│                                   │   Strategy Manager │   │
│                                   └─────────┬──────────┘   │
│                                             │              │
│       ┌───────────────────────────────┬─────┼───────────┐ │
│       │                               │     │           │ │
│  ┌────▼─────┐                   ┌────▼──────▼───┐ ┌────▼─────┐
│  │  Risk    │                   │  Execution    │ │ Portfolio │
│  │ Manager  │                   │   Engine      │ │ Manager   │
│  └────┬─────┘                   └──────┬────────┘ └────┬─────┘
│       │                                │               │       │
│       └──────────────────────┬─────────┴───────────────┘       │
│                              │                                 │
│                        ┌─────▼─────────┐                       │
│                        │ Order Manager │                       │
│                        └─────┬─────────┘                       │
│                              │                                 │
│            ┌─────────────────┼────────────────────┐            │
│            │                 │                    │            │
│      ┌─────▼─────┐    ┌─────▼─────┐        ┌─────▼─────┐      │
│      │PostgreSQL │    │  Redis    │        │  Broker   │      │
│      │ Database  │    │  Cache    │        │   API     │      │
│      └───────────┘    └───────────┘        └───────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**Trading Engine (`core/engine.py`)**  
- Main orchestrator  
- Runs async event loop  
- Receives market data  
- Generates signals  
- Coordinates strategies, risk, execution, portfolio  

**Strategy Manager (`core/strategy_engine.py`)**  
- Loads/enables/disables strategies  
- Runs strategy logic and generates raw signals  
- Tracks per-strategy performance and stats  

**Risk Manager (`core/risk_manager.py`)**  
- Pre-trade checks (limits, drawdown, correlation)  
- Position sizing calculations  
- Circuit breakers and kill switches  
- Enforces time and volatility filters  

**Execution Engine (`core/execution_engine.py`)**  
- Converts approved signals to orders  
- Routes orders to broker API  
- Tracks order life-cycle and fills  
- Models slippage and partial fills  

**Portfolio Manager (`core/portfolio.py`)**  
- Tracks positions and P&L  
- Monitors margin usage  
- Calculates equity, drawdown, exposure  

**Order Manager (`core/order_manager.py`)**  
- Creates and validates orders  
- Handles pending orders, modifications, cancellations  
- Provides order status/query interface  

**Database – PostgreSQL**  
- Durable storage for trades, orders, portfolio history  
- Strategy performance and analytics  
- Configuration backup (optional)  

**Cache – Redis**  
- Fast in-memory data for market state and quotes  
- Position snapshots / sessions  
- Rate-limiting and internal queues  

**Broker API (`integrations/`)**  
- Zerodha/Upstox connectors  
- Handles authentication and tokens  
- WebSocket feeds for prices  
- REST APIs for orders and account info  

**Dashboard (`dashboard/app_v2.py`)**  
- Streamlit-based UI  
- Real-time monitoring (P&L, positions, risk)  
- Manual controls (start/stop, modes)  
- Report viewing / basic configuration views  

***

## Directory Structure

```text
alpha-prime-v2/
├── config/                     # Configuration files
│   ├── strategies/            # Strategy configs
│   │   ├── mean_reversion.yaml
│   │   ├── breakout_scalper.yaml
│   │   └── ...
│   ├── risk_config.yaml       # Risk rules (CRITICAL)
│   ├── broker_config.yaml     # Broker settings
│   ├── trading_hours.yaml     # Market hours & holidays
│   └── general_config.yaml    # System-wide flags
│
├── core/                      # Core trading logic
│   ├── engine.py              # Main engine
│   ├── strategy_engine.py     # Strategy orchestration
│   ├── risk_manager.py        # Risk management
│   ├── execution_engine.py    # Execution layer
│   ├── portfolio.py           # Portfolio tracking
│   ├── order_manager.py       # Orders
│   └── market_hours.py        # Market hours logic
│
├── strategies/                # Strategy implementations
│   ├── base_strategy.py       # Abstract base
│   ├── mean_reversion.py
│   ├── breakout_scalper.py
│   ├── momentum.py
│   └── ...
│
├── integrations/              # External integrations
│   ├── zerodha/
│   │   ├── client.py
│   │   ├── websocket.py
│   │   └── auth.py
│   ├── upstox/
│   ├── redis_client.py        # Redis wrapper
│   └── email_alerts.py        # Email notifications
│
├── database/                  # Database layer
│   ├── connection.py          # Connection pool
│   ├── models.py              # ORM models
│   ├── migrations/            # Alembic migration scripts
│   └── queries.py             # Common queries
│
├── data/                      # Local data
│   ├── market_data/           # Historical cache
│   ├── models/                # ML models (optional)
│   └── cache/                 # File-based cache
│
├── dashboard/                 # Streamlit dashboard
│   ├── app_v2.py              # Main app
│   ├── pages/
│   │   ├── overview.py
│   │   ├── strategies.py
│   │   ├── performance.py
│   │   └── settings.py
│   └── components/            # UI components
│
├── scripts/                   # Utility scripts
│   ├── setup.sh
│   ├── health_check.py
│   ├── run_engine.py
│   ├── backtest.py
│   ├── optimize_strategy.py
│   ├── generate_report.py
│   ├── backup.py
│   ├── reset_portfolio.py
│   ├── stress_test.py
│   └── ...
│
├── tests/                     # Test suite
│   ├── smoke/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── logs/                      # Logs
│   ├── alpha_prime.log
│   ├── trades.log
│   ├── risk.log
│   ├── errors.log
│   └── archived/
│
├── reports/                   # Generated reports
│   ├── daily/
│   ├── weekly/
│   ├── monthly/
│   └── backtest/
│
├── docs/                      # Documentation
│   ├── 01_quick_start_install.md
│   ├── 02_daily_startup_checklist.md
│   ├── 03_profit_maximization_guide.md
│   └── 04_additional_notes.md   # This file
│
├── .env                       # Secrets (not committed)
├── .env.example               # Example env
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml
```

**Key files:**

- `.env` – ALL secrets (DB URL, API keys, email, Telegram)  
- `config/risk_config.yaml` – your risk rules (most important)  
- `core/engine.py` – engine entry point  
- `logs/alpha_prime.log` – first place to look on errors  

***

## Common Errors & Solutions

### Category 1: Database Errors

#### Error: `psycopg2.OperationalError: could not connect to server`

Symptoms:

- FATAL: could not connect to PostgreSQL server  
- Connection refused on port 5432  

**Causes & fixes:**

1. **PostgreSQL not running**

```bash
# macOS
brew services start postgresql@14

# Linux
sudo systemctl start postgresql
sudo systemctl status postgresql

# Windows
# Open services.msc → find PostgreSQL → Start
```

2. **Wrong port in `.env`**

```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/alpha_prime
#                            ^^^^ check port matches DB
```

Verify port:

```bash
psql -U postgres -c "SHOW port;"
```

3. **Database doesn’t exist**

```bash
createdb alpha_prime

# or in psql
psql -U postgres
CREATE DATABASE alpha_prime;
```

4. **Permission denied**

```bash
psql -U postgres
ALTER USER postgres PASSWORD 'newpassword';

# update .env with new password
```

***

#### Error: `sqlalchemy.exc.ProgrammingError: relation "trades" does not exist`

Cause: migrations not run, schema missing.

**Fix:**

```bash
# DB-only setup
bash scripts/setup.sh --db-only

# or with Alembic
cd database/migrations
alembic upgrade head
```

***

#### Error: `FATAL: sorry, too many clients already`

Cause: connection pool exhausted.

**Fixes:**

- Restart engine  
- Increase pool size:

```python
# database/connection.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
)
```

- Audit code for sessions not closed.

***

### Category 2: Redis Errors

#### Error: `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379`

Cause: Redis not running.

```bash
# macOS
brew services start redis
redis-cli ping  # PONG

# Linux
sudo systemctl start redis
redis-cli ping

# Windows
cd C:\Redis
redis-server.exe redis.windows.conf
redis-cli ping
```

***

#### Error: `WRONGTYPE Operation against a key holding the wrong kind of value`

Cause: key type mismatch.

```bash
redis-cli
DEL problematic_key

# or DANGEROUS: clear all cache
FLUSHALL
```

***

#### Error: `OOM command not allowed when used memory > 'maxmemory'`

Cause: Redis out of memory.

```bash
redis-cli INFO memory

# set higher limit & eviction policy
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# restart
brew services restart redis    # macOS
sudo systemctl restart redis   # Linux
```

***

### Category 3: Broker API Errors

#### Error: `TokenException: Invalid access token`

Causes:

- Token expired (daily for Zerodha)  
- Wrong API key/secret  

**Fix:**

```bash
python integrations/zerodha/auth.py --login
# follow interactive login, token auto-saved
```

Check `.env`:

```bash
ZERODHA_API_KEY=...
ZERODHA_API_SECRET=...
```

Test:

```bash
python scripts/broker_test.py --verify
```

***

#### Error: `InputException: Insufficient funds`

Fixes:

```bash
python scripts/broker_test.py --margin  # check margin

# lower risk
# config/risk_config.yaml
per_trade_risk: 0.2   # down from 0.5
```

Close some positions to free margin.

***

#### Error: `NetworkException: Connection timeout`

Check:

```bash
# broker status (example)
curl -I https://kite.zerodha.com/

# internet connectivity
ping 8.8.8.8
```

Add throttling to client if rate limits hit.

***

#### Error: `OrderException: Order rejected by OMS`

Causes:

- Outside trading hours  
- Symbol not tradable / wrong format  
- Order type not allowed  
- Price outside circuit limits  
- Wrong lot size  

Debug:

```bash
python scripts/broker_test.py --test-order --symbol NIFTY50 --dry-run
```

***

### Category 4: System Errors

#### Error: `ModuleNotFoundError: No module named 'core'`

Cause: venv not active or wrong directory.

```bash
# activate venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

# ensure in project root
pwd
```

If still broken:

```bash
pip install -r requirements.txt
```

***

#### Error: `PermissionError: [Errno 13] Permission denied: 'logs/alpha_prime.log'`

Fix:

```bash
ls -la logs/
chmod 644 logs/*.log
rm logs/alpha_prime.log
touch logs/alpha_prime.log
```

***

#### Error: `RuntimeError: This event loop is already running`

Only relevant if embedding engine in another asyncio environment (Jupyter, etc.). For scripts, just use `asyncio.run(main())`. If you embed:

```python
import nest_asyncio, asyncio
nest_asyncio.apply()
asyncio.run(main())
```

***

#### Error: `Port 8000 already in use`

Fix:

```bash
# macOS / Linux
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

Kill offending process, or:

```bash
python -m dashboard.app_v2 --port 8001
```

***

### Category 5: Strategy Errors

#### Error: `Strategy 'MeanReversion' failed: division by zero`

Fix:

```bash
tail -n 100 logs/errors.log

python scripts/configure_strategies.py --disable MeanReversion

python scripts/backtest.py --strategy MeanReversion --debug
```

***

#### Error: `Signal rejected: Risk limit exceeded`

System working as intended.

Check:

```bash
python scripts/show_config.py --risk --current
grep "Signal rejected" logs/risk.log | tail -n 20
```

If too aggressive:

```yaml
# config/risk_config.yaml
max_open_positions: 5
```

***

## Configuration Files Explained

### `config/risk_config.yaml`

Core risk control file:

```yaml
risk_limits:
  daily_max_loss: 2.0         # % of equity
  max_drawdown: 4.0           # % from peak
  per_trade_risk: 0.25        # % per trade
  max_open_positions: 3
  max_correlated_positions: 1

  kill_switch:
    daily_loss_threshold: 1.8
    total_drawdown_threshold: 3.5
    consecutive_losses: 5

position_sizing:
  method: "fixed_fractional"
  kelly_fraction: 0.5

time_filters:
  avoid_open_minutes: 15
  avoid_close_minutes: 15
  blackout_periods:
    - start: "12:00"
      end: "13:00"
      reason: "Lunch hour low liquidity"

volatility_filters:
  enabled: true
  india_vix_threshold: 25
  india_vix_stop: 35
  reduce_multiplier: 0.5

circuit_breakers:
  consecutive_losses:
    enabled: true
    threshold: 3
    action: "reduce_size"
    reduction: 0.5
    reset_after_wins: 2

  daily_half_limit:
    enabled: true
    action: "pause_new"
```

**Safe editing:**

```bash
cp config/risk_config.yaml config/risk_config.yaml.backup
nano config/risk_config.yaml
python scripts/validate_config.py --risk
```

***

### `config/strategies/*.yaml`

Example `config/strategies/mean_reversion.yaml`:

```yaml
strategy:
  name: "MeanReversion_NIFTY50"
  type: "mean_reversion"
  enabled: true

  symbols:
    - "NIFTY 50"
    - "NIFTY BANK"

  parameters:
    lookback_period: 20
    entry_threshold: 2.0
    exit_threshold: 0.5
    stop_loss_pct: 0.02
    take_profit_pct: 0.03

  risk_per_trade: 0.25
  max_positions: 2

  trading_hours:
    start: "09:30"
    end: "15:15"

  entry_conditions:
    min_volume_ratio: 1.2
    max_spread_bps: 10

  backtest_metrics:
    sharpe_ratio: 1.8
    max_drawdown: 3.2
    win_rate: 0.63
    last_tested: "2026-02-01"
```

***

### `.env`

Secrets & environment:

```bash
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/alpha_prime

# Redis
REDIS_URL=redis://localhost:6379/0

# Broker – Zerodha
ZERODHA_API_KEY=...
ZERODHA_API_SECRET=...
ZERODHA_USER_ID=AB1234
ZERODHA_PASSWORD=...
ZERODHA_TOTP_SECRET=...

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Environment
ENVIRONMENT=production  # development/staging/production
LOG_LEVEL=INFO

ENABLE_PAPER_TRADING=false
ENABLE_REAL_TRADING=true
ENABLE_TELEGRAM_ALERTS=true
ENABLE_EMAIL_ALERTS=true
```

Security:

```bash
chmod 600 .env
```

***

## Database Schema

### `trades` table (core trade log)

```sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,            -- BUY/SELL
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(15, 2) NOT NULL,
    exit_price DECIMAL(15, 2),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    pnl DECIMAL(15, 2),
    pnl_percentage DECIMAL(10, 4),
    status VARCHAR(20) NOT NULL,          -- OPEN/CLOSED/CANCELLED
    stop_loss DECIMAL(15, 2),
    take_profit DECIMAL(15, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trades_strategy     ON trades(strategy_name);
CREATE INDEX idx_trades_symbol       ON trades(symbol);
CREATE INDEX idx_trades_entry_time   ON trades(entry_time);
CREATE INDEX idx_trades_status       ON trades(status);
```

### `orders` table

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    broker_order_id VARCHAR(100),
    trade_id VARCHAR(50) REFERENCES trades(trade_id),
    symbol VARCHAR(50) NOT NULL,
    order_type VARCHAR(20) NOT NULL,      -- MARKET/LIMIT/SL
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(15, 2),
    trigger_price DECIMAL(15, 2),
    status VARCHAR(20) NOT NULL,          -- PENDING/FILLED/CANCELLED/REJECTED
    filled_quantity INTEGER DEFAULT 0,
    average_price DECIMAL(15, 2),
    placed_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    rejection_reason TEXT
);
```

### `portfolio_history` table

```sql
CREATE TABLE portfolio_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    cash DECIMAL(15, 2) NOT NULL,
    positions_value DECIMAL(15, 2) NOT NULL,
    daily_pnl DECIMAL(15, 2),
    total_pnl DECIMAL(15, 2),
    open_positions INTEGER,
    margin_used DECIMAL(15, 2),
    margin_available DECIMAL(15, 2)
);

CREATE INDEX idx_portfolio_timestamp ON portfolio_history(timestamp);
```

### `strategy_performance` table

```sql
CREATE TABLE strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    total_pnl DECIMAL(15, 2),
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(strategy_name, date)
);
```

***

### Useful Queries

Today’s trades:

```sql
SELECT * FROM trades
WHERE DATE(entry_time) = CURRENT_DATE
ORDER BY entry_time DESC;
```

Open positions:

```sql
SELECT * FROM trades
WHERE status = 'OPEN'
ORDER BY entry_time;
```

Strategy performance last 30 days:

```sql
SELECT
    strategy_name,
    COUNT(*) AS total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
    SUM(pnl) AS total_pnl,
    AVG(pnl) AS avg_pnl,
    MAX(pnl) AS best_trade,
    MIN(pnl) AS worst_trade
FROM trades
WHERE entry_time >= CURRENT_DATE - INTERVAL '30 days'
  AND status = 'CLOSED'
GROUP BY strategy_name;
```

90-day equity curve:

```sql
SELECT
    DATE(timestamp) AS date,
    MAX(total_value) AS peak_value,
    MIN(total_value) AS trough_value,
    LAST_VALUE(total_value) OVER (PARTITION BY DATE(timestamp) ORDER BY timestamp
                                  RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS end_value
FROM portfolio_history
WHERE timestamp >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(timestamp)
ORDER BY date;
```

***

## Logs & Debugging

### Log Files

| File                  | Purpose                    | Retention | Level   |
|-----------------------|----------------------------|-----------|---------|
| `logs/alpha_prime.log` | Main system log           | 30 days   | INFO    |
| `logs/trades.log`      | Trades and order exec     | 90 days   | DEBUG   |
| `logs/risk.log`        | Risk events & rejections  | 30 days   | WARNING |
| `logs/errors.log`      | Errors only               | 90 days   | ERROR   |
| `logs/performance.log` | Performance metrics       | 30 days   | INFO    |

Log levels:

```text
DEBUG    – very detailed (debugging only)
INFO     – normal events
WARNING  – potential issues / risk rejections
ERROR    – errors, system continues
CRITICAL – severe, system stops
```

### Patterns

Successful trade:

```text
[2026-02-11 10:15:23] INFO: Signal received: MeanReversion - BUY NIFTY50
[2026-02-11 10:15:23] INFO: Risk check passed: position size 100
[2026-02-11 10:15:24] INFO: Order placed: ORDER123 ...
[2026-02-11 10:15:25] INFO: Order filled: ORDER123, avg price: 21450.50
[2026-02-11 10:15:25] INFO: Trade opened: TRADE456 ...
```

Risk rejection:

```text
[2026-02-11 11:30:15] WARNING: Signal rejected by risk manager
[2026-02-11 11:30:15] WARNING: Reason: Daily loss limit at 1.8% (threshold: 2.0%)
[2026-02-11 11:30:15] WARNING: No new positions allowed until next day
```

Broker error:

```text
[2026-02-11 12:45:30] ERROR: Order placement failed: ORDER789
[2026-02-11 12:45:30] ERROR: Broker response: Insufficient funds
[2026-02-11 12:45:30] INFO: Retrying with reduced position size...
```

### Commands

Tail logs:

```bash
tail -f logs/alpha_prime.log
tail -f logs/errors.log
tail -n 100 logs/alpha_prime.log
```

Filter:

```bash
tail -f logs/alpha_prime.log | grep "MeanReversion"

grep "ERROR" logs/alpha_prime.log | grep "2026-02-11"
grep "rejected" logs/risk.log
```

Log analysis:

```bash
python scripts/analyze_logs.py --today --errors-only
python scripts/analyze_logs.py --date 2026-02-11 --summary
python scripts/analyze_logs.py --trade TRADE456 --detailed
```

Debug mode:

```bash
export LOG_LEVEL=DEBUG
python scripts/run_engine.py --mode live
```

Or via `.env`:

```bash
LOG_LEVEL=DEBUG
```

***

## Pre-Live Checklist

### Before First Real Live Trade

**System validation:**

- [ ] `pytest tests/smoke/ -v` passes  
- [ ] All live strategies fully backtested  
- [ ] Walk-forward and Monte Carlo OK  
- [ ] Risk limits set and tested  
- [ ] DB migrations applied  
- [ ] Redis running and tested with `redis-cli ping`  
- [ ] Broker API authenticated (`broker_test.py --verify`)  

**Strategy validation:**

- [ ] 3+ years of backtests  
- [ ] Backtest max DD < 50% of firm limit  
- [ ] Worst backtest day < 50% daily loss limit  
- [ ] Profitable or flat in bull, bear, sideways  
- [ ] Monte Carlo violation probability < 5%  
- [ ] Paper traded ≥30 days with similar metrics  

**Risk management:**

- [ ] Daily max loss = about 40% of firm’s limit  
- [ ] Per-trade risk ≤ 0.25%  
- [ ] Max positions ≤ 3–5  
- [ ] Kill switch and circuit breakers enabled  
- [ ] Time and volatility filters active  

**Infra:**

- [ ] Stable power / UPS  
- [ ] Backup internet path  
- [ ] Broker manual platform ready  
- [ ] Emergency contacts accessible  

**Knowledge:**

- [ ] Know emergency flatten script  
- [ ] Know how to stop engine (CTRL+C)  
- [ ] Have read docs 01–03  

***

## Safe Experimentation Guide

### Golden Rule

**Never experiment with real money.**

Pipeline:

```text
Idea
 ↓
Backtest (3+ years)
 ↓
Walk-Forward Test
 ↓
Monte Carlo Simulation
 ↓
Paper Trade (30+ days)
 ↓
Live @ 10% size (2 weeks)
 ↓
Live @ 25% size (2 weeks)
 ↓
Live @ 50% size (1 month)
 ↓
Live @ 100% size
```

### Steps

**1. Backtest**

```bash
python scripts/backtest.py --strategy NewStrategy \
  --start 2023-01-01 \
  --end 2026-02-11 \
  --metrics all \
  --report detailed
```

**2. Walk-Forward**

```bash
python scripts/backtest.py --strategy NewStrategy \
  --walk-forward \
  --train-period 365 \
  --test-period 90 \
  --step 30
```

**3. Monte Carlo**

```bash
python scripts/backtest.py --strategy NewStrategy \
  --monte-carlo \
  --simulations 10000
```

**4. Paper Trade (30 days)**

```bash
python scripts/configure_strategy.py --strategy NewStrategy \
  --enable \
  --mode paper

python scripts/run_engine.py --mode paper --duration 30
```

Compare backtest vs paper:

```bash
python scripts/compare_results.py --backtest vs paper --strategy NewStrategy
```

**5. Gradual live introduction**

```bash
# Week 1–2: 10%
python scripts/configure_strategy.py --strategy NewStrategy \
  --enable --mode live --size-multiplier 0.1

# Week 3–4: 25%
... --size-multiplier 0.25

# Week 5–8: 50%
... --size-multiplier 0.5

# Week 9+: 100%
... --size-multiplier 1.0
```

***

## Maintenance Tasks

### Daily (≈5 minutes)

- [ ] Check daily report (`generate_report.py --today`)  
- [ ] Scan `logs/errors.log` for new errors  
- [ ] Confirm P&L ≈ broker P&L  
- [ ] Update journal if anything notable  

### Weekly (≈15 minutes)

```bash
python scripts/generate_report.py --week --detailed
python scripts/cleanup_logs.py --older-than 30
python scripts/backup.py --weekly
```

- [ ] Review strategy performance vs expectations  
- [ ] Check disk space, memory, Redis usage  
- [ ] Note any recurring issues  

### Monthly (≈1 hour)

```bash
python scripts/health_check.py --full
python scripts/db_maintenance.py --vacuum --analyze
pip list --outdated
python scripts/backup.py --monthly --verify
python scripts/generate_report.py --month --pdf --detailed
python scripts/archive_old_data.py --older-than 90
```

- [ ] Review drawdown vs targets  
- [ ] Consider risk tweaks if needed  

### Quarterly

- [ ] Full stress test:

```bash
python scripts/stress_test.py --all --report
```

- [ ] Re-backtest all live strategies  
- [ ] Test restoring from backup  
- [ ] Security/key audit  

***

## API Reference (High-Level)

### Engine (scripts)

```bash
# Monitor mode (no trades)
python scripts/run_engine.py --mode monitor

# Live trading
python scripts/run_engine.py --mode live --confirm

# Paper trading
python scripts/run_engine.py --mode paper

# Stop (graceful)
python scripts/run_engine.py --mode stop

# Emergency stop
python scripts/run_engine.py --mode emergency-stop
```

### Programmatic Engine Usage (internal)

```python
from core.engine import TradingEngine

engine = TradingEngine()
await engine.initialize()
await engine.start(mode="live")

status = await engine.get_status()
print(status)

await engine.stop()
```

*(Similar patterns apply for `StrategyEngine`, `RiskManager`, `OrderManager` as described in the paste.)*

***

## Performance Tuning (Quick Pointers)

### Database

Indexes for frequent queries:

```sql
CREATE INDEX idx_trades_date ON trades(DATE(entry_time));
CREATE INDEX idx_trades_pnl  ON trades(pnl) WHERE pnl IS NOT NULL;
CREATE INDEX idx_trades_strategy_date
  ON trades(strategy_name, entry_time);
```

Vacuum:

```bash
python scripts/db_maintenance.py --vacuum --analyze
# or
psql -U postgres -d alpha_prime -c "VACUUM ANALYZE;"
```

### Redis

```bash
redis-cli INFO memory
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET maxmemory 4gb
```

Configure persistence only if needed.

### Python

Use pooling:

```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

Profile slow code with `cProfile` or similar.

***

## Backup & Recovery

### Automated

Example cron (Linux):

```text
0 4 * * * /path/to/alpha-prime-v2/scripts/backup.py --daily --compress
```

### Manual

```bash
# Full backup
python scripts/backup.py --full --output backups/full_$(date +%Y%m%d).tar.gz

# DB only
python scripts/backup.py --db-only --output backups/db_$(date +%Y%m%d).sql
```

### Restore (DB)

```bash
dropdb alpha_prime
createdb alpha_prime
psql -U postgres -d alpha_prime -f backups/db_20260211.sql

cd database/migrations
alembic upgrade head
```

### Full Disaster Recovery

1. Fresh install (follow `01_quick_start_install.md`)  
2. Restore:

```bash
python scripts/restore.py --backup backups/latest.tar.gz --confirm
python scripts/health_check.py --full
python scripts/run_engine.py --mode paper --duration 60
python scripts/run_engine.py --mode live --confirm
```

***

## Security & API Keys

- Rotate **broker API keys** every ~90 days  
- Limit `.env` permissions:

```bash
chmod 600 .env
```

- Never log secrets or print them  
- Restrict DB user permissions to required tables only  

***

## Broker-Specific Notes (Skeleton)

*(Fill with your broker quirks: Zerodha margin rules, product types, rate limits, etc.)*

- Zerodha:
  - Intraday vs CNC vs NRML  
  - Leverage schedules  
  - Order freeze quantities / circuit limit behavior  

***

## Personal Notes & Journal

**Use this section as your scratchpad. Example:**

```text
2026-02-11
- Observed slightly higher slippage on BANKNIFTY during RBI speech.
- Might need a tighter volatility filter or time blackout.

2026-02-13
- MeanReversion_NIFTY50 underperformed 3rd week in a row.
- Plan: run deeper regime test before deciding to reduce allocation.
```

This file is meant to be **living documentation**. When you discover a new failure mode or fix, add it here so future you doesn’t have to relearn the same lesson.