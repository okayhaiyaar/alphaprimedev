```markdown
# ALPHA-PRIME v2.0

**AI-Powered Autonomous Trading System**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Status](https://img.shields.io/badge/status-Beta-yellow.svg)](https://github.com/yourusername/alpha-prime)  
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://pytest.org/)  
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **⚠️ IMPORTANT:** This is a **paper trading** system. No real money is involved.  
> It is designed for research and simulation only. Always test extensively before any live trading consideration.

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Architecture](#architecture)  
4. [Quick Start](#-quick-start-5-minutes)  
   - [Prerequisites](#prerequisites)  
   - [Installation (Docker)](#option-1-docker-recommended)  
   - [Installation (Local Python)](#option-2-local-python)  
5. [Configuration](#configuration)  
6. [Usage Guide](#usage-guide)  
   - [Daily Workflow](#daily-workflow)  
   - [Dashboard Pages](#dashboard-pages)  
7. [Module Reference](#module-reference)  
   - [Core 8 Modules](#core-8-modules)  
   - [Risk Management Modules](#risk-management-modules)  
   - [Profitability Boosters](#profitability-boosters)  
   - [Intelligence Modules](#intelligence-modules)  
   - [Validation Modules](#validation-modules)  
8. [Risk Parameters](#risk-parameters)  
9. [Data & Timezone](#data--timezone)  
10. [Development](#️-development)  
11. [Testing Strategy](#testing-strategy)  
12. [Deployment](#deployment)  
13. [Troubleshooting](#troubleshooting)  
14. [Roadmap](#roadmap)  
15. [Contributing](#contributing)  
16. [Security & Compliance](#security--compliance)  
17. [License](#-license)  
18. [Acknowledgments](#acknowledgments)  
19. [Support & Community](#support--community)

---

## Overview

ALPHA-PRIME v2.0 is an institutional-style, Python-based **AI trading research system** that combines GPT‑4o reasoning with deterministic quantitative analysis and strict risk management. It is built as a full-stack workflow: **research → analysis → decision → execution → monitoring**, with an emphasis on robustness, observability, and safety.

The system is designed around the Indian market context (IST timezone, NSE/BSE hours) and is **SEBI-conscious**: the current version is **paper trading only**, producing simulated orders and detailed logs, not live broker executions. The architecture is broker-agnostic but can be extended in future to integrate with Zerodha/Groww under compliant constraints.

At a high level, ALPHA-PRIME:

- Scans market movers each day at market open (09:30 IST).  
- Aggregates intelligence from SEC filings, news APIs, and social sentiment sources.  
- Computes a rich set of technical indicators (RSI, MACD, Bollinger Bands, EMAs, ATR, etc.).  
- Uses a GPT‑4o “Oracle” with a constrained JSON schema and Python-enforced risk rules.  
- Simulates trades in a portfolio ledger with **atomic JSON persistence** and full audit trail.  
- Monitors strategy performance, model drift, and data quality over time.

What makes it different:

- **Evidence-bound LLM reasoning** – the model is restricted to referenced sources and deterministic numeric inputs.  
- **Capital preservation first** – strict caps on per-trade risk, portfolio risk, daily loss, and correlation.  
- **Validation-first design** – walk-forward, out-of-sample checks, drift detection, and TCA hooks.  
- **Full observability** – structured logging, metrics, equity curves, and dashboard visualizations.

---

## Key Features

### Core Capabilities

- 🔍 Multi-source intelligence (SEC EDGAR via APIs, news providers, Reddit, Twitter/X when configured)  
- 📊 20+ technical indicators powered by `pandas-ta`  
- 🧠 GPT‑4o Oracle with strict JSON schema and conflict resolution logic  
- 💼 Paper trading engine with atomic `portfolio.json` persistence and CSV trade history  
- 📈 Multi-page Streamlit dashboard with a dark, Bloomberg-style aesthetic  
- 🚨 Discord webhook alerts for high-confidence signals (optional)  
- ⏰ Automated daily market cycle via Python scheduler and/or Dockerized services  

### Risk Management

- 🛡️ Dynamic **ATR-based position sizing** with hard caps on risk per trade  
- 🔴 Circuit breakers: daily loss limit, consecutive loss halt, VIX-based shutdown  
- 📉 Correlation monitoring to avoid over-concentrated, highly correlated positions  
- 📊 Real-time portfolio heat and exposure visualization in the dashboard  

### Validation & QA

- ✅ Walk-forward optimization hooks and out-of-sample backtest runner  
- 📉 Model and data drift monitoring for the Oracle and strategies  
- 🧪 Backtesting framework with separate in-sample / out-of-sample evaluation  
- 🔍 Data quality checks for missing data, outliers, and corporate actions  

### Intelligence Enhancements

- 📄 SEC Form 4 insider transaction tracking  
- 📊 Unusual options activity detection (if data sources are configured)  
- 🏢 Sector/industry rotation and peer performance analysis  
- 🗓️ Earnings and macro event calendar integration to avoid event risk  

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                      ALPHA-PRIME v2.0                        │
│                   (Python 3.10+ System)                      │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼──────┐       ┌─────▼─────┐
   │ HUNTER  │          │MATHEMATICIAN│       │  ORACLE   │
   │research │          │ data_engine │       │  brain    │
   └────┬────┘          └─────┬──────┘       └─────┬─────┘
        │                     │                     │
        │  SEC filings       │  yfinance           │  GPT-4o
        │  News/Social       │  pandas-ta          │  Risk logic
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   PORTFOLIO       │
                    │   portfolio.py    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼──────┐       ┌─────▼─────┐
   │DASHBOARD│          │ SCHEDULER  │       │  ALERTS   │
   │app_v2   │          │ scheduler  │       │ alerts    │
   └─────────┘          └────────────┘       └───────────┘
```

Core modules:

- **research_engine.py** – “Hunter” that collects intel from SEC, news, social APIs.  
- **data_engine.py** – “Mathematician” that pulls market data and computes indicators.  
- **brain.py** – “Oracle” that calls GPT‑4o under strict constraints and risk rules.  
- **app.py / dashboard/app_v2.py** – Streamlit “War Room” dashboard.  
- **alerts.py** – Discord and other notifications.  
- **portfolio.py** – Paper trading ledger with atomic JSON state and CSV trade history.  
- **scheduler.py** – Cron-like orchestrator for daily scan and signal generation.  
- **config.py** – Environment loading, logging setup, and validation.

Extended modules (Tiered):

- **Risk**: `position_sizer.py`, `circuit_breakers.py`, `correlation_monitor.py`, `risk_metrics.py`  
- **Profitability**: `regime_detector.py`, `multi_timeframe.py`, `event_calendar.py`, `strategy_registry.py`, `execution_optimizer.py`  
- **Intelligence**: `insider_tracker.py`, `options_flow.py`, `sector_analysis.py`, `sentiment_aggregator.py`  
- **Validation**: `walk_forward.py`, `drift_monitor.py`, `backtest_validator.py`, `data_quality.py`

---

## 🚀 Quick Start (5 minutes)

### Prerequisites

- Python **3.10+** (3.11 recommended)  
- Git  
- OpenAI API key (create one at <https://platform.openai.com/api-keys>)  
- Docker & Docker Compose (optional but recommended for reproducible runtime)

---

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/alpha-prime.git
cd alpha-prime

# Create environment file from template
cp .env.example .env

# Edit .env and set at least:
#   OPENAI_API_KEY=sk-...
#   LOG_LEVEL=INFO
#   PAPER_TRADING_ONLY=true
```

Build and start services:

```bash
# Build image and start dashboard + scheduler + watchdog
docker-compose up -d

# Check running containers
docker-compose ps

# View logs from dashboard
docker-compose logs -f dashboard
```

Access the dashboard:

- Open: <http://localhost:8501>

Stop services:

```bash
docker-compose down
```

---

### Option 2: Local Python

```bash
# Clone repository
git clone https://github.com/yourusername/alpha-prime.git
cd alpha-prime

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browser (Chromium)
playwright install chromium
```

Configure environment:

```bash
cp .env.example .env
# Edit .env and set at least:
#   OPENAI_API_KEY=sk-...
#   LOG_LEVEL=INFO
#   PAPER_TRADING_ONLY=true
```

Validate environment:

```bash
python config.py  # should log INFO and exit cleanly
```

Run dashboard:

```bash
streamlit run dashboard/app_v2.py \
  --server.port=8501 \
  --server.address=0.0.0.0
```

Run scheduler (separate terminal):

```bash
source .venv/bin/activate
python scheduler.py
```

---

## Configuration

All configuration is managed via `.env` (never commit your actual `.env` to version control). See `.env.example` for a complete template.

### Critical - AI & APIs

- `OPENAI_API_KEY` – **[REQUIRED]** OpenAI key for GPT‑4o.  
- `OPENAI_MODEL` – Default `gpt-4o`.  
- `OPENAI_MAX_TOKENS` – Hard cap (e.g., `4000`).  
- `OPENAI_TEMPERATURE` – Sampling temperature (default `0.7`).

### Notifications

- `DISCORD_WEBHOOK_URL` – Optional Discord webhook for alerts.  
- `DISCORD_RATE_LIMIT_RETRY` – `true|false`, auto-retry on 429.  
- `ALERT_MIN_CONFIDENCE` – Minimum Oracle confidence (0–100) for alerts.

### Social & Data APIs (Optional)

- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`  
- `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_BEARER_TOKEN`  
- `STOCKTWITS_API_KEY` – Only if licensed.  
- `SEC_API_KEY` – For sec-api.io, if you use it.  
- `NEWS_API_KEY`, `ALPHAVANTAGE_API_KEY` – Optional additional feeds.

### System Configuration

- `LOG_LEVEL` – `DEBUG|INFO|WARNING|ERROR|CRITICAL` (default `INFO`).  
- `TIMEZONE` – Default `Asia/Kolkata`.  
- `MARKET_OPEN_TIME` – `09:30`.  
- `MARKET_CLOSE_TIME` – `15:30`.

### File Paths

- `PORTFOLIO_PATH` – `data/portfolio.json`.  
- `TRADE_HISTORY_PATH` – `data/trade_history.csv`.  
- `CACHE_DIR` – `data/cache`.  
- `LOG_DIR` – `logs`.  
- `BACKUP_DIR` – `backups`.

### Trading Parameters

- `STARTING_CASH` – Initial portfolio value (e.g., `10000.0`).  
- `COMMISSION_PER_TRADE` – Commission per order (float).  
- `MAX_RISK_PER_TRADE_PCT` – Default `2.0`.  
- `MAX_PORTFOLIO_RISK_PCT` – Default `6.0`.  
- `DAILY_LOSS_LIMIT_PCT` – Default `3.0`.  
- `POSITION_SIZE_METHOD` – `ATR|FIXED|KELLY` (default `ATR`).

### Risk Management

- `CIRCUIT_BREAKER_ENABLED` – `true|false`.  
- `CONSECUTIVE_LOSS_LIMIT` – Default `3`.  
- `VIX_SHUTDOWN_THRESHOLD` – Default `35`.  
- `CORRELATION_LIMIT` – Default `0.7`.

### Strategy Settings

- `ENABLE_MULTI_TIMEFRAME` – `true|false`.  
- `ENABLE_REGIME_FILTER` – `true|false`.  
- `ENABLE_EARNINGS_FILTER` – `true|false`.  
- `REGIME_LOOKBACK_DAYS` – Default `60`.

### Validation & Monitoring

- `ENABLE_DRIFT_MONITORING` – `true|false`.  
- `DRIFT_CHECK_FREQUENCY_HOURS` – Default `24`.  
- `BACKTEST_WALK_FORWARD_ENABLED` – `true|false`.  
- `DATA_QUALITY_CHECKS_ENABLED` – `true|false`.

### Performance Optimization

- `CACHE_TTL_MINUTES` – Default `5`.  
- `MAX_CONCURRENT_REQUESTS` – Default `5`.  
- `REQUEST_TIMEOUT_SECONDS` – Default `30`.

### Development / Debug

- `DEBUG_MODE` – `true|false`.  
- `ENABLE_AUTO_TRADE` – Default `false` (keep `false` for safety).  
- `PAPER_TRADING_ONLY` – Default `true`.  
- `MOCK_API_CALLS` – `true|false` for running tests offline.

---

## Usage Guide

### Daily Workflow

1. **Environment ready**: `.env` configured, Docker or local Python running.  
2. **09:30 IST**: `scheduler.py` (or `scheduler` service) runs the daily scan.  
3. **Market movers**: Top instruments are selected based on volume/volatility filters.  
4. **Intel gathering**: `research_engine` fetches SEC filings, news, and social sentiment.  
5. **Technicals**: `data_engine` computes indicators and price action structures.  
6. **Oracle**: `brain.consult_oracle()` calls GPT‑4o with curated intel and numerics.  
7. **Risk checks**: Circuit breakers and position sizing rules are applied.  
8. **Paper trade**: If allowed, a trade is executed in `portfolio.py`.  
9. **Alerts**: High-confidence signals trigger Discord alerts (if enabled).  
10. **Monitoring**: Drift and performance metrics are logged and visible in the dashboard.

---

### Dashboard Pages

- 🏠 **Home**  
  Live chart, current Oracle verdict, latest signals, and intel snippets.

- 📊 **Performance**  
  Equity curve, drawdowns, win rate, average R-multiple, and risk-adjusted metrics.

- 🔬 **Research**  
  Detailed view of intel per ticker (SEC filings, news headlines, social sentiment, links).

- ⚙️ **Strategy Lab**  
  A/B test strategies, adjust filters (timeframes, regimes, events), see leaderboard.

- 🛡️ **Risk Monitor**  
  Portfolio heat map, per-asset exposure, correlation matrix, circuit breaker status.

- 📈 **Backtest**  
  Interactive backtest runner with walk-forward splits and performance summaries.

---

## Module Reference

### Core 8 Modules

- `config.py`  
  - Loads `.env`, validates required variables, and initializes structured logging.  
  - Exposes `get_settings()` and `get_logger(name)`.

- `research_engine.py`  
  - Fetches SEC filings (10-K, 10-Q, 8-K, 13F) and caches them to `data/cache/sec/`.  
  - Integrates optional Reddit, Twitter/X, and news feeds when keys are provided.

- `data_engine.py`  
  - Uses `yfinance` or other data providers to pull OHLCV data.  
  - Computes indicators (RSI, MACD, EMA, ATR, Bollinger Bands, etc.) via `pandas-ta`.

- `brain.py`  
  - Wraps GPT‑4o API calls with strict Pydantic schema (`OracleDecision`).  
  - Applies deterministic risk overrides (e.g., force WAIT if overbought + hype).  
  - Ensures all stop-loss / take-profit math is done in Python, not in the LLM.

- `dashboard/app_v2.py`  
  - Streamlit multi-page dashboard.  
  - Uses Plotly for dark-themed charts with overlays and markers.  
  - Displays signals, portfolio, risk metrics, and backtest summaries.

- `alerts.py`  
  - Sends Discord webhooks for high-confidence signals.  
  - Redacts sensitive values and handles rate limits gracefully.

- `portfolio.py`  
  - Manages `Portfolio` state (positions, P&L) with atomic writes to `data/portfolio.json`.  
  - Records append-only trade history in `data/trade_history.csv`.  
  - Provides helper functions to compute realized/unrealized P&L and exposure.

- `scheduler.py`  
  - Orchestrates the daily workflow at configured times (`schedule` or long-running loop).  
  - In Docker Compose, can be run as a dedicated service (`alpha-prime-scheduler`).

---

### Risk Management Modules

- `position_sizer.py` – ATR-based position sizing respecting max risk per trade and portfolio risk caps.  
- `circuit_breakers.py` – Daily loss limit, consecutive loss halts, VIX-based shutdown logic.  
- `correlation_monitor.py` – Computes rolling correlation matrix and flags clusters.  
- `risk_metrics.py` – Portfolio VaR, beta, and other heat metrics.

### Profitability Boosters

- `regime_detector.py` – Classifies regimes (bull, bear, range, high-vol).  
- `multi_timeframe.py` – Ensures alignment across 1H / 4H / Daily.  
- `event_calendar.py` – Integrates earnings and key macro events.  
- `strategy_registry.py` – Registry and A/B test harness for strategies.  
- `execution_optimizer.py` – TCA and slippage modeling hooks.

### Intelligence Modules

- `insider_tracker.py` – Tracks SEC Form 4 insider trades.  
- `options_flow.py` – Analyzes unusual options activity if data available.  
- `sector_analysis.py` – Sector/industry-relative strength and rotation.  
- `sentiment_aggregator.py` – Fuses multiple sentiment sources into composite scores.

### Validation Modules

- `walk_forward.py` – Walk-forward optimization loops over time windows.  
- `drift_monitor.py` – Monitors model and data drift versus historical baselines.  
- `backtest_validator.py` – Enforces out-of-sample tests and holds out data.  
- `data_quality.py` – Checks for missing bars, outliers, and corporate action anomalies.

---

## Risk Parameters

| Parameter                    | Default | Description                                                |
|-----------------------------|---------|------------------------------------------------------------|
| `MAX_RISK_PER_TRADE_PCT`    | 2.0     | Max % of portfolio equity risked per trade                |
| `MAX_PORTFOLIO_RISK_PCT`    | 6.0     | Max aggregate open risk across all positions              |
| `DAILY_LOSS_LIMIT_PCT`      | 3.0     | Circuit breaker if intraday loss exceeds this level       |
| `CONSECUTIVE_LOSS_LIMIT`    | 3       | Halt after N consecutive losing trades                    |
| `VIX_SHUTDOWN_THRESHOLD`    | 35      | Stop opening new trades when volatility index exceeds     |
| `CORRELATION_LIMIT`         | 0.7     | Max allowed correlation between held positions            |
| `STARTING_CASH`             | 10000.0 | Initial paper trading bankroll                            |
| `POSITION_SIZE_METHOD`      | ATR     | Method for computing position size                        |

All risk rules are **configurable but conservative by default**. Do not relax them without understanding the implications.

---

## Data & Timezone

- All **timestamps are stored in UTC** using ISO 8601 strings.  
- Display in the UI uses `TIMEZONE` (default `Asia/Kolkata`).  
- Market hours defaults: 09:30–15:30 IST, matching Indian cash equity session.  
- Data feeds: primary (e.g., `yfinance`), optional backup providers (AlphaVantage, etc.).  

Data quality rules:

- Stale data is detected and logged.  
- Missing bars may be forward-filled only when appropriate (e.g., minor gaps).  
- Corporate actions (splits, dividends) are detected and flagged.

---

## 🛠️ Development

### Project Structure (High Level)

```text
alpha-prime/
  app.py
  brain.py
  data_engine.py
  research_engine.py
  alerts.py
  portfolio.py
  scheduler.py
  config.py

  dashboard/
    app_v2.py
    ...

  risk/
    position_sizer.py
    circuit_breakers.py
    correlation_monitor.py
    risk_metrics.py

  intelligence/
    insider_tracker.py
    options_flow.py
    sector_analysis.py
    sentiment_aggregator.py

  validation/
    walk_forward.py
    drift_monitor.py
    backtest_validator.py
    data_quality.py

  data/
    portfolio.json
    trade_history.csv
    cache/
    .gitkeep

  logs/
    alpha_prime.log
    .gitkeep

  tests/
    unit/
    integration/
    fixtures/

  scripts/
    run_backtest.py
    generate_report.py

  .env.example
  .gitignore
  .dockerignore
  Dockerfile
  docker-compose.yml
  requirements.txt
  pyproject.toml
  README.md
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=. --cov-report=term-missing
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint
flake8

# Type check
mypy .
```

### Adding a New Strategy

Patterns:

- Implement a new strategy in `strategy_registry.py` with a clear interface.  
- Use `data_engine` for technicals and `research_engine` for intel.  
- Add tests in `tests/unit/strategy/` and integration coverage in `tests/integration/`.  
- Register the strategy in `StrategyRegistry` so it appears in the Strategy Lab dashboard.

---

## Testing Strategy

- **Unit tests** for risk/position sizing, P&L calculations, and circuit breakers.  
- **Integration tests** for end-to-end scan → oracle → trade flow using mocked APIs.  
- **No real API calls** in CI – use fixtures under `tests/fixtures/` (JSON, CSV).  
- Coverage reports written to `htmlcov/` and CLI summary.

Key philosophy:

- Trading functions **never raise** in production paths; they log errors and return safe defaults.  
- External API failures must not crash the process; they degrade gracefully and trigger warnings.  

---

## Deployment

### Docker Compose (Included)

The repo includes:

- `Dockerfile` – production-ready image (Python 3.11 slim, non-root, healthcheck).  
- `.dockerignore` – optimized context, excludes data/logs/secrets.  
- `docker-compose.yml` – defines three services:

Services:

- `dashboard` – Streamlit UI (`alpha-prime-dashboard`, port 8501).  
- `scheduler` – Background trading cycle runner (`alpha-prime-scheduler`).  
- `watchdog` – Optional health monitor (`alpha-prime-watchdog`).

Basic commands:

```bash
# Start all
docker-compose up -d

# Check status
docker-compose ps

# Stop
docker-compose down
```

### Other Targets (Patterns)

- AWS ECS / Fargate: use the Docker image and appropriate task definitions.  
- Cloud Run / App Runner: containerized deployment with health checks on `/_stcore/health`.  
- On-prem: Docker or direct systemd services calling `app_v2` and `scheduler`.

---

## Troubleshooting

| Issue                                           | Possible Cause / Fix                                                                 |
|------------------------------------------------|--------------------------------------------------------------------------------------|
| `ModuleNotFoundError: pandas_ta`               | Run `pip install pandas-ta --no-cache-dir` and ensure venv is active                |
| Playwright browser not found                   | Run `playwright install chromium` (or `--force` if retrying)                        |
| `OPENAI_API_KEY` not found                     | Check `.env` and ensure Docker `env_file: .env` or local env exports                |
| OpenAI rate limit errors                       | Reduce frequency, enable retries (tenacity), or upgrade API plan                    |
| Portfolio state looks corrupted                | Stop services, restore from `backups/`, or delete and rebuild paper portfolio       |
| Dashboard shows no data                        | Ensure `scheduler.py` has run and that data providers/API keys are valid            |
| Healthcheck failing in Docker                  | Confirm `/_stcore/health` endpoint and that port `8501` is exposed correctly        |
| Timezone mismatches                            | Ensure `TIMEZONE=Asia/Kolkata` and system clock is correct                          |

---

## Roadmap

- [ ] Live broker integration (Zerodha / Groww) under strict SEBI compliance (paper-only now).  
- [ ] Options trading support (strategies + risk-specific modules).  
- [ ] Multi-asset support (indices, FX, crypto – research-only mode).  
- [ ] Enhanced execution simulation (venue-specific slippage and latency models).  
- [ ] Strategy marketplace and plugin API.  
- [ ] Optional mobile/React-based frontend consuming a backend API.  

---

## Contributing

Contributions are welcome as long as they respect the **safety-first** philosophy and avoid:

- Any ToS-violating scraping or login bypass automation.  
- Any “guaranteed profits” marketing or unsafe defaults.  

Typical flow:

1. Fork the repo on GitHub.  
2. Create a feature branch: `feat/my-feature`.  
3. Add tests and keep coverage high.  
4. Run `black`, `isort`, `flake8`, `mypy` before committing.  
5. Open a pull request with a clear description and rationale.

Consider adding or updating:

- `docs/` – for deeper architectural explanations.  
- `tests/fixtures/` – for new API mocks.  
- `CHANGELOG.md` – with versioned entries.

---

## Security & Compliance

- No browser automation for login bypass; only official APIs and permitted endpoints.  
- Secrets are **never** baked into Docker images or committed to git.  
- Environment validation at startup ensures necessary keys and directories exist.  
- Designed with **SEBI AI/ML** considerations in mind: current implementation is **paper-only**.  
- Logging redacts sensitive tokens and webhook URLs.

If you discover a security issue, please open a **private** channel (email) instead of a public issue.

---

## 📄 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

**Disclaimer:**  
This software is for **educational and research** purposes only and does **not** constitute financial advice. Trading and investing involve substantial risk of loss. Past performance does not guarantee future results. You are solely responsible for any use of this software.

---

## Acknowledgments

- OpenAI – for the GPT‑4o API powering the Oracle layer.  
- The `pandas`, `pandas-ta`, `numpy`, `scipy`, `statsmodels`, `yfinance` and `quantstats` communities.  
- Streamlit maintainers for an excellent rapid dashboarding framework.  
- Contributors to `evidently`, `scikit-learn`, and other OSS libraries used in this project.

---

## Support & Community

- 📧 Email: [support@alpha-prime.dev](mailto:support@alpha-prime.dev)  
- 🐛 Issues: <https://github.com/yourusername/alpha-prime/issues>  
- 📖 Docs: (placeholder) <https://docs.alpha-prime.dev/>  
- 💬 Discord: (placeholder invite) <https://discord.gg/your-server-code>

If you deploy or extend ALPHA-PRIME in your own research stack, consider sharing anonymized learnings and improvements via issues or pull requests.
```