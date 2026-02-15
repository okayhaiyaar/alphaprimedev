***

# ALPHA-PRIME v2.0 – Master Guide  
**Architecture, Operations, Risk & Research Manual**

Last updated: February 11, 2026  

***

## Table of Contents

1. [Part I – Executive Overview](#part-i--executive-overview)  
   1. [Chapter 1 – What ALPHA‑PRIME v2.0 Is](#chapter-1--what-alpha-prime-v20-is)  
   2. [Chapter 2 – Core Design Principles](#chapter-2--core-design-principles)  
   3. [Chapter 3 – High‑Level Data Flow](#chapter-3--high-level-data-flow)  

2. [Part II – System Architecture & Design](#part-ii--system-architecture--design)  
   4. [Chapter 4 – Component Map](#chapter-4--component-map)  
   5. [Chapter 5 – Directory & Module Structure](#chapter-5--directory--module-structure)  
   6. [Chapter 6 – Execution Model & Concurrency](#chapter-6--execution-model--concurrency)  
   7. [Chapter 7 – Data Model & Schema](#chapter-7--data-model--schema)  

3. [Part III – Installation, Configuration & Environments](#part-iii--installation-configuration--environments)  
   8. [Chapter 8 – Environments](#chapter-8--environments)  
   9. [Chapter 9 – First‑Time Setup (Summary)](#chapter-9--first-time-setup-summary)  
   10. [Chapter 10 – Configuration System](#chapter-10--configuration-system)  

4. [Part IV – Operation & Runbooks](#part-iv--operation--runbooks)  
   11. [Chapter 11 – Daily Operations Overview](#chapter-11--daily-operations-overview)  
   12. [Chapter 12 – Runbooks](#chapter-12--runbooks)  

5. [Part V – Strategies, Research & Backtesting](#part-v--strategies-research--backtesting)  
   13. [Chapter 13 – Strategy Framework](#chapter-13--strategy-framework)  
   14. [Chapter 14 – Built‑in Strategies](#chapter-14--built-in-strategies)  
   15. [Chapter 15 – Backtesting Engine](#chapter-15--backtesting-engine)  
   16. [Chapter 16 – Research Workflow](#chapter-16--research-workflow)  

6. [Part VI – Risk, Prop Firms & Capital Management](#part-vi--risk-prop-firms--capital-management)  
   17. [Chapter 17 – Risk Model](#chapter-17--risk-model)  
   18. [Chapter 18 – Prop Firm Constraints](#chapter-18--prop-firm-constraints)  
   19. [Chapter 19 – Position Sizing](#chapter-19--position-sizing)  
   20. [Chapter 20 – Profit Extraction & Scaling](#chapter-20--profit-extraction--scaling)  

7. [Part VII – Performance, Monitoring & Observability](#part-vii--performance-monitoring--observability)  
   21. [Chapter 21 – Logging & Metrics](#chapter-21--logging--metrics)  
   22. [Chapter 22 – Dashboards & Reports](#chapter-22--dashboards--reports)  
   23. [Chapter 23 – Health & Stress Testing](#chapter-23--health--stress-testing)  

8. [Part VIII – Maintenance, Testing & Deployment](#part-viii--maintenance-testing--deployment)  
   24. [Chapter 24 – Test Suite](#chapter-24--test-suite)  
   25. [Chapter 25 – Maintenance Routines](#chapter-25--maintenance-routines)  
   26. [Chapter 26 – Deployment Patterns](#chapter-26--deployment-patterns)  

9. [Part IX – Troubleshooting & Incident Response](#part-ix--troubleshooting--incident-response)  
   27. [Chapter 27 – Troubleshooting Index](#chapter-27--troubleshooting-index)  
   28. [Chapter 28 – Incident Playbooks](#chapter-28--incident-playbooks)  
   29. [Chapter 29 – Post‑Mortems](#chapter-29--post-mortems)  

10. [Part X – Security, Compliance & Secrets](#part-x--security-compliance--secrets)  
    30. [Chapter 30 – Secrets Management](#chapter-30--secrets-management)  
    31. [Chapter 31 – Operational Security](#chapter-31--operational-security)  
    32. [Chapter 32 – Compliance & Record‑Keeping](#chapter-32--compliance--record-keeping)  

11. [Part XI – Extension, Customization & Roadmap](#part-xi--extension-customization--roadmap)  
    33. [Chapter 33 – Adding New Strategies](#chapter-33--adding-new-strategies)  
    34. [Chapter 34 – New Broker Integrations](#chapter-34--new-broker-integrations)  
    35. [Chapter 35 – Feature Roadmap](#chapter-35--feature-roadmap)  

12. [Part XII – Personal Playbook & Journal Templates](#part-xii--personal-playbook--journal-templates)  

13. [Appendices](#appendices)  
    - [Appendix A – Glossary](#appendix-a--glossary)  
    - [Appendix B – CLI Cheat Sheet](#appendix-b--cli-cheat-sheet)  
    - [Appendix C – Config Reference](#appendix-c--config-reference)  
    - [Appendix D – Diagrams Collection](#appendix-d--diagrams-collection)  
    - [Appendix E – FAQ](#appendix-e--faq)  

***

## Part I – Executive Overview

### Chapter 1 – What ALPHA‑PRIME v2.0 Is

ALPHA‑PRIME v2.0 is a modular, multi‑strategy algo trading platform for Indian markets (NSE, NIFTY, BANKNIFTY, major equities) with production‑grade risk management, execution, and observability. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

**Core capabilities:**

- Multi‑strategy trading (mean reversion, breakout scalper, momentum) with pluggable strategy modules. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Broker integration layer (Zerodha/Upstox style) with REST + WebSocket connectivity for orders and live data. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Real‑time risk management, including per‑trade risk, daily loss limits, drawdown control, correlation caps, and kill switches. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Persistent data layer (PostgreSQL) plus Redis cache for stateful intraday operation and historical analytics. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Backtesting, walk‑forward, Monte Carlo, and reporting tooling via `scripts/backtest.py`, `optimize_strategy.py`, and report generators. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Dashboard UI (`dashboard/app_v2.py`) for monitoring, lightweight manual control, and report browsing. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

**Problems this solves for you (trader + builder):**

- Removes manual execution risk: orders, sizing, and exits are fully systematic.  
- Encodes your risk rules so your future self cannot easily override them on tilt.  
- Provides a single platform for research, backtesting, paper trading, and live deployment.  
- Creates a durable operational footprint (DB, logs, reports) so you can audit and iterate strategies with data.

***

### Chapter 2 – Core Design Principles

1. **Reliability over raw speed**  
   - Target: robust execution during Indian trading hours with modest latency, not HFT microsecond games.  
   - Preference for Python + PostgreSQL + Redis with careful design rather than prematurely optimizing every micro‑path. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

2. **Risk‑first design**  
   - Every trade path passes through risk checks: capital at risk, limits, correlation, time filters, volatility filters. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
   - Prop‑firm rules (daily loss, total drawdown) are treated as first‑class constraints configured in `risk_config.yaml`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

3. **Modularity**  
   - Clear separation between strategy logic, risk management, order execution, portfolio tracking, data storage, and presentation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
   - Strategies never talk to broker directly; they emit signals consumed by the risk and execution layers.

4. **Observability**  
   - Structured logs (system, trades, risk, errors, performance) with consistent patterns and log levels. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
   - Persistent trade/portfolio history and derived performance metrics to validate live behavior vs backtests. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

5. **Automation with human‑in‑the‑loop**  
   - System can run unattended, but dashboard + CLI allow you to stop trading, emergency‑flatten, or adjust configs safely. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
   - Critical operations (live mode, emergency stop, restore) require explicit confirmation or are wrapped in clear scripts.

6. **Separation of research vs production**  
   - Research experiments live in branch strategies, different configs, and non‑live modes (paper, backtest). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
   - Production configs are conservative, version‑controlled, and gated by checklists and tests (see Chapter 24, Chapter 25).

***

### Chapter 3 – High‑Level Data Flow

#### 3.1 Real‑Time Trading Loop

```text
Market Data (WebSocket / Polling)
        │
        ▼
  Strategy Engine
  (core/strategy_engine.py)
        │  emits signals
        ▼
   Risk Manager
   (core/risk_manager.py)
        │  approves/rejects, sizes positions
        ▼
  Execution Engine
  (core/execution_engine.py)
        │  orders → Broker API
        ▼
      Broker
  (integrations/zerodha, upstox)
        │  fills / updates
        ▼
  Order Manager & Portfolio
  (core/order_manager.py, core/portfolio.py)
        │   state updates → DB + Redis
        ▼
PostgreSQL + Redis
(database/*, integrations/redis_client.py)
        │
        ▼
 Dashboard & Reports
 (dashboard/app_v2.py, scripts/generate_report.py)
```

#### 3.2 Backtest / Offline Loop

```text
Historical Market Data (data/market_data)
        │
        ▼
 Backtest Runner (scripts/backtest.py)
        │  feeds candles/ticks to strategies
        ▼
   Strategy Engine (offline mode)
        │  signals
        ▼
  Simulated Risk & Execution
        │  pseudo orders, fills, slippage
        ▼
  In-Memory / File-Based Results
        │
        ├─► CSV/DB (reports/backtest/)
        └─► Plots / Metrics / HTML/PDF reports
```

Backtests bypass live broker APIs and Redis; they operate on recorded data and simulated fills while reusing the same strategy logic and risk model where possible. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

## Part II – System Architecture & Design

### Chapter 4 – Component Map

At a coarse level:

- **Data layer** – PostgreSQL schema (`trades`, `orders`, `portfolio_history`, `strategy_performance`) plus Redis for real‑time cache. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Strategy layer** – `strategies/` and `core/strategy_engine.py` implementing signal generation based on configured parameters. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Risk layer** – `core/risk_manager.py` applying risk_limits, position sizing, time/volatility filters, and circuit breakers. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Execution layer** – `core/execution_engine.py` and `core/order_manager.py` translating approved signals into broker orders. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Portfolio layer** – `core/portfolio.py` and DB models tracking positions, P&L, margin, and equity history. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Integration layer** – `integrations/` for Zerodha/Upstox clients, Redis wrapper, email/Telegram alerts. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Presentation layer** – `dashboard/app_v2.py` and pages/components implementing the UI. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Each component has a clear contract:

- Strategies output **signals** (desired side, instrument, stop/take‑profit, confidence).  
- Risk transforms signals into **approved trade instructions** (or rejects them).  
- Execution and order management transform instructions into **orders** and track **fills**.  
- Portfolio updates **positions** and writes to DB; dashboard reads from DB/Redis.

***

### Chapter 5 – Directory & Module Structure

ALPHA‑PRIME root tree (curated):

```text
alpha-prime-v2/
├── config/
│   ├── strategies/
│   ├── risk_config.yaml
│   ├── broker_config.yaml
│   ├── trading_hours.yaml
│   └── general_config.yaml
├── core/
├── strategies/
├── integrations/
├── database/
├── data/
├── dashboard/
├── scripts/
├── tests/
├── logs/
├── reports/
├── docs/
└── project root files (.env, requirements.txt, README.md, pyproject.toml)
```

#### 5.1 `config/`

- **What lives here:** all configuration not meant to change intraday: risk rules, strategy parameters, market hours, broker endpoints. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `risk_config.yaml` – global risk limits, sizing method, time/vol filters, circuit breakers. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `strategies/*.yaml` – per‑strategy parameters, symbol lists, overrides. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `broker_config.yaml` – base URLs, timeouts, product types.  
  - `trading_hours.yaml` – session times, holidays, half‑days.  
- **How to extend:** add new strategy YAMLs, create presets (e.g. `risk_config_prop_challenge.yaml`) and symlink/copy into `risk_config.yaml`.

#### 5.2 `core/`

- **What lives here:** core engine components, independent of any specific strategy or broker. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `engine.py` – orchestrates the trading loop.  
  - `strategy_engine.py` – loads and runs strategies.  
  - `risk_manager.py` – pre‑trade risk, sizing, circuit breakers.  
  - `execution_engine.py` – order dispatch to `integrations`.  
  - `portfolio.py` – portfolio state and P&L.  
  - `order_manager.py` – life‑cycle of orders.  
  - `market_hours.py` – is_market_open logic. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **How to extend:** add new risk rules, extend order types, support partial fill handling, add new signals.

#### 5.3 `strategies/`

- **What lives here:** implementations of trading logic that operate on data and emit signals. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `base_strategy.py` – defines `on_start`, `on_bar`, `on_tick`, `on_stop` and the interface to the engine. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `mean_reversion.py`, `breakout_scalper.py`, `momentum.py` – concrete implementations. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **How to extend:** subclass `BaseStrategy`, implement lifecycle methods, add a YAML config under `config/strategies`, wire into `configure_strategies.py`.

#### 5.4 `integrations/`

- **What lives here:** external adapters. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `zerodha/client.py`, `zerodha/websocket.py`, `zerodha/auth.py` – broker client, WS connector, login helper. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `upstox/` – analogous for Upstox.  
  - `redis_client.py` – Redis wrapper.  
  - `email_alerts.py` – SMTP/email integration.  
- **How to extend:** implement new broker clients by following the Zerodha adapter pattern, add new alert transports (e.g. Slack).

#### 5.5 `database/`

- **What lives here:** DB connectivity and ORM models. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `connection.py` – `create_engine` with pooling, connection settings. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `models.py` – SQLAlchemy models for `trades`, `orders`, `portfolio_history`, `strategy_performance`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `migrations/` – Alembic scripts.  
  - `queries.py` – common analytics queries.  

#### 5.6 `dashboard/`

- **What lives here:** Streamlit dashboard. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `app_v2.py` – entry point, routing.  
  - `pages/overview.py`, `pages/strategies.py`, `pages/performance.py`, `pages/settings.py`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `components/` – reusable UI pieces.  

#### 5.7 `scripts/`

- **What lives here:** operational CLI tooling. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Key files:**  
  - `setup.sh` – initial setup / DB migrations. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `health_check.py` – verifies DB, Redis, broker, disk, memory. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `run_engine.py` – single entry for run modes (monitor, live, paper, stop, emergency‑stop). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `backtest.py` – backtesting runner. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `optimize_strategy.py` – parameter optimization. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `generate_report.py` – daily/weekly/monthly reports. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `backup.py` – backup automation. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - `reset_portfolio.py`, `stress_test.py`, `analyze_logs.py`, `cleanup_logs.py`, `db_maintenance.py`, `archive_old_data.py`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

#### 5.8 `tests/`, `logs/`, `reports/`, `docs/`, root files

- `tests/` – smoke, unit, integration tests.  
- `logs/` – logs separated by concern (`alpha_prime.log`, `trades.log`, `risk.log`, `errors.log`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- `reports/` – output artifacts from `generate_report.py` and backtests. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- `docs/` – field manuals and this master guide. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- `.env` – all secrets, never committed; `.env.example` for templates. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 6 – Execution Model & Concurrency

ALPHA‑PRIME uses an asynchronous, event‑driven model within Python, layered around the engine and strategy/risk/execution components. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

- **Async I/O:** broker HTTP calls (via aiohttp or similar) and WebSocket streams are non‑blocking; market events and order updates are consumed via callbacks/tasks. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Strategy scheduling:** strategies are invoked on bars or ticks within the event loop; signals are passed into risk manager queues.  
- **Risk + execution concurrency:** multiple signals, risk checks, and orders may be in flight simultaneously; concurrency control is handled via internal queues and DB transactions (e.g. to avoid double‑spending capital).  
- **Background tasks:**  
  - Periodic health checks (`health_check.py`, or internal tasks in engine).  
  - Periodic portfolio snapshots into `portfolio_history`.  
  - Log rotation and maintenance scripts run out‑of‑band (`cron`, systemd timers).

**Conceptual concurrency diagram:**

```text
Event Loop
  ├─ Task: consume_market_data()
  ├─ Task: run_strategies_on_bar()
  ├─ Task: apply_risk_checks()
  ├─ Task: submit_orders_to_broker()
  ├─ Task: handle_broker_responses()
  ├─ Task: update_portfolio_and_db()
  └─ Task: emit_metrics_and_logs()
```

Blocking operations (e.g. DB writes) are isolated and tuned via connection pooling and pre‑pinging to avoid stalls. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 7 – Data Model & Schema

Conceptual entities:

- **Order** – intent to transact a symbol with quantity, price, type, status. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Trade** – logical trade lifecycle (entry, exit, P&L) linked to one or more orders. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Portfolio snapshot** – time‑stamped view of total equity, cash, margin, P&L. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Strategy performance** – aggregated metrics per strategy per day. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Persistent schema is in PostgreSQL (see full DDL in `04_additional_notes.md`). Redis caches high‑frequency, transient state (latest quotes, current open positions state, etc.) to reduce DB load and keep dashboards/snappy logic. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

## Part III – Installation, Configuration & Environments

### Chapter 8 – Environments

Three main environments:

| Env          | Mode                            | Behavior                                                   |
|--------------|----------------------------------|------------------------------------------------------------|
| `development`| Local, dev DB, paper broker      | Verbose logs, paper trading only, relaxed risk defaults    |
| `paper`      | Live data, simulated orders      | Real‑time, no real money; full risk model, reports         |
| `live`       | Real broker + real capital       | Strict risk enforcement, conservative defaults, alerts     |

Control via `.env`:

```bash
ENVIRONMENT=development   # or paper, production/live
LOG_LEVEL=DEBUG           # dev
ENABLE_PAPER_TRADING=true
ENABLE_REAL_TRADING=false
```

Production uses `ENVIRONMENT=production`, `LOG_LEVEL=INFO`, `ENABLE_REAL_TRADING=true`, `ENABLE_PAPER_TRADING=false` with real broker keys. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 9 – First‑Time Setup (Summary)

Full step‑by‑step is in `01_quick_start_install.md`; here’s the conceptual summary:

- **Python & venv:** create and activate venv, install `requirements.txt`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Database:** install PostgreSQL, create `alpha_prime` DB, run migrations via `scripts/setup.sh --db-only`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Redis:** install and run Redis (`brew services start redis` / `systemctl start redis`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Configs:** copy `.env.example` → `.env`, fill credentials; tune `risk_config.yaml` and strategy YAMLs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Smoke tests:** run `pytest tests/smoke/ -v` before any live experiments. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 10 – Configuration System

Core config surfaces:

#### 10.1 Environment variables (`.env`)

See full listing in `04_additional_notes.md`. This captures: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

- DATABASE_URL, REDIS_URL  
- Broker keys and secrets  
- Email/Telegram config  
- Environment and feature flags  

#### 10.2 Risk config (`config/risk_config.yaml`)

Already heavily specified in `04_additional_notes.md` and `03_profit_maximization_guide.md`. Your main presets: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

- **Dev/playground:** higher per‑trade risk allowed, but `ENABLE_REAL_TRADING=false`.  
- **Prop challenge:** 0.25% per trade, daily max loss ≈ 40% of firm limit, total max DD ≈ 40% of firm limit, strong circuit breakers (see Chapter 18).  
- **Live personal:** similar settings, but drawdown and scaling per your personal rules.

#### 10.3 Strategy configs (`config/strategies/*.yaml`)

Describe each strategy’s symbols, parameters, overrides, trading hours, and minimal liquidity/spread conditions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

#### 10.4 Trading hours & holidays (`trading_hours.yaml`)

Central source of truth for `is_market_open` logic and blocking trading on holidays/special days. This drives `core/market_hours.py` and CLI tools. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

## Part IV – Operation & Runbooks

### Chapter 11 – Daily Operations Overview

A “healthy” day with ALPHA‑PRIME looks like:

- **Pre‑market (8:45–9:15):** environment checks, `health_check.py`, broker verification, risk config sanity, strategy selection for today; see `02_daily_startup_checklist.md`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Market hours (9:15–3:30):** engine in monitor → live, dashboard running, periodic monitoring (every 10–30 minutes), only intervening on true incidents. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **End‑of‑day (3:25–4:00):** stop new trades, close intraday positions, generate daily report and backup, optionally journal; see `02_daily_startup_checklist.md` and `04_additional_notes.md`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

For profit and risk framing, refer back to `03_profit_maximization_guide.md` (summarized and formalized in Part VI). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 12 – Runbooks

Each runbook: Goal → Inputs → Steps → Outputs.

#### 12.1 Morning Startup Runbook (short form)

- **Goal:** from powered‑off to READY TO TRADE in ~10 minutes.  
- **Inputs:** powered machine, internet, broker credentials, DB + Redis installed.  
- **Steps:** see `02_daily_startup_checklist.md` for exact checkboxes and commands. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Outputs:** engine in monitor/live mode, dashboard up, no critical errors, risk and strategy configs verified.

#### 12.2 Intraday Monitoring Runbook

- **Goal:** ensure the system is behaving as expected; intervene only on incident‑class events.  
- **Checks:**  
  - Engine logs: no ERROR/CRITICAL spikes.  
  - Dashboard: status “TRADING ACTIVE”, P&L within expected variance.  
  - Risk logs: no frequent “limit near” warnings.  
- **Actions on anomalies:** follow Chapter 28 incident playbooks.

#### 12.3 End‑of‑Day Shutdown & Reconciliation

- **Goal:** cleanly end trading, reconcile with broker, persist artifacts.  
- **Steps:**  
  - Stop new trades, optionally close all open positions.  
  - Generate daily report and inspect key metrics.  
  - Run backup (`backup.py --daily --compress`).  
  - Stop engine + dashboard, deactivate venv.  
- **Outputs:** no open positions/orders, reports generated, backup stored.

***

## Part V – Strategies, Research & Backtesting

### Chapter 13 – Strategy Framework

`strategies/base_strategy.py` defines the contract:

- Lifecycle: `on_start()`, `on_bar()`, `on_tick()`, `on_stop()`.  
- Input: data slices (bars, ticks, indicators), config parameters.  
- Output: signal objects (direction, size hint or risk, stop loss, take profit, metadata).

Strategies do not talk to the broker, DB, or Redis directly; they only emit signals to the strategy engine, which forwards them into risk + execution (see Chapter 4). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 14 – Built‑in Strategies

You can expand this with full descriptions; outline:

- **Mean Reversion (NIFTY/BANKNIFTY):**  
  - Uses deviations from moving average, enters when price > N std dev away, exits near mean or stop. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
  - Strong in range/choppy regimes, weak in trending breakouts.  

- **Breakout Scalper:**  
  - Uses breakout of recent ranges with tight stops and small R multiple.  
  - Works best in moderate volatility, gets whipsawed in noisy chop.

- **Momentum/Trend:**  
  - Slower timeframes, uses higher‑highs/lowers, but you typically avoid this in prop challenge due to higher DD.

Each strategy’s YAML lists parameters and last backtest metrics; see `config/strategies/*.yaml` and DB `strategy_performance`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 15 – Backtesting Engine

`scripts/backtest.py`:

- Reads historical market data from `data/market_data/`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Instantiates strategy classes with config parameters.  
- Simulates order fills under latency/slippage assumptions.  
- Outputs results into `reports/backtest/` and optionally DB.

Example:

```bash
python scripts/backtest.py --strategy MeanReversion_NIFTY50 \
  --start 2023-01-01 \
  --end 2026-02-11 \
  --metrics all \
  --report detailed
```

Supports:

- Portfolio tests (multiple strategies).  
- Walk‑forward testing (`--walk-forward`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Monte Carlo simulations (`--monte-carlo --simulations 10000`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Metrics: annualized return, max DD, worst day, win rate, profit factor, Sharpe, max consecutive losses (see `04_additional_notes.md`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 16 – Research Workflow

End‑to‑end path:

1. Idea → pseudo‑code and quick prototype in `strategies/dev/`.  
2. Backtest (3+ years) with realistic costs and slippage.  
3. Walk‑forward (multiple rolling windows) to avoid overfitting.  
4. Regime tests (bull, bear, sideways).  
5. Monte Carlo to estimate distribution of returns and drawdowns.  
6. Paper trading (30+ days) in `mode paper`.  
7. Graduated live rollout (10% → 25% → 50% → 100% risk allocation).  

The safe experimentation pipeline is detailed in `04_additional_notes.md > Safe Experimentation Guide`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

## Part VI – Risk, Prop Firms & Capital Management

### Chapter 17 – Risk Model

Core quantities:

- **Per‑trade risk** – % of equity at risk per trade (loss if SL hit).  
- **Daily max loss** – % of equity allowed to be lost in a single day.  
- **Total max drawdown** – % from equity peak beyond which trading stops.  
- **Correlation limits** – number of correlated exposures allowed simultaneously.  
- **Circuit breakers** – rules based on consecutive losses, half‑daily limit, etc.  
- **Kill switches** – immediate stops when thresholds are breached (configured in `kill_switch` block). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

`core/risk_manager.py` conceptually enforces:

- Per‑signal checks vs available risk budget and per‑trade risk.  
- Checks of daily P&L and drawdown (using DB/Redis states).  
- Time filters (open/close minutes, blackout windows).  
- Volatility filters (India VIX thresholds).  

Config is in `config/risk_config.yaml` (see detailed sample in `04_additional_notes.md`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 18 – Prop Firm Constraints

Abstract prop rules:

- Daily loss limit (e.g. 5%).  
- Max or trailing drawdown (e.g. 10%).  
- Profit target (e.g. 8–10% in 30 days).  
- Min trading days.  

**Mapping into `risk_config.yaml`:**

- Set `daily_max_loss` ≈ 40% of firm daily limit (2% for a 5% firm limit).  
- Set `max_drawdown` ≈ 40% of firm total DD (4% for a 10% limit).  
- Configure `kill_switch` thresholds slightly below these to stop early. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Create presets:

- `risk_config_prop_challenge.yaml` – ultra‑conservative (0.25% per trade).  
- `risk_config_funded_conservative.yaml` – maybe 0.30–0.35% with same daily/DD.  

***

### Chapter 19 – Position Sizing

Two main concepts:

- **Fixed fractional sizing:** risk a fixed % of current equity each trade (default `method: fixed_fractional`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Kelly criterion:** theoretical optimal fraction, but only used to inform upper bounds; you trade at 25–50% of Kelly at most.

Example configuration (already present):

```yaml
position_sizing:
  method: "fixed_fractional"
  kelly_fraction: 0.5
```

During challenges, you stick to 0.25% fixed fractional; for matured accounts you may go to 0.35–0.4% after long consistent performance (see `03_profit_maximization_guide.md`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 20 – Profit Extraction & Scaling

Rules:

- Scale up risk **only** after stable performance over 3–6 months.  
- Increase per‑trade risk by at most 0.05% per month (e.g. from 0.25% → 0.30%).  
- Withdraw 50–70% of monthly profits to lock gains and reduce psychological pressure (see `03_profit_maximization_guide.md`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

This part is essentially a structured restatement of the profit and scaling content from `03_profit_maximization_guide.md` tied back to risk config knobs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

## Part VII – Performance, Monitoring & Observability

### Chapter 21 – Logging & Metrics

Logging taxonomy and file layout are defined in `04_additional_notes.md`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Key practices:

- Use `tail -f logs/alpha_prime.log` for overall events; use `grep` to filter strategies or error types. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- Treat `logs/errors.log` as your high‑signal incident list.  
- Use `logs/risk.log` for risk rejections and near‑limit behavior.  
- Use `logs/trades.log` to match with broker statements.

***

### Chapter 22 – Dashboards & Reports

Dashboard pages (`dashboard/pages/*.py`): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

- **Overview:** portfolio equity, P&L, open positions, engine status.  
- **Strategies:** which are enabled, live performance vs backtest.  
- **Performance:** equity curves, per‑strategy metrics, histograms.  
- **Settings:** read‑only view of key config parameters.

Reports (`scripts/generate_report.py`) produce daily/weekly/monthly documents summarizing performance and risk metrics, and are stored under `reports/`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 23 – Health & Stress Testing

`health_check.py`:

- Verifies DB connectivity, Redis, broker API, market hours logic, disk space, and memory. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

`stress_test.py`:

- Simulates heavy load on strategies, DB, and Redis to reveal bottlenecks (see details in `04_additional_notes.md`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

You should run full `--full` health checks regularly and `stress_test.py` quarterly.

***

## Part VIII – Maintenance, Testing & Deployment

### Chapter 24 – Test Suite

Tests are split into:

- **Smoke:** quick checks the system basically runs (`tests/smoke`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Unit:** component‑level tests for strategies, risk, and DB. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)
- **Integration:** end‑to‑end flows: engine + broker mocks + DB. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Run:

```bash
pytest tests/smoke/ -v
pytest tests/unit/ -v
pytest tests/integration/ -v
```

Add tests whenever you change a critical module.

***

### Chapter 25 – Maintenance Routines

All checklists and scripts are already defined in `04_additional_notes.md` under “Maintenance Tasks”; this chapter just collates them as policy: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

- Daily: report review, error log scan, journal update.  
- Weekly: weekly report, performance review, log cleanup, weekly backup.  
- Monthly: health audit, DB maintenance, dependency review, API key rotations check, monthly backup.  
- Quarterly: system audit, stress test, re‑backtest, disaster recovery drill, security audit.

***

### Chapter 26 – Deployment Patterns

Recommended patterns:

- **Single local machine:** as you’re currently doing – venv + CLI, manual start/stop with checklists.  
- **VPS / server:** systemd service or tmux/screen sessions wrapping `run_engine.py` and `dashboard.app_v2`, plus cron for backups and log cleanup. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Rolling out updates:

1. Pull repo changes.  
2. Run smoke + unit + integration tests.  
3. If passing, restart engine during non‑market hours.  

***

## Part IX – Troubleshooting & Incident Response

### Chapter 27 – Troubleshooting Index

Index is already detailed in `04_additional_notes.md` (“Common Errors & Solutions”). For quick access, maintain a table mapping: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

| Symptom                          | Likely cause                  | Reference           |
|----------------------------------|-------------------------------|---------------------|
| Cannot connect to DB            | Postgres down / wrong .env    | Section DB errors   |
| Redis WRONGTYPE / OOM           | Key type mismatch / memory    | Redis errors        |
| TokenException from broker      | Expired/invalid token         | Broker API errors   |
| No trades while signals exist   | Risk rejection                | Strategy / risk     |
| Dashboard port in use           | Old process still bound       | System errors       |

Fill and maintain this over time with line‑numbers/anchors into `04_additional_notes.md`.

***

### Chapter 28 – Incident Playbooks

For each scenario, define:

- Immediate actions.  
- Diagnostics.  
- Long‑term fix.

Use and extend the emergency and disaster recovery content from `04_additional_notes.md` (emergency flatten, restore, broker contact details). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 29 – Post‑Mortems

Template:

```text
Incident: [short name]
Date:
Duration:
Impact (P&L, uptime):

Timeline:
- HH:MM: first symptom
- HH:MM: detection
- HH:MM: mitigation
- HH:MM: recovery

Root Causes:
- Technical:
- Process:
- Human:

What Went Well:
What Went Poorly:
Permanent Fixes:
- Code changes:
- Config changes:
- Process changes:

Follow-Up Checks:
- Tests added:
- Docs updated:
```

Keep major incidents documented in `docs/post_mortems/` or a section in this file.

***

## Part X – Security, Compliance & Secrets

### Chapter 30 – Secrets Management

Handled via `.env` with strict file permissions (`chmod 600 .env`) and explicit .gitignore. For off‑machine backups, encrypt `.env` (e.g. with GPG) and only store `.env.gpg`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Rotate:

- Broker keys every ~90 days.  
- DB passwords every ~180 days.  
- Email/Telegram tokens yearly. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 31 – Operational Security

- Locked, updated OS.  
- Encrypted disk; strong login credentials.  
- Avoid running engine on general‑purpose “browsing” machine during trading hours.  
- Protect broker credentials and TOTP secrets; never echo them in shells. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 32 – Compliance & Record‑Keeping

ALPHA‑PRIME stores:

- All trades and orders in `trades` and `orders` tables.  
- Equity curve in `portfolio_history`.  
- Strategy metrics in `strategy_performance`.  
- Logs in `logs/` and reports in `reports/`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

This is enough to reconstruct trading history, verify risk adherence, and meet basic audit needs for personal/proprietary trading.

***

## Part XI – Extension, Customization & Roadmap

### Chapter 33 – Adding New Strategies

Process:

1. Create a new class in `strategies/` subclassing `BaseStrategy`.  
2. Implement lifecycle methods.  
3. Add config YAML under `config/strategies`.  
4. Wire into `configure_strategies.py` and tests.  
5. Backtest → walk‑forward → Monte Carlo → paper → graduated live (as per Chapter 16, plus `Safe Experimentation Guide`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

### Chapter 34 – New Broker Integrations

Follow the pattern in `integrations/zerodha`:

- Implement a client with standardized methods: `place_order`, `cancel_order`, `get_positions`, `get_margins`, `subscribe_ticks`.  
- Implement an auth flow.  
- Provide WebSocket wrappers for live data and order updates.  
- Map broker‑specific error codes into generic error types.

***

### Chapter 35 – Feature Roadmap

A living list for future you:

- Additional markets (BSE, global indices, FX).  
- More advanced regime detection and auto risk scaling.  
- ML‑based signal modules, integrated with `strategies/` and `data/models/`.  
- Better dashboard UX (heatmaps, alerts, drill‑downs).  
- CI/CD for tests + packaging this as a private pip distributable.

***

## Part XII – Personal Playbook & Journal Templates

This part aggregates and centralizes all your personal templates from `03_profit_maximization_guide.md` and `04_additional_notes.md` into one place: prop templates, risk presets, daily/weekly/monthly review, mistake logs, “never again” rules. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

Use the templates already written there and copy them directly into this section when you actually maintain the file in your repo.

***

## Appendices

### Appendix A – Glossary

Populate using terms from `04_additional_notes.md` and your trading vocabulary: signal, order, fill, slippage, drawdown, Kelly, risk‑of‑ruin, etc. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

### Appendix B – CLI Cheat Sheet

Summarize `scripts/*.py` usage from `04_additional_notes.md` as a table. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

### Appendix C – Config Reference

Short, compressed explanations for `risk_config.yaml` fields, key strategy YAML keys, and `.env` variables. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

### Appendix D – Diagrams Collection

Copy the key ASCII diagrams (system architecture, data flow, experimentation pipeline) into one place for quick visual recall. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

### Appendix E – FAQ

Seed ideas from the troubleshooting and emergency sections of `04_additional_notes.md`: “What if DB is down?”, “What if Redis is full?”, “What if broker rejects orders?”, “Can I run this on a VPS?”, etc. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/2887426/561e819d-1ef2-4ca5-82ec-df9639929fce/paste.txt)

***

This gives you a coherent, strongly structured master guide with explicit cross‑links to the four field manuals and reuses all the technical detail already captured in `04_additional_notes.md`; you can now expand each chapter incrementally inside your repo as you iterate on ALPHA‑PRIME.