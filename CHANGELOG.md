# Changelog

All notable changes to **ALPHA-PRIME** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned

- Live broker integration (Zerodha/Groww APIs) with SEBI-compliant controls
- Options trading support (strategies, risk, and greeks-aware sizing)
- Multi-asset expansion (indices, crypto, forex; research mode first)
- Mobile/Web companion app for monitoring and approvals
- Real-time strategy optimization and parameter search
- Community strategy marketplace and plugin system
- Deeper TCA and execution venue modeling
- Pluggable feature store for ML-based models
- Advanced anomaly detection on intel and price feeds

---

## [2.0.0] - 2026-02-07

### Added

- **Risk Management Suite**
  - Dynamic ATR-based position sizing with hard caps on risk per trade
  - Circuit breakers for daily loss limits, consecutive loss halts, and VIX-based shutdown
  - Real-time correlation monitoring across open positions with configurable thresholds
  - Portfolio heat, VaR-style metrics, and exposure overview panels

- **Intelligence Modules**
  - SEC Form 4 insider trading tracker with caching and evidence linking
  - Unusual options activity detection hooks for flow-based signals
  - Sector rotation and peer performance analysis for relative strength
  - Multi-source sentiment aggregator (Reddit, X/Twitter, news when configured)

- **Validation Framework**
  - Walk-forward optimization engine for rolling window evaluations
  - Model drift detection for both data distributions and performance metrics
  - Out-of-sample backtesting framework with configurable splits
  - Automated data quality checks for missing data, outliers, and corporate actions

- **Profitability Enhancements**
  - Market regime detector (Bull, Bear, Range, HighVol) with configurable lookback
  - Multi-timeframe confluence filters (1H + 4H + Daily) for stronger signals
  - Earnings and macro calendar integration with pre/post event blocking
  - Transaction cost analysis (TCA) and slippage modeling scaffolding
  - Strategy A/B testing framework with registry and leaderboard support

- **Dashboard v2 (Multi-page)**
  - 🏠 **Home**: Live signals, price charts, and Oracle verdict stream
  - 📊 **Performance**: Equity curve, drawdown stats, and performance metrics
  - 🔬 **Research**: Deep intel viewer with links to filings, news, and sentiment
  - ⚙️ **Strategy Lab**: A/B testing interface and strategy comparison
  - 🛡️ **Risk Monitor**: Portfolio heatmap, exposure, and circuit breaker status
  - 📈 **Backtest**: Interactive backtest runner with walk-forward summaries

- **Infrastructure & Operations**
  - Dockerfile and `.dockerignore` for production-ready container builds
  - `docker-compose.yml` to orchestrate dashboard, scheduler, and watchdog services
  - Watchdog service for process and health monitoring
  - Automated backup system for portfolio state and key artifacts
  - Rotating logs with configurable size-based retention
  - Health check endpoints for container orchestration (Streamlit health integration)

- **Configuration & Observability**
  - Comprehensive `.env.example` with grouped environment variables and documentation
  - Centralized logging configuration with structured JSON logs
  - IST (Asia/Kolkata) timezone support with UTC storage of all timestamps
  - Enhanced config validation on startup (env variables, directories, permissions)

### Changed

- Upgraded from a single-page Streamlit dashboard to a multi-page v2 interface
- Migrated from basic backtesting to a full walk-forward validation framework
- Enhanced Oracle logic with deterministic conflict resolution rules and safety overrides
- Replaced fixed position sizing with ATR-based dynamic allocation respecting risk caps
- Restructured the project into a modular architecture (core + risk + intelligence + validation)
- Improved scheduler design for more reliable daily market cycles and retries
- Refined caching strategy for SEC filings, news, and sentiment with TTL controls
- Updated logging format to structured logs with consistent module naming

### Deprecated

- Deprecated ad-hoc, hard-coded strategy definitions in favor of `strategy_registry.py`
- Deprecated single-timeframe-only filters in favor of `multi_timeframe` confluence module
- Deprecated monolithic research methods in favor of `get_god_tier_intel(ticker)` interface

### Removed

- Removed legacy single-page dashboard layout and associated legacy plotting helpers
- Removed non-structured, print-based logging in favor of centralized logger usage
- Removed prototype backtest scripts that did not support walk-forward or OOS validation

### Fixed

- Fixed portfolio corruption issues by enforcing atomic JSON writes via temp files and `os.replace`
- Fixed race conditions in concurrent data fetches that caused partial indicator computation
- Fixed memory leaks in long-running scheduler processes by tightening lifecycle and cleanup
- Fixed stale cache invalidation logic for SEC filings, news, and sentiment sources
- Fixed missing error handling in network retry loops with more specific exceptions and logging
- Fixed timezone inconsistencies between scheduler, dashboard, and stored timestamps
- Fixed handling of duplicate bars and minor data gaps in time series

### Security

- Enforced non-root Docker user (`alpha`, UID 1000) in production containers
- Redacted API keys, webhook URLs, and other secrets in all logs
- Standardized on environment variables for secrets (no secrets in code or images)
- Hardened Docker image by using `python:3.11-slim` and minimizing installed system packages
- Improved error logging to avoid leaking sensitive context in stack traces

---

## [1.0.0] - 2025-11-15

### Added

- Initial public release of ALPHA-PRIME
- Core 4-module architecture:
  - **Hunter** (`research_engine.py`): SEC and news research
  - **Mathematician** (`data_engine.py`): Market data and indicators
  - **Oracle** (`brain.py`): LLM-based decision engine
  - **Dashboard** (`app.py`): Streamlit single-page UI
- Paper trading engine with JSON-based portfolio persistence
- Discord webhook notifications for high-confidence signals
- Automated scheduler for daily market cycles at market open
- Basic technical indicators:
  - RSI, MACD, EMAs, Bollinger Bands, and a small set of moving averages
- SEC EDGAR filings integration using official endpoints
- Social sentiment integration via Reddit (PRAW) for ticker mentions
- GPT‑4o Oracle decision engine with fixed JSON-like output structure
- `.env`-based configuration for API keys and environment settings
- Basic logging system with configurable log level and log file output

### Changed

- Consolidated configuration into a dedicated config module with helper functions
- Standardized dataset formats (OHLCV) across data fetches and indicators
- Improved Discord alert formatting with clearer signal summaries and links
- Refined Oracle prompts to reduce hallucinations and enforce risk cautions

### Deprecated

- Deprecated manually-edited JSON configuration in favor of `.env` + typed settings
- Deprecated experimental console-only mode in favor of Streamlit dashboard

### Removed

- Removed early experimental Jupyter-based orchestration notebooks
- Removed hard-coded ticker lists in favor of scanner-driven market movers

### Fixed

- Fixed SEC filing parser encoding issues on some filings
- Fixed yfinance data fetch problems due to inconsistent intervals
- Fixed missing error handling in social sentiment fetch routines
- Fixed sporadic dashboard crashes caused by incomplete data frames

### Security

- Introduced `.gitignore` entries for `.env`, portfolio, and logs
- Restricted access to secrets via `.env` instead of hardcoded placeholders
- Reduced logging verbosity for external API responses to avoid leaking payloads

### Known Issues

- No walk-forward validation or robust backtest tooling (overfitting risk)
- Fixed position sizing ignores volatility and risk per trade
- No circuit breakers – portfolio can experience large drawdowns in simulation
- Single-timeframe analysis (daily) without intraday confluence
- No model drift detection or monitoring
- Portfolio state corruption possible if process interrupted mid-write
- Limited error recovery for failed API calls and rate limiting

---

## [0.2.0-beta] - 2025-09-01

### Added

- Discord alert integration for new trade signals and critical warnings
- Paper trading portfolio tracker with basic P&L calculations
- Automated scheduler prototype for daily scan and signal generation
- Configuration management via `.env` and simple loader utilities
- Expanded technical indicator set (additional EMAs and momentum measures)

### Changed

- Switched from purely manual LLM prompting to automated LLM orchestration flow
- Improved technical indicator accuracy and alignment with reference implementations
- Refactored research pipeline into clearer steps (filings, news, social)
- Cleaned up dashboard layout with better separation of sections

### Deprecated

- Deprecated manual command-line driven trade workflow in favor of scheduled scans
- Deprecated older indicator functions superseded by `pandas-ta` usage

### Removed

- Removed early prototype notebooks used only for exploratory analysis
- Removed experimental CSV-based config files superseded by `.env`

### Fixed

- Fixed SEC filing parser encoding and date parsing issues
- Fixed memory leaks in long-running yfinance data fetch loops
- Fixed improper exception handling in some network requests
- Fixed issues with repeated indicator computation on overlapping windows

### Security

- Tightened `.gitignore` around logs, cache, and backup files
- Clarified `.env.example` to avoid accidental key commits

---

## [0.1.0-alpha] - 2025-07-15

### Added

- Initial proof-of-concept release of ALPHA-PRIME
- Basic research module combining SEC filings and news headlines
- Simple technical analysis:
  - RSI and MACD on daily bars
- Manual LLM prompting for trade decisions via console interface
- Console-based interface showing latest intel and suggested actions
- Minimal configuration using hardcoded constants and simple settings file

### Changed

- Iterated on LLM prompts to include more numeric context and risk notes
- Improved news parsing to extract headlines and URLs more reliably

### Deprecated

- Deprecated early, hardcoded ticker universe as the project evolved
- Deprecated direct console-only flow in favor of a future web dashboard

### Removed

- Removed temporary scripts for one-off research tasks
- Removed unused experimental notebooks and CSV dumps

### Fixed

- Fixed issues with SEC fetching pagination and basic rate limiting
- Fixed date alignment bugs between SEC events and price data
- Fixed early crashes when no filings were available for a given ticker

### Security

- Introduced first `.gitignore` for `.env` and local data directories
- Documented the requirement to keep API keys out of source control

---

***

## Version History Summary

| Version      | Release Date | Highlights                                              |
|-------------|--------------|---------------------------------------------------------|
| 2.0.0       | 2026-02-07   | Full production system with risk management + validation |
| 1.0.0       | 2025-11-15   | Initial public release with core 4-module architecture |
| 0.2.0-beta  | 2025-09-01   | Automation, scheduler, alerts, expanded indicators     |
| 0.1.0-alpha | 2025-07-15   | Proof of concept with manual LLM prompting             |

[Unreleased]: https://github.com/yourusername/alpha-prime/compare/v2.0.0...HEAD  
[2.0.0]: https://github.com/yourusername/alpha-prime/compare/v1.0.0...v2.0.0  
[1.0.0]: https://github.com/yourusername/alpha-prime/compare/v0.2.0-beta...v1.0.0  
[0.2.0-beta]: https://github.com/yourusername/alpha-prime/compare/v0.1.0-alpha...v0.2.0-beta  
[0.1.0-alpha]: https://github.com/yourusername/alpha-prime/releases/tag/v0.1.0-alpha
