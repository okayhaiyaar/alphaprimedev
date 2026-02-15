"""
============================================================
ALPHA-PRIME v2.0 - Configuration Management
============================================================
Centralized configuration, environment validation, and logging.

This module MUST be imported before any other ALPHA-PRIME module.
It handles:
- Environment variable loading (.env)
- Settings validation (fail-fast on missing critical keys)
- Global logger configuration (console + rotating file)
- Startup health checks

Usage:
    from config import get_settings, get_logger

    settings = get_settings()
    logger = get_logger(__name__)

    logger.info(f"Using OpenAI model: {settings.openai_model}")

CLI Validation:
    python config.py  # Validates environment and displays summary
============================================================
"""

import logging
import logging.handlers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ──────────────────────────────────────────────────────────
# SETTINGS DATA CLASS
# ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration settings loaded from environment.

    All settings are loaded from a .env file or environment variables.
    Required settings will raise ValueError if missing or invalid.
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CRITICAL - AI & APIs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.7

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NOTIFICATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    discord_webhook_url: Optional[str] = None
    alert_min_confidence: int = 80

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SOCIAL SENTIMENT APIs (Optional)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "ALPHA-PRIME:v2.0"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SYSTEM CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"
    market_open_time: str = "09:30"
    market_close_time: str = "15:30"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FILE PATHS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    portfolio_path: str = "data/portfolio.json"
    trade_history_path: str = "data/trade_history.csv"
    cache_dir: str = "data/cache"
    log_dir: str = "logs"
    backup_dir: str = "backups"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADING PARAMETERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    starting_cash: float = 10000.0
    commission_per_trade: float = 0.0
    max_risk_per_trade_pct: float = 2.0
    max_portfolio_risk_pct: float = 6.0
    daily_loss_limit_pct: float = 3.0
    position_size_method: str = "ATR"  # ATR | FIXED | KELLY

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RISK MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    circuit_breaker_enabled: bool = True
    consecutive_loss_limit: int = 3
    vix_shutdown_threshold: float = 35.0
    correlation_limit: float = 0.7

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    enable_multi_timeframe: bool = True
    enable_regime_filter: bool = True
    enable_earnings_filter: bool = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDATION & MONITORING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    enable_drift_monitoring: bool = True
    drift_check_frequency_hours: int = 24

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PERFORMANCE OPTIMIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cache_ttl_minutes: int = 5
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 30

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEVELOPMENT / DEBUG
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    debug_mode: bool = False
    enable_auto_trade: bool = False
    paper_trading_only: bool = True
    mock_api_calls: bool = False

    def __post_init__(self) -> None:
        """
        Validate critical settings after initialization.

        Raises:
            ValueError: If any required setting is missing or invalid.
        """
        # Validate OpenAI key presence
        if not self.openai_api_key:
            raise ValueError(
                "CRITICAL ERROR: OPENAI_API_KEY not set.\n"
                "ALPHA-PRIME cannot function without this key.\n"
                "Get your key at: https://platform.openai.com/api-keys\n"
                "Add it to your .env file as OPENAI_API_KEY=sk-..."
            )

        # Basic format validation (do not log full key)
        if not self.openai_api_key.startswith("sk-"):
            raise ValueError(
                "INVALID OPENAI_API_KEY format.\n"
                "Key should start with 'sk-' or 'sk-proj-'."
            )

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {self.log_level}. "
                f"Must be one of: {', '.join(valid_levels)}"
            )

        # Validate risk parameters
        if not (0 < self.max_risk_per_trade_pct <= 10):
            raise ValueError("max_risk_per_trade_pct must be between 0 and 10.")

        if not (0 < self.daily_loss_limit_pct <= 20):
            raise ValueError("daily_loss_limit_pct must be between 0 and 20.")

        if self.position_size_method not in ("ATR", "FIXED", "KELLY"):
            raise ValueError("position_size_method must be one of: ATR, FIXED, KELLY.")

        if self.correlation_limit <= 0 or self.correlation_limit > 1:
            raise ValueError("correlation_limit must be in the range (0, 1].")


# ──────────────────────────────────────────────────────────
# GLOBAL STATE (Singleton Pattern)
# ──────────────────────────────────────────────────────────

_SETTINGS: Optional[Settings] = None
_LOGGER_INITIALIZED: bool = False


def _parse_bool(value: str, default: bool) -> bool:
    """
    Parse a boolean value from environment string.

    Args:
        value: Raw environment string value.
        default: Default boolean if value is empty.

    Returns:
        bool: Parsed boolean value.
    """
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_int(value: Optional[str], default: int) -> int:
    """
    Parse an integer from environment string.

    Args:
        value: Raw environment string value.
        default: Default integer if value is empty.

    Returns:
        int: Parsed integer value.
    """
    return int(value) if value not in (None, "") else default


def _parse_float(value: Optional[str], default: float) -> float:
    """
    Parse a float from environment string.

    Args:
        value: Raw environment string value.
        default: Default float if value is empty.

    Returns:
        float: Parsed float value.
    """
    return float(value) if value not in (None, "") else default


def load_settings() -> Settings:
    """
    Load and cache settings from environment.

    This is called automatically by get_settings() on first access.
    It will:
    - Ensure a .env file exists at project root.
    - Load environment variables via python-dotenv.
    - Build and validate a Settings instance.

    Returns:
        Settings: Validated settings object.

    Raises:
        FileNotFoundError: If .env file is missing.
        ValueError: If required settings are missing or invalid.
    """
    global _SETTINGS

    if _SETTINGS is not None:
        return _SETTINGS

    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            "CRITICAL ERROR: .env file not found in project root.\n"
            "Create one by copying .env.example:\n"
            "  cp .env.example .env\n"
            "Then edit .env and set OPENAI_API_KEY and other values."
        )

    # Load environment variables from .env (override existing env by default here)
    load_dotenv(env_path, override=True)

    settings = Settings(
        # AI & APIs
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        openai_max_tokens=_parse_int(os.getenv("OPENAI_MAX_TOKENS"), 4000),
        openai_temperature=_parse_float(os.getenv("OPENAI_TEMPERATURE"), 0.7),
        # Notifications
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
        alert_min_confidence=_parse_int(os.getenv("ALERT_MIN_CONFIDENCE"), 80),
        # Social APIs
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID") or None,
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET") or None,
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "ALPHA-PRIME:v2.0"),
        # System
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        timezone=os.getenv("TIMEZONE", "Asia/Kolkata"),
        market_open_time=os.getenv("MARKET_OPEN_TIME", "09:30"),
        market_close_time=os.getenv("MARKET_CLOSE_TIME", "15:30"),
        # Paths
        portfolio_path=os.getenv("PORTFOLIO_PATH", "data/portfolio.json"),
        trade_history_path=os.getenv("TRADE_HISTORY_PATH", "data/trade_history.csv"),
        cache_dir=os.getenv("CACHE_DIR", "data/cache"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        backup_dir=os.getenv("BACKUP_DIR", "backups"),
        # Trading
        starting_cash=_parse_float(os.getenv("STARTING_CASH"), 10000.0),
        commission_per_trade=_parse_float(os.getenv("COMMISSION_PER_TRADE"), 0.0),
        max_risk_per_trade_pct=_parse_float(os.getenv("MAX_RISK_PER_TRADE_PCT"), 2.0),
        max_portfolio_risk_pct=_parse_float(os.getenv("MAX_PORTFOLIO_RISK_PCT"), 6.0),
        daily_loss_limit_pct=_parse_float(os.getenv("DAILY_LOSS_LIMIT_PCT"), 3.0),
        position_size_method=os.getenv("POSITION_SIZE_METHOD", "ATR").upper(),
        # Risk
        circuit_breaker_enabled=_parse_bool(
            os.getenv("CIRCUIT_BREAKER_ENABLED"), True
        ),
        consecutive_loss_limit=_parse_int(
            os.getenv("CONSECUTIVE_LOSS_LIMIT"), 3
        ),
        vix_shutdown_threshold=_parse_float(
            os.getenv("VIX_SHUTDOWN_THRESHOLD"), 35.0
        ),
        correlation_limit=_parse_float(os.getenv("CORRELATION_LIMIT"), 0.7),
        # Strategy
        enable_multi_timeframe=_parse_bool(
            os.getenv("ENABLE_MULTI_TIMEFRAME"), True
        ),
        enable_regime_filter=_parse_bool(
            os.getenv("ENABLE_REGIME_FILTER"), True
        ),
        enable_earnings_filter=_parse_bool(
            os.getenv("ENABLE_EARNINGS_FILTER"), True
        ),
        # Validation
        enable_drift_monitoring=_parse_bool(
            os.getenv("ENABLE_DRIFT_MONITORING"), True
        ),
        drift_check_frequency_hours=_parse_int(
            os.getenv("DRIFT_CHECK_FREQUENCY_HOURS"), 24
        ),
        # Performance
        cache_ttl_minutes=_parse_int(os.getenv("CACHE_TTL_MINUTES"), 5),
        max_concurrent_requests=_parse_int(
            os.getenv("MAX_CONCURRENT_REQUESTS"), 5
        ),
        request_timeout_seconds=_parse_int(
            os.getenv("REQUEST_TIMEOUT_SECONDS"), 30
        ),
        # Debug
        debug_mode=_parse_bool(os.getenv("DEBUG_MODE"), False),
        enable_auto_trade=_parse_bool(os.getenv("ENABLE_AUTO_TRADE"), False),
        paper_trading_only=_parse_bool(os.getenv("PAPER_TRADING_ONLY"), True),
        mock_api_calls=_parse_bool(os.getenv("MOCK_API_CALLS"), False),
    )

    _SETTINGS = settings
    return _SETTINGS


def get_settings() -> Settings:
    """
    Get the cached global Settings instance.

    Returns:
        Settings: Global settings object loaded from environment.
    """
    if _SETTINGS is None:
        return load_settings()
    return _SETTINGS


# ──────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ──────────────────────────────────────────────────────────


def setup_logging() -> None:
    """
    Configure global logging for ALPHA-PRIME.

    Creates:
    - Console handler (INFO+)
    - Rotating file handler (DEBUG+, 5MB, 3 backups)
    - Format: [TIMESTAMP] [LEVEL] module_name - message

    Called automatically by get_logger() on first access.
    """
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    settings = get_settings()

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("alpha_prime")
    root_logger.setLevel(settings.log_level)
    root_logger.propagate = False
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "alpha_prime.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True

    root_logger.info("=" * 70)
    root_logger.info("ALPHA-PRIME Logging System Initialized")
    root_logger.info("Log Level: %s", settings.log_level)
    root_logger.info("Log File: %s", log_dir / "alpha_prime.log")
    root_logger.info("=" * 70)


def get_logger(name: str = "alpha_prime") -> logging.Logger:
    """
    Get a configured logger instance for a module.

    Args:
        name: Logger name, usually __name__ of the calling module.

    Returns:
        logging.Logger: Logger configured with global handlers.
    """
    if not _LOGGER_INITIALIZED:
        setup_logging()

    if name == "alpha_prime":
        return logging.getLogger("alpha_prime")
    return logging.getLogger("alpha_prime").getChild(name)


# ──────────────────────────────────────────────────────────
# STARTUP VALIDATION
# ──────────────────────────────────────────────────────────


def validate_environment() -> bool:
    """
    Run comprehensive environment validation checks.

    Validates:
    - Settings loading and critical keys
    - Directory existence (data, logs, cache, backups)
    - Basic write permissions in log directory

    Returns:
        bool: True if all checks pass.

    Raises:
        RuntimeError: If any critical check fails.
    """
    logger = get_logger(__name__)

    try:
        settings = get_settings()
        logger.info("Settings loaded successfully.")

        # Log partial key only (first 6 chars)
        masked_key = f"{settings.openai_api_key[:6]}***"
        logger.info("OpenAI API key: %s", masked_key)

        if settings.discord_webhook_url:
            logger.info("Discord webhook configured.")
        else:
            logger.warning(
                "Discord webhook NOT configured. High-confidence alerts will be disabled."
            )

        if settings.reddit_client_id:
            logger.info("Reddit API configured.")
        else:
            logger.info("Reddit API not configured. Social sentiment will be limited.")

        # Ensure directories exist
        dirs_to_ensure = {
            "portfolio": Path(settings.portfolio_path).parent,
            "trade_history": Path(settings.trade_history_path).parent,
            "cache": Path(settings.cache_dir),
            "logs": Path(settings.log_dir),
            "backups": Path(settings.backup_dir),
        }

        for label, path in dirs_to_ensure.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory ensured for %s: %s", label, path)

        logger.info("Data, cache, log, and backup directories are ready.")

        # Validate write permissions in log dir
        test_file = Path(settings.log_dir) / ".write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            logger.info("Write permissions verified in log directory.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Cannot write to log directory %s: %s", settings.log_dir, exc)
            raise

        return True

    except Exception as exc:  # noqa: BLE001
        logger.critical("Environment validation failed: %s", exc)
        raise RuntimeError(
            "Cannot start ALPHA-PRIME due to configuration errors."
        ) from exc


# ──────────────────────────────────────────────────────────
# CLI VALIDATION TOOL
# ──────────────────────────────────────────────────────────


def _print_summary(settings: Settings) -> None:
    """
    Print a human-readable configuration summary to stdout.

    Args:
        settings: Loaded Settings instance.
    """
    print("\nConfiguration Summary:")
    print(f"  - OpenAI Model: {settings.openai_model}")
    print(f"  - OpenAI Key: {'SET' if settings.openai_api_key else 'MISSING'}")
    print(
        f"  - Discord Webhook: "
        f"{'SET' if settings.discord_webhook_url else 'NOT SET'}"
    )
    print(f"  - Log Level: {settings.log_level}")
    print(f"  - Portfolio Path: {settings.portfolio_path}")
    print(f"  - Trade History Path: {settings.trade_history_path}")
    print(f"  - Cache Dir: {settings.cache_dir}")
    print(f"  - Log Dir: {settings.log_dir}")
    print(f"  - Backup Dir: {settings.backup_dir}")
    print(f"  - Paper Trading: {'ENABLED' if settings.paper_trading_only else 'DISABLED'}")
    print(
        f"  - Circuit Breakers: "
        f"{'ENABLED' if settings.circuit_breaker_enabled else 'DISABLED'}"
    )
    print(f"  - Max Risk/Trade: {settings.max_risk_per_trade_pct}%")
    print(f"  - Daily Loss Limit: {settings.daily_loss_limit_pct}%")
    print(f"  - Timezone: {settings.timezone}")
    print(
        f"  - Market Hours (Local): {settings.market_open_time} - {settings.market_close_time}"
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ALPHA-PRIME Configuration Validator")
    print("=" * 70 + "\n")

    try:
        validate_environment()
        cfg = get_settings()
        _print_summary(cfg)
        print("\nAll checks passed. ALPHA-PRIME is ready to launch.\n")
    except Exception as exc:  # noqa: BLE001
        print("\nFATAL ERROR:")
        print(exc)
        print("\nFix the configuration issues above and re-run `python config.py`.\n")
        raise SystemExit(1)
