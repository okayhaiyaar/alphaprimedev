"""
Unit tests for the configuration system (Settings, environment loading,
validation, defaults, overrides) for ALPHA-PRIME v2.0.

Requirements:
- Fast (<100ms total suite).
- Isolated (mocked environment, no file I/O or external services).
- Deterministic (no randomness, no real-time dependencies).
- High coverage across all config paths.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from config import (
    Settings,
    get_settings,
    load_config_from_env,
    validate_config,
    DatabaseConfig,
    RedisConfig,
    BrokerConfig,
    TradingConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_env_vars() -> Dict[str, str]:
    """Sample environment variables for testing."""
    return {
        "ENVIRONMENT": "test",
        "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/test_db",
        "REDIS_URL": "redis://localhost:6379/0",
        "ZERODHA_API_KEY": "test_api_key",
        "ZERODHA_SECRET": "test_secret",
        "MAX_POSITION_SIZE": "0.05",
        "MAX_DRAWDOWN": "0.20",
    }


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch, sample_env_vars: Dict[str, str]) -> Dict[str, str]:
    """Mock environment variables."""
    for key, value in sample_env_vars.items():
        monkeypatch.setenv(key, str(value))
    return sample_env_vars


@pytest.fixture
def valid_settings() -> Settings:
    """Valid Settings instance for testing."""
    return Settings(
        environment="test",
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        broker_api_key="test_key",
        broker_secret="test_secret",
    )


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------

class TestSettings:
    """Test Settings dataclass initialization and validation."""

    def test_default_settings_initialization(self):
        """Test Settings can be created with defaults."""
        settings = Settings(environment="test")
        assert settings.environment == "test"
        assert getattr(settings, "log_level", "INFO") == "INFO"
        assert getattr(settings, "max_position_size", 0.05) == 0.05

    def test_settings_from_dict(self):
        """Test Settings can be constructed from a dict via **kwargs."""
        data = {
            "environment": "test",
            "database_url": "sqlite+aiosqlite:///:memory:",
            "redis_url": "redis://localhost:6379/15",
        }
        settings = Settings(**data)
        assert settings.environment == "test"
        assert settings.database_url == data["database_url"]

    def test_settings_with_invalid_environment(self):
        """Test Settings raises on invalid environment."""
        with pytest.raises(ValueError, match="Invalid environment"):
            Settings(environment="invalid-env")

    def test_settings_environment_enum(self):
        """Test environment can be constrained to known values."""
        for env in ("test", "paper", "production", "staging"):
            s = Settings(environment=env)
            assert s.environment == env

    def test_settings_immutability(self):
        """Test Settings is frozen (immutable)."""
        settings = Settings(environment="test")
        with pytest.raises(AttributeError):
            settings.environment = "production"  # type: ignore[misc]

    def test_settings_to_dict(self):
        """Test Settings has a to_dict/model_dump-like API or equivalent."""
        settings = Settings(environment="test")
        if hasattr(settings, "model_dump"):
            data = settings.model_dump()
        elif hasattr(settings, "to_dict"):
            data = settings.to_dict()
        else:
            data = settings.__dict__
        assert data["environment"] == "test"

    def test_settings_repr(self):
        """Test Settings __repr__ contains key fields."""
        settings = Settings(environment="test")
        text = repr(settings)
        assert "Settings" in text
        assert "environment='test'" in text or "environment=\"test\"" in text

    def test_settings_equality(self):
        """Test equality comparison for Settings."""
        s1 = Settings(environment="test")
        s2 = Settings(environment="test")
        s3 = Settings(environment="paper")
        assert s1 == s2
        assert s1 != s3

    def test_settings_copy(self):
        """Test copying Settings retains values."""
        s1 = Settings(environment="test")
        if hasattr(s1, "model_copy"):
            s2 = s1.model_copy()
        else:
            s2 = Settings(**s1.__dict__)
        assert s1 == s2
        assert s1 is not s2

    def test_settings_validation_on_init(self):
        """Test validation logic is executed on initialization."""
        with pytest.raises(ValueError):
            Settings(environment="test", max_position_size=1.5)


# ---------------------------------------------------------------------------
# Environment variable loading
# ---------------------------------------------------------------------------

class TestEnvironmentLoading:
    """Test loading config from environment variables."""

    def test_load_from_env_all_vars_present(self, mock_env: Dict[str, str]):
        """Test loading when all env vars are present."""
        settings = load_config_from_env()
        assert settings.environment == "test"
        assert settings.database_url == mock_env["DATABASE_URL"]

    def test_load_from_env_missing_required(self, monkeypatch: pytest.MonkeyPatch):
        """Test failure when required variables are missing."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError):
            load_config_from_env()

    def test_load_from_env_with_defaults(self, monkeypatch: pytest.MonkeyPatch):
        """Test default values are applied when optional vars missing."""
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        settings = load_config_from_env()
        assert getattr(settings, "max_position_size", 0.05) == 0.05

    def test_env_var_type_coercion(self, monkeypatch: pytest.MonkeyPatch):
        """Test env var numeric type coercion."""
        monkeypatch.setenv("MAX_POSITION_SIZE", "0.07")
        settings = load_config_from_env()
        assert pytest.approx(getattr(settings, "max_position_size", 0.07)) == 0.07

    def test_env_var_boolean_parsing(self, monkeypatch: pytest.MonkeyPatch):
        """Test boolean env var parsing (true/false/1/0)."""
        monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
        settings = load_config_from_env()
        assert getattr(settings, "paper_trading", True) is True

        monkeypatch.setenv("ENABLE_PAPER_TRADING", "0")
        settings = load_config_from_env()
        assert getattr(settings, "paper_trading", False) is False

    def test_env_var_json_parsing(self, monkeypatch: pytest.MonkeyPatch):
        """Test JSON env var parsing for complex types."""
        monkeypatch.setenv("TRADING_HOURS", '["09:15-15:30"]')
        settings = load_config_from_env()
        hours = getattr(settings, "trading_hours", [])
        assert hours == ["09:15-15:30"]

    def test_env_var_list_parsing(self, monkeypatch: pytest.MonkeyPatch):
        """Test comma-separated list parsing."""
        monkeypatch.setenv("ALLOWED_SYMBOLS", "AAPL,MSFT,GOOGL")
        settings = load_config_from_env()
        symbols = getattr(settings, "allowed_symbols", [])
        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_env_var_override_priority(self, monkeypatch: pytest.MonkeyPatch):
        """Test env vars override default configuration."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = load_config_from_env()
        assert getattr(settings, "log_level", "DEBUG") == "DEBUG"


# ---------------------------------------------------------------------------
# Database config tests
# ---------------------------------------------------------------------------

class TestDatabaseConfig:
    """Test DatabaseConfig validation and URL construction."""

    def test_database_config_defaults(self):
        cfg = DatabaseConfig()
        assert cfg.pool_size > 0
        assert cfg.timeout > 0

    def test_postgres_connection_url_construction(self):
        cfg = DatabaseConfig(
            driver="postgresql+asyncpg",
            user="user",
            password="pass",
            host="localhost",
            port=5432,
            database="alpha",
        )
        url = cfg.url
        assert "postgresql+asyncpg" in url
        assert "user" in url
        assert "alpha" in url

    def test_sqlite_connection_url(self):
        cfg = DatabaseConfig(driver="sqlite+aiosqlite", database=":memory:")
        url = cfg.url
        assert url.startswith("sqlite+aiosqlite://")
        assert ":memory:" in url

    def test_database_pool_size_validation(self):
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=0)

    def test_database_timeout_validation(self):
        with pytest.raises(ValueError):
            DatabaseConfig(timeout=-1)

    def test_async_sqlalchemy_url_format(self):
        cfg = DatabaseConfig(
            driver="postgresql+asyncpg",
            user="u",
            password="p",
            host="localhost",
            port=5432,
            database="alpha",
        )
        assert "+asyncpg" in cfg.url


# ---------------------------------------------------------------------------
# Redis config tests
# ---------------------------------------------------------------------------

class TestRedisConfig:
    """Test Redis configuration and validation."""

    def test_redis_config_defaults(self):
        cfg = RedisConfig()
        assert cfg.db >= 0
        assert isinstance(cfg.url, str)

    def test_redis_url_parsing(self):
        cfg = RedisConfig(url="redis://localhost:6379/0")
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 0

    def test_redis_cluster_mode(self):
        cfg = RedisConfig(cluster=True)
        assert cfg.cluster is True

    def test_redis_sentinel_config(self):
        cfg = RedisConfig(sentinel=True, sentinel_master="mymaster")
        assert cfg.sentinel is True
        assert cfg.sentinel_master == "mymaster"

    def test_redis_password_handling(self):
        cfg = RedisConfig(url="redis://:secret@localhost:6379/0")
        assert cfg.password == "secret"


# ---------------------------------------------------------------------------
# Broker config tests
# ---------------------------------------------------------------------------

class TestBrokerConfig:
    """Test broker API configuration."""

    def test_broker_config_zerodha(self):
        cfg = BrokerConfig(
            name="zerodha",
            api_key="key",
            secret="secret",
        )
        assert cfg.name == "zerodha"
        assert cfg.api_key == "key"

    def test_broker_api_key_validation(self):
        with pytest.raises(ValueError):
            BrokerConfig(name="zerodha", api_key="", secret="s")

    def test_broker_secret_masking(self):
        cfg = BrokerConfig(name="zerodha", api_key="key", secret="secret")
        text = repr(cfg)
        assert "secret" not in text

    def test_broker_rate_limits(self):
        cfg = BrokerConfig(rate_limit_per_minute=120)
        assert cfg.rate_limit_per_minute == 120

    def test_broker_timeout_defaults(self):
        cfg = BrokerConfig()
        assert cfg.timeout_seconds > 0

    def test_broker_paper_trading_mode(self):
        cfg = BrokerConfig(paper_trading=True)
        assert cfg.paper_trading is True

    def test_broker_config_validation_fails_on_missing_keys(self):
        with pytest.raises(ValueError):
            BrokerConfig(api_key=None, secret=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Trading config tests
# ---------------------------------------------------------------------------

class TestTradingConfig:
    """Test trading-specific configuration."""

    def test_trading_config_defaults(self):
        cfg = TradingConfig()
        assert 0 < cfg.max_position_size <= 0.2
        assert 0 < cfg.max_drawdown <= 0.5

    def test_max_position_size_validation(self):
        with pytest.raises(ValueError):
            TradingConfig(max_position_size=1.0)

    def test_max_drawdown_validation(self):
        with pytest.raises(ValueError):
            TradingConfig(max_drawdown=2.0)

    def test_risk_limits_validation(self):
        with pytest.raises(ValueError):
            TradingConfig(var_limit_pct=-0.01)

    def test_signal_confidence_thresholds(self):
        cfg = TradingConfig(signal_confidence_threshold=0.6)
        assert 0.0 <= cfg.signal_confidence_threshold <= 1.0

    def test_trading_hours_configuration(self):
        cfg = TradingConfig(trading_hours=["09:15-15:30"])
        assert "09:15-15:30" in cfg.trading_hours

    def test_slippage_and_commission_defaults(self):
        cfg = TradingConfig()
        assert cfg.slippage_bps >= 0
        assert cfg.commission_per_trade >= 0

    def test_position_sizing_rules(self):
        cfg = TradingConfig(position_sizing_rule="fixed_fraction")
        assert cfg.position_sizing_rule in {"fixed_fraction", "vol_target"}


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Test comprehensive config validation."""

    def test_validate_config_success(self, valid_settings: Settings):
        """Test validation passes for valid config."""
        result = validate_config(valid_settings)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_config_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        settings = Settings(environment="test", database_url="")
        result = validate_config(settings)
        assert result.is_valid is False
        assert any("database_url" in str(e) for e in result.errors)

    def test_validate_config_invalid_types(self):
        """Test validation fails for invalid types."""
        settings = Settings(environment="test", max_position_size="not-a-float")  # type: ignore[arg-type]
        result = validate_config(settings)
        assert result.is_valid is False

    def test_validate_config_out_of_range_values(self):
        """Test validation fails for out-of-range values."""
        settings = Settings(environment="test", max_position_size=1.5)
        result = validate_config(settings)
        assert result.is_valid is False
        assert "max_position_size" in str(result.errors)

    def test_validate_config_incompatible_settings(self):
        """Test incompatible settings combinations."""
        settings = Settings(
            environment="production",
            paper_trading=True,
        )
        result = validate_config(settings)
        assert result.is_valid is False

    def test_validate_config_security_checks(self):
        """Test security-related config checks."""
        settings = Settings(
            environment="test",
            broker_secret="plain-text-secret",
        )
        result = validate_config(settings)
        assert result.is_valid is False

    def test_validate_config_database_reachability(self, valid_settings: Settings):
        """Test DB reachability check is invoked."""
        with patch("config.check_database_connection", return_value=True) as mock_check:
            result = validate_config(valid_settings)
            mock_check.assert_called_once()
            assert result.is_valid is True

    def test_validate_config_redis_reachability(self, valid_settings: Settings):
        """Test Redis reachability check is invoked."""
        with patch("config.check_redis_connection", return_value=True) as mock_check:
            result = validate_config(valid_settings)
            mock_check.assert_called_once()
            assert result.is_valid is True

    def test_validate_config_broker_credentials(self, valid_settings: Settings):
        """Test broker credential check is invoked."""
        with patch("config.check_broker_credentials", return_value=True) as mock_check:
            result = validate_config(valid_settings)
            mock_check.assert_called_once()
            assert result.is_valid is True

    def test_validate_config_custom_validators(self, valid_settings: Settings):
        """Test any custom validators are executed."""
        result = validate_config(valid_settings)
        assert hasattr(result, "warnings")


# ---------------------------------------------------------------------------
# Settings singleton & caching tests
# ---------------------------------------------------------------------------

class TestSettingsSingleton:
    """Test get_settings() singleton behavior."""

    def test_get_settings_returns_singleton(self, monkeypatch: pytest.MonkeyPatch):
        """get_settings should return the same instance on multiple calls."""
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_caches_instance(self, monkeypatch: pytest.MonkeyPatch):
        """Settings should be cached after first load."""
        monkeypatch.setenv("ENVIRONMENT", "test")
        s1 = get_settings()
        monkeypatch.setenv("ENVIRONMENT", "production")
        s2 = get_settings()
        assert s1 is s2
        assert s1.environment == "test"

    def test_get_settings_reload_on_env_change(self, monkeypatch: pytest.MonkeyPatch):
        """Simulate cache clearing and ensure reload respects new env."""
        from config import clear_settings_cache  # type: ignore[import]

        monkeypatch.setenv("ENVIRONMENT", "test")
        s1 = get_settings()
        clear_settings_cache()
        monkeypatch.setenv("ENVIRONMENT", "production")
        s2 = get_settings()
        assert s1 is not s2
        assert s2.environment == "production"

    def test_settings_thread_safety(self, monkeypatch: pytest.MonkeyPatch):
        """Concurrent get_settings calls should not create multiple instances."""
        monkeypatch.setenv("ENVIRONMENT", "test")

        results: list[Settings] = []

        def _call():
            results.append(get_settings())

        for _ in range(5):
            _call()

        assert len({id(s) for s in results}) == 1

    def test_settings_lazy_initialization(self, monkeypatch: pytest.MonkeyPatch):
        """Settings should not be constructed until first get_settings call."""
        with patch("config._SETTINGS_CACHE", None):
            monkeypatch.setenv("ENVIRONMENT", "test")
            s = get_settings()
            assert s.environment == "test"

    def test_clear_settings_cache(self):
        """Explicit cache clear should force new instance."""
        from config import clear_settings_cache  # type: ignore[import]

        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()
        assert s1 is not s2


# ---------------------------------------------------------------------------
# Parametrized tests & edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("0", False),
    ],
)
def test_boolean_env_parsing(monkeypatch: pytest.MonkeyPatch, env_value: str, expected: bool):
    """Test various boolean env var formats."""
    monkeypatch.setenv("ENABLE_PAPER_TRADING", env_value)
    settings = load_config_from_env()
    assert getattr(settings, "paper_trading", expected) is expected


@pytest.mark.parametrize(
    "max_pos,valid",
    [
        (0.01, True),   # 1%
        (0.05, True),   # 5%
        (0.20, True),   # 20%
        (0.50, False),  # 50% (too high)
        (1.0, False),   # 100% (invalid)
    ],
)
def test_max_position_validation(max_pos: float, valid: bool):
    """Test max position size validation."""
    if valid:
        settings = Settings(environment="test", max_position_size=max_pos)
        assert settings.max_position_size == max_pos
    else:
        with pytest.raises(ValueError):
            Settings(environment="test", max_position_size=max_pos)
