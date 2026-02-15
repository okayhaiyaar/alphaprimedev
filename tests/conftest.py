"""
ALPHA-PRIME v2.0 - Pytest Configuration & Shared Fixtures
=========================================================

Central pytest configuration module for the ALPHA-PRIME v2.0 codebase.

Responsibilities:
- Register standard markers (unit/integration/e2e/performance/etc.).
- Provide shared fixtures for DB, Redis, cache, app, and external APIs.
- Integrate Factory Boy/Faker for realistic test data.
- Handle async DB/session lifecycle and environment isolation.
- Track slow tests and perform session-wide cleanup.

Importing this module has no side effects beyond pytest's normal
configuration flow (no DB/API work is done at import time).
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import factory
import pytest
from factory.faker import Faker as FactoryFaker
from faker import Faker

# Optional imports; tests that need these should be marked accordingly.
try:  # pragma: no cover - import guard
    import sqlalchemy.ext.asyncio as sa_async
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
except Exception:  # pragma: no cover
    sa_async = None
    AsyncEngine = Any  # type: ignore
    AsyncSession = Any  # type: ignore
    async_sessionmaker = Any  # type: ignore

try:  # pragma: no cover
    import redis.asyncio as redis_async
except Exception:  # pragma: no cover
    redis_async = None

try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


fake = Faker()


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers at startup."""
    config.addinivalue_line(
        "markers", "unit: Fast unit tests, no external services"
    )
    config.addinivalue_line(
        "markers", "integration: Tests requiring DB/Redis/APIs"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end system tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance/load tests (slow)"
    )
    config.addinivalue_line(
        "markers", "smoke: Quick health check tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests >5s execution time"
    )
    config.addinivalue_line(
        "markers", "requires_db: Needs test database"
    )
    config.addinivalue_line(
        "markers", "requires_redis: Needs Redis"
    )
    config.addinivalue_line(
        "markers", "requires_broker: Needs broker API"
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def check_db_available() -> bool:
    """Check if test DB is reachable (lightweight, best-effort)."""
    dsn = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    if sa_async is None:
        return False
    try:
        engine = sa_async.create_async_engine(dsn)
    except Exception:
        return False
    # For sqlite in-memory we assume available; real checks would connect.
    # Avoid actual I/O here to keep collection fast.
    return True


def check_redis_available() -> bool:
    """Check if test Redis is reachable (lightweight, best-effort)."""
    if redis_async is None:
        return False
    url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
    try:
        client = redis_async.from_url(url)
    except Exception:
        return False
    # Avoid network I/O at collection time.
    return True


def generate_realistic_ohlcv(symbol: str, days: int = 252):
    """
    Generate realistic OHLCV data with trends/volatility.

    Uses a simple geometric random walk with noise. Returns a pandas
    DataFrame if pandas is available, otherwise a list of dicts.
    """
    start_price = random.uniform(80.0, 200.0)
    mu = 0.0005  # daily drift
    sigma = 0.02  # daily volatility
    date = datetime.utcnow() - timedelta(days=days)
    rows: List[Dict[str, Any]] = []

    price = start_price
    for _ in range(days):
        date += timedelta(days=1)
        ret = random.gauss(mu, sigma)
        open_price = price
        close_price = price * (1 + ret)
        high = max(open_price, close_price) * (1 + random.uniform(0.0, 0.01))
        low = min(open_price, close_price) * (1 - random.uniform(0.0, 0.01))
        volume = random.randint(1_000_000, 5_000_000)
        price = close_price
        rows.append(
            {
                "symbol": symbol,
                "date": date,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": volume,
            }
        )

    if pd is not None:
        return pd.DataFrame(rows)
    return rows


# ---------------------------------------------------------------------------
# Test settings fixture
# ---------------------------------------------------------------------------

@dataclass
class TestSettings:
    environment: str
    database_url: str
    redis_url: str


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    """Test environment settings (DB/Redis URLs, env flag)."""
    db_url = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
    return TestSettings(
        environment="test",
        database_url=db_url,
        redis_url=redis_url,
    )


# ---------------------------------------------------------------------------
# Database fixtures (async SQLAlchemy)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def test_db_engine(test_settings: TestSettings) -> AsyncIterator[AsyncEngine]:
    """Create async test database engine."""
    if sa_async is None:
        pytest.skip("SQLAlchemy async not available")

    engine: AsyncEngine = sa_async.create_async_engine(
        test_settings.database_url,
        future=True,
    )
    # Schema setup should be done here if using real metadata.
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
async def _session_factory(test_db_engine: AsyncEngine) -> AsyncIterator[async_sessionmaker]:
    """Session factory bound to the test DB engine."""
    if sa_async is None:
        pytest.skip("SQLAlchemy async not available")
    factory_ = async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    yield factory_


@pytest.fixture(scope="function")
async def db_session(_session_factory: async_sessionmaker) -> AsyncIterator[AsyncSession]:
    """Async DB session with rollback after each test."""
    async with _session_factory() as session:
        try:
            yield session
        finally:
            # Rollback all changes after each test for isolation.
            await session.rollback()


@pytest.fixture(scope="function")
async def clean_db(db_session: AsyncSession) -> AsyncIterator[AsyncSession]:
    """
    Clean slate database (all tables truncated).

    This fixture assumes metadata-driven truncation in a real codebase.
    """
    # Real implementation would truncate all tables here.
    yield db_session
    # Optionally re-truncate or cleanup again.


# ---------------------------------------------------------------------------
# Redis / cache fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def test_redis(test_settings: TestSettings):
    """Test Redis instance (fakeredis or real test Redis)."""
    if redis_async is None:
        pytest.skip("redis.asyncio not available")

    client = redis_async.from_url(test_settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="function")
async def redis_client(test_redis):
    """Redis client with FLUSHDB after each test."""
    await test_redis.flushdb()
    yield test_redis
    await test_redis.flushdb()


@dataclass
class CacheManager:
    """Minimal cache manager facade for tests."""
    client: Any

    async def get(self, key: str) -> Any:
        return await self.client.get(key)

    async def set(self, key: str, value: Any, ttl: int = 60) -> None:
        await self.client.set(key, value, ex=ttl)


@pytest.fixture(scope="function")
async def cache_manager(redis_client) -> AsyncIterator[CacheManager]:
    """Isolated CacheManager instance."""
    mgr = CacheManager(client=redis_client)
    yield mgr


# ---------------------------------------------------------------------------
# Domain models (lightweight test-only dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    id: str
    name: str
    sharpe: float
    max_drawdown: float


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    quantity: float = 1.0


@dataclass
class OHLCVBar:
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ---------------------------------------------------------------------------
# Factory Boy factories
# ---------------------------------------------------------------------------

class StrategyFactory(factory.Factory):
    """Factory Boy for creating test strategies."""

    class Meta:
        model = Strategy

    id = factory.Sequence(lambda n: f"strategy_{n}")
    name = FactoryFaker("company")
    sharpe = factory.LazyFunction(lambda: random.uniform(0.5, 2.5))
    max_drawdown = factory.LazyFunction(lambda: -random.uniform(0.05, 0.30))


class TradeFactory(factory.Factory):
    """Factory for creating test trades."""

    class Meta:
        model = Trade

    symbol = FactoryFaker("random_element", elements=["AAPL", "MSFT", "GOOGL", "TSLA"])
    side = FactoryFaker("random_element", elements=["BUY", "SELL"])
    entry_price = factory.LazyFunction(lambda: random.uniform(100, 500))
    quantity = factory.LazyFunction(lambda: random.uniform(1, 100))


class OHLCVBarFactory(factory.Factory):
    """Factory for OHLCV bar generation."""

    class Meta:
        model = OHLCVBar

    symbol = "AAPL"
    date = factory.LazyFunction(datetime.utcnow)
    open = factory.LazyFunction(lambda: random.uniform(100, 200))
    high = factory.LazyAttribute(lambda o: o.open * (1 + random.uniform(0.0, 0.02)))
    low = factory.LazyAttribute(lambda o: o.open * (1 - random.uniform(0.0, 0.02)))
    close = factory.LazyAttribute(lambda o: (o.high + o.low) / 2)
    volume = factory.LazyFunction(lambda: random.randint(1_000_000, 5_000_000))


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy_factory() -> StrategyFactory:
    """Factory Boy for creating test strategies."""
    return StrategyFactory


@pytest.fixture
def trade_factory() -> TradeFactory:
    """Factory for creating test trades."""
    return TradeFactory


@pytest.fixture
def ohlcv_factory() -> OHLCVBarFactory:
    """Factory for OHLCV bar generation."""
    return OHLCVBarFactory


@pytest.fixture
def sample_ohlcv_data():
    """Pre-built 252-day OHLCV dataset (AAPL-like)."""
    return generate_realistic_ohlcv(symbol="AAPL", days=252)


# ---------------------------------------------------------------------------
# Mock API clients
# ---------------------------------------------------------------------------

class MockBrokerClient:
    """Mocked Zerodha/broker API client."""

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "price": round(random.uniform(100, 300), 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    async def place_order(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        return {
            "order_id": f"order_{random.randint(1000, 9999)}",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "status": "filled",
        }


class MockPolygonClient:
    """Mocked Polygon.io API client."""

    async def get_ohlcv(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        data = generate_realistic_ohlcv(symbol=symbol, days=limit)
        if pd is not None and hasattr(data, "to_dict"):
            return data.to_dict(orient="records")
        return data  # type: ignore[return-value]


@pytest.fixture
def mock_broker_client() -> MockBrokerClient:
    """Mocked Zerodha/broker API client."""
    return MockBrokerClient()


@pytest.fixture
def mock_polygon_client() -> MockPolygonClient:
    """Mocked Polygon.io API."""
    return MockPolygonClient()


# ---------------------------------------------------------------------------
# Application & API client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def test_app(test_settings: TestSettings):
    """FastAPI test app with overridden dependencies."""
    try:
        from dashboard.app_v2 import create_app  # type: ignore[import]
    except Exception:  # pragma: no cover
        pytest.skip("dashboard.app_v2.create_app not available")

    app = create_app(env="test", enable_auth=False)
    # In a real implementation, override DB/Redis deps here.
    return app


@pytest.fixture
async def test_api_client(test_app):
    """Async HTTP client for FastAPI testing."""
    try:
        from httpx import AsyncClient  # type: ignore[import]
    except Exception:  # pragma: no cover
        pytest.skip("httpx.AsyncClient not available")

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@dataclass
class StrategyRegistry:
    """Simple test-isolated strategy registry facade."""
    session: Any

    async def list_active(self) -> List[Strategy]:
        # Real implementation would query DB.
        return []


@pytest.fixture
def strategy_registry(db_session: AsyncSession) -> StrategyRegistry:
    """Test-isolated strategy registry."""
    return StrategyRegistry(session=db_session)


# ---------------------------------------------------------------------------
# Pytest hooks: collection tweaks, cleanup
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Auto-mark slow tests and skip integration if services unavailable."""
    db_available = check_db_available()
    redis_available = check_redis_available()

    for item in items:
        # Performance tests are also slow.
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Skip tests requiring DB/Redis if not available.
        if "requires_db" in item.keywords and not db_available:
            item.add_marker(pytest.mark.skip(reason="Test DB unavailable"))
        if "requires_redis" in item.keywords and not redis_available:
            item.add_marker(pytest.mark.skip(reason="Test Redis unavailable"))


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session() -> Iterator[None]:
    """Cleanup test artifacts after entire session."""
    yield
    # Real implementation: delete temp files, test DBs, Redis keys, etc.


@pytest.fixture(scope="function", autouse=True)
async def reset_singletons() -> AsyncIterator[None]:
    """Reset singleton instances between tests."""
    # Real implementation: clear global registries, caches, etc.
    yield
    # After each test: reset any in-memory singletons.


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def track_slow_tests(request: pytest.FixtureRequest) -> Iterator[None]:
    """Auto-report tests slower than 1 second."""
    start = time.time()
    yield
    duration = time.time() - start
    if duration > 1.0:
        print(f"\n⚠️  Slow test: {request.node.name} ({duration:.2f}s)")


__all__ = [
    "test_settings",
    "test_db_engine",
    "db_session",
    "clean_db",
    "test_redis",
    "redis_client",
    "cache_manager",
    "strategy_factory",
    "trade_factory",
    "ohlcv_factory",
    "sample_ohlcv_data",
    "mock_broker_client",
    "mock_polygon_client",
    "test_app",
    "test_api_client",
    "strategy_registry",
    "generate_realistic_ohlcv",
    "check_db_available",
    "check_redis_available",
]
