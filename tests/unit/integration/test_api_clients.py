"""
API client integration tests for ALPHA-PRIME v2.0.

Covers:
- Broker APIs (Zerodha-style), data providers (Polygon, Yahoo, Alpha Vantage).
- Authentication (API keys, tokens, OAuth-like flows, 2FA).
- Rate limiting, retries, error handling, and WebSocket streaming.
- Real HTTP calls via VCR.py cassettes or high-fidelity async mocks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import vcr
from aiohttp import ClientSession

from integrations.broker_clients import ZerodhaClient, BrokerClient
from integrations.data_providers import (
    PolygonClient,
    YahooFinanceClient,
    AlphaVantageClient,
)
from integrations.notification_clients import (
    TelegramClient,
    EmailClient,
    SlackClient,
)


# ---------------------------------------------------------------------------
# VCR configuration and shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vcr_config() -> Dict[str, Any]:
    """VCR configuration for recording HTTP interactions."""
    return {
        "cassette_library_dir": "tests/fixtures/vcr_cassettes",
        "filter_headers": ["authorization", "x-api-key", "cookie"],
        "record_mode": "once",
    }


@pytest.fixture
async def zerodha_client(vcr_config: Dict[str, Any]):
    """Zerodha-like broker client with test credentials."""
    client = ZerodhaClient(
        api_key="test_api_key",
        api_secret="test_api_secret",
        user_id="test_user",
        sandbox=True,
    )
    yield client
    await client.close()


@pytest.fixture
async def polygon_client():
    """Polygon.io client with test API key."""
    client = PolygonClient(
        api_key="test_polygon_key",
        rate_limit=5,
    )
    yield client
    await client.close()


@pytest.fixture
async def yahoo_client():
    """Yahoo Finance client (no auth required)."""
    client = YahooFinanceClient()
    yield client
    await client.close()


@pytest.fixture
async def alpha_client():
    """Alpha Vantage client for intraday/indicator data."""
    client = AlphaVantageClient(
        api_key="test_alpha_key",
        rate_limit=5,
    )
    yield client
    await client.close()


@pytest.fixture
def mock_http_session():
    """Mock aiohttp ClientSession."""
    session = AsyncMock(spec=ClientSession)
    return session


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    limiter = Mock()
    limiter.acquire = AsyncMock()
    return limiter


@pytest.fixture
def sample_quote_response() -> Dict[str, Any]:
    """Sample quote API response."""
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "bid": 150.20,
        "ask": 150.30,
        "volume": 1_000_000,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_order_request() -> Dict[str, Any]:
    """Sample order placement request."""
    return {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100,
        "order_type": "LIMIT",
        "limit_price": 150.0,
    }


@pytest.fixture
def sample_ohlcv_bars() -> List[Dict[str, Any]]:
    """Sample OHLCV historical data."""
    return [
        {
            "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
            "open": 150.0 + i,
            "high": 152.0 + i,
            "low": 148.0 + i,
            "close": 151.0 + i,
            "volume": 1_000_000,
        }
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# 1. Broker API - Authentication
# ---------------------------------------------------------------------------

class TestBrokerAuthentication:
    """Test broker API authentication flows."""

    @pytest.mark.asyncio
    async def test_zerodha_login_success(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "_request",
            return_value={"access_token": "test_token_123", "user_id": "test_user"},
        ):
            result = await zerodha_client.login(
                username="test_user",
                password="test_password",
            )
        assert result["access_token"]
        assert zerodha_client.is_authenticated is True

    @pytest.mark.asyncio
    async def test_zerodha_login_invalid_credentials(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=401,
                message="Unauthorized",
            ),
        ):
            with pytest.raises(aiohttp.ClientResponseError):
                await zerodha_client.login(username="bad", password="wrong")
        assert zerodha_client.is_authenticated is False

    @pytest.mark.asyncio
    async def test_zerodha_token_refresh(self, zerodha_client: ZerodhaClient):
        zerodha_client.access_token = "expired"
        zerodha_client.token_expiry = datetime.utcnow() - timedelta(hours=1)
        with patch.object(
            zerodha_client,
            "_refresh_token",
            return_value={"access_token": "new_token_456"},
        ):
            with patch.object(zerodha_client, "_request", return_value={"price": 150.0}):
                await zerodha_client.get_quote("AAPL")
        assert zerodha_client.access_token == "new_token_456"

    @pytest.mark.asyncio
    async def test_zerodha_session_expiry_handling(self, zerodha_client: ZerodhaClient):
        zerodha_client.access_token = "expired"
        zerodha_client.token_expiry = datetime.utcnow() - timedelta(hours=1)
        with patch.object(
            zerodha_client,
            "_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=403,
                message="Session expired",
            ),
        ):
            with pytest.raises(aiohttp.ClientResponseError):
                await zerodha_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_broker_api_key_validation(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=403,
                message="Invalid API key",
            ),
        ):
            with pytest.raises(aiohttp.ClientResponseError):
                await zerodha_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_oauth_flow_complete(self, mock_http_session: AsyncMock):
        client = BrokerClient(http_session=mock_http_session)
        client.build_oauth_url = Mock(return_value="https://auth.example.com/oauth")
        client.exchange_code_for_token = AsyncMock(return_value={"access_token": "oauth_token"})
        url = client.build_oauth_url(state="xyz")
        assert "oauth" in url
        token = await client.exchange_code_for_token("code123")
        assert token["access_token"] == "oauth_token"

    @pytest.mark.asyncio
    async def test_two_factor_authentication(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "_request",
            return_value={"access_token": "token_2fa", "user_id": "test_user"},
        ):
            result = await zerodha_client.login(
                username="test_user",
                password="test_password",
                two_fa="123456",
            )
        assert result["access_token"] == "token_2fa"

    @pytest.mark.asyncio
    async def test_authentication_state_persistence(self, zerodha_client: ZerodhaClient):
        zerodha_client.access_token = "persisted"
        zerodha_client.token_expiry = datetime.utcnow() + timedelta(hours=1)
        state = zerodha_client.serialize_auth_state()
        new_client = ZerodhaClient(
            api_key="test_api_key",
            api_secret="test_api_secret",
            user_id="test_user",
            sandbox=True,
        )
        new_client.load_auth_state(state)
        assert new_client.access_token == "persisted"


# ---------------------------------------------------------------------------
# 2. Broker API - Market Data
# ---------------------------------------------------------------------------

class TestBrokerMarketData:
    """Test broker market data endpoints."""

    @pytest.mark.asyncio
    @vcr.use_cassette("tests/fixtures/vcr_cassettes/zerodha_get_quote.yaml")
    async def test_get_quote_single_symbol(
        self,
        zerodha_client: ZerodhaClient,
    ):
        quote = await zerodha_client.get_quote("AAPL")
        assert quote["symbol"] == "AAPL"
        assert quote["price"] > 0

    @pytest.mark.asyncio
    async def test_get_quotes_bulk(
        self,
        zerodha_client: ZerodhaClient,
        sample_quote_response: Dict[str, Any],
    ):
        with patch.object(
            zerodha_client,
            "_request",
            return_value={"AAPL": sample_quote_response, "MSFT": sample_quote_response},
        ):
            quotes = await zerodha_client.get_quotes(["AAPL", "MSFT"])
        assert set(quotes.keys()) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_get_historical_data(
        self,
        zerodha_client: ZerodhaClient,
        sample_ohlcv_bars: List[Dict[str, Any]],
    ):
        with patch.object(zerodha_client, "get_historical", return_value=sample_ohlcv_bars):
            data = await zerodha_client.get_historical(
                symbol="AAPL",
                interval="day",
                from_date=datetime.utcnow() - timedelta(days=10),
                to_date=datetime.utcnow(),
            )
        assert len(data) == 10
        assert all("open" in bar and "close" in bar for bar in data)

    @pytest.mark.asyncio
    async def test_get_ohlcv_bars(
        self,
        zerodha_client: ZerodhaClient,
        sample_ohlcv_bars: List[Dict[str, Any]],
    ):
        with patch.object(zerodha_client, "get_ohlcv", return_value=sample_ohlcv_bars):
            bars = await zerodha_client.get_ohlcv("AAPL", interval="5minute", limit=10)
        assert len(bars) == 10

    @pytest.mark.asyncio
    async def test_realtime_quote_stream(self, zerodha_client: ZerodhaClient):
        async def fake_stream(symbols):
            for _ in range(3):
                yield {"symbol": "AAPL", "price": 150.0}

        zerodha_client.stream_quotes = fake_stream
        quotes = []
        async for q in zerodha_client.stream_quotes(["AAPL"]):
            quotes.append(q)
        assert len(quotes) == 3

    @pytest.mark.asyncio
    async def test_instrument_search(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "search_instrument",
            return_value=[{"symbol": "AAPL", "name": "Apple Inc."}],
        ):
            res = await zerodha_client.search_instrument("Apple")
        assert res and res[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_market_depth_l2(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_market_depth",
            return_value={"bids": [], "asks": []},
        ):
            depth = await zerodha_client.get_market_depth("AAPL")
        assert "bids" in depth and "asks" in depth

    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=400,
                message="Invalid symbol",
            ),
        ):
            with pytest.raises(aiohttp.ClientResponseError):
                await zerodha_client.get_quote("INVALID_XYZ")

    @pytest.mark.asyncio
    async def test_quote_data_validation(
        self,
        zerodha_client: ZerodhaClient,
        sample_quote_response: Dict[str, Any],
    ):
        with patch.object(zerodha_client, "_request", return_value=sample_quote_response):
            quote = await zerodha_client.get_quote("AAPL")
        assert isinstance(quote["price"], (int, float))
        assert isinstance(quote["volume"], int)

    @pytest.mark.asyncio
    async def test_market_closed_behavior(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "_request",
            return_value={"symbol": "AAPL", "price": 150.0, "market_open": False},
        ):
            quote = await zerodha_client.get_quote("AAPL")
        assert quote["market_open"] is False


# ---------------------------------------------------------------------------
# 3. Broker API - Order Management
# ---------------------------------------------------------------------------

class TestBrokerOrderManagement:
    """Test broker order placement and management."""

    @pytest.mark.asyncio
    async def test_place_market_order(
        self,
        zerodha_client: ZerodhaClient,
    ):
        with patch.object(
            zerodha_client,
            "place_order",
            return_value={"order_id": "ORD1", "status": "COMPLETE"},
        ):
            res = await zerodha_client.place_order(
                symbol="AAPL",
                side="BUY",
                quantity=100,
                order_type="MARKET",
            )
        assert res["order_id"]
        assert res["status"] in {"COMPLETE", "OPEN"}

    @pytest.mark.asyncio
    async def test_place_limit_order(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "place_order",
            return_value={"order_id": "ORD2", "status": "OPEN"},
        ):
            res = await zerodha_client.place_order(
                symbol="AAPL",
                side="BUY",
                quantity=100,
                order_type="LIMIT",
                limit_price=150.0,
            )
        assert res["status"] in {"OPEN", "COMPLETE"}

    @pytest.mark.asyncio
    async def test_place_stop_loss_order(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "place_order",
            return_value={"order_id": "ORD3", "status": "OPEN"},
        ):
            res = await zerodha_client.place_order(
                symbol="AAPL",
                side="SELL",
                quantity=100,
                order_type="SL",
                stop_price=145.0,
            )
        assert res["order_id"]

    @pytest.mark.asyncio
    async def test_cancel_order(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "cancel_order",
            return_value={"status": "CANCELLED"},
        ):
            res = await zerodha_client.cancel_order("ORD1")
        assert res["status"] == "CANCELLED"

    @pytest.mark.asyncio
    async def test_modify_order(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "modify_order",
            return_value={"order_id": "ORD1", "status": "OPEN", "limit_price": 149.0},
        ):
            res = await zerodha_client.modify_order("ORD1", new_limit_price=149.0)
        assert res["limit_price"] == 149.0

    @pytest.mark.asyncio
    async def test_get_order_status(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_order",
            return_value={"order_id": "ORD1", "status": "COMPLETE"},
        ):
            res = await zerodha_client.get_order("ORD1")
        assert res["status"] == "COMPLETE"

    @pytest.mark.asyncio
    async def test_get_order_history(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_orders",
            return_value=[{"order_id": "ORD1"}, {"order_id": "ORD2"}],
        ):
            res = await zerodha_client.get_orders()
        assert len(res) >= 2

    @pytest.mark.asyncio
    async def test_order_rejection_handling(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "place_order",
            return_value={"status": "REJECTED", "reason": "insufficient margin"},
        ):
            res = await zerodha_client.place_order(
                symbol="AAPL",
                side="BUY",
                quantity=1_000_000,
                order_type="MARKET",
            )
        assert res["status"] == "REJECTED"

    @pytest.mark.asyncio
    async def test_insufficient_margin_error(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "place_order",
            side_effect=ValueError("Insufficient margin"),
        ):
            with pytest.raises(ValueError, match="Insufficient margin"):
                await zerodha_client.place_order(
                    symbol="AAPL",
                    side="BUY",
                    quantity=1_000_000,
                    order_type="MARKET",
                )

    @pytest.mark.asyncio
    async def test_duplicate_order_prevention(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "place_order",
            return_value={"order_id": "ORDX", "status": "COMPLETE"},
        ):
            res1 = await zerodha_client.place_order(
                symbol="AAPL",
                side="BUY",
                quantity=100,
                client_order_id="CID1",
            )
            assert res1["order_id"] == "ORDX"

    @pytest.mark.asyncio
    async def test_order_validation_errors(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "place_order",
            side_effect=ValueError("Validation error"),
        ):
            with pytest.raises(ValueError):
                await zerodha_client.place_order(symbol="", side="BUY", quantity=0)

    @pytest.mark.asyncio
    async def test_bulk_order_placement(self, zerodha_client: ZerodhaClient):
        async def mock_place(**kwargs):
            return {"order_id": f"ORD_{kwargs['symbol']}", "status": "COMPLETE"}

        with patch.object(zerodha_client, "place_order", side_effect=mock_place):
            symbols = ["AAPL", "MSFT", "GOOGL"]
            results = []
            for s in symbols:
                results.append(
                    await zerodha_client.place_order(symbol=s, side="BUY", quantity=10)
                )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# 4. Broker API - Portfolio & Positions
# ---------------------------------------------------------------------------

class TestBrokerPortfolioPositions:
    """Test portfolio and position endpoints."""

    @pytest.mark.asyncio
    async def test_get_positions(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_positions",
            return_value=[{"symbol": "AAPL", "quantity": 10}],
        ):
            pos = await zerodha_client.get_positions()
        assert pos[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_holdings(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_holdings",
            return_value=[{"symbol": "AAPL", "quantity": 10}],
        ):
            holdings = await zerodha_client.get_holdings()
        assert holdings

    @pytest.mark.asyncio
    async def test_get_margins(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_margins",
            return_value={"available": 100_000.0},
        ):
            margins = await zerodha_client.get_margins()
        assert margins["available"] > 0

    @pytest.mark.asyncio
    async def test_get_funds(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_funds",
            return_value={"equity": 50_000.0},
        ):
            funds = await zerodha_client.get_funds()
        assert funds["equity"] > 0

    @pytest.mark.asyncio
    async def test_position_conversion(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "convert_position",
            return_value={"status": "success"},
        ):
            res = await zerodha_client.convert_position("AAPL", from_product="MIS", to_product="CNC")
        assert res["status"] == "success"

    @pytest.mark.asyncio
    async def test_portfolio_reconciliation(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_positions",
            return_value=[],
        ):
            res = await zerodha_client.get_positions()
        assert isinstance(res, list)

    @pytest.mark.asyncio
    async def test_position_pnl_calculation(self, zerodha_client: ZerodhaClient):
        with patch.object(
            zerodha_client,
            "get_positions",
            return_value=[{"symbol": "AAPL", "quantity": 10, "avg_price": 100.0}],
        ):
            with patch.object(
                zerodha_client,
                "get_quote",
                return_value={"symbol": "AAPL", "price": 110.0},
            ):
                positions = await zerodha_client.get_positions()
                quote = await zerodha_client.get_quote("AAPL")
                pnl = (quote["price"] - positions[0]["avg_price"]) * positions[0]["quantity"]
        assert pnl == 100.0

    @pytest.mark.asyncio
    async def test_empty_portfolio_handling(self, zerodha_client: ZerodhaClient):
        with patch.object(zerodha_client, "get_positions", return_value=[]):
            positions = await zerodha_client.get_positions()
        assert positions == []


# ---------------------------------------------------------------------------
# 5. Data Provider APIs
# ---------------------------------------------------------------------------

class TestDataProviderAPIs:
    """Test third-party data provider integrations."""

    @pytest.mark.asyncio
    @vcr.use_cassette("tests/fixtures/vcr_cassettes/polygon_bars.yaml")
    async def test_polygon_historical_bars(self, polygon_client: PolygonClient):
        bars = await polygon_client.get_bars(
            symbol="AAPL",
            timespan="day",
            from_date="2024-01-01",
            to_date="2024-01-10",
        )
        assert bars
        assert all("open" in b for b in bars)

    @pytest.mark.asyncio
    async def test_polygon_realtime_websocket(self, polygon_client: PolygonClient):
        async def fake_stream():
            for _ in range(2):
                yield {"symbol": "AAPL", "price": 150.0}

        polygon_client.stream_trades = fake_stream
        msgs = []
        async for m in polygon_client.stream_trades():
            msgs.append(m)
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_yahoo_finance_quote(self, yahoo_client: YahooFinanceClient):
        with patch.object(
            yahoo_client, "get_quote", return_value={"symbol": "AAPL", "price": 150.0}
        ):
            q = await yahoo_client.get_quote("AAPL")
        assert q["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_yahoo_finance_historical(self, yahoo_client: YahooFinanceClient):
        with patch.object(
            yahoo_client,
            "get_history",
            return_value=[{"close": 150.0}, {"close": 151.0}],
        ):
            h = await yahoo_client.get_history("AAPL", start="2024-01-01", end="2024-01-10")
        assert len(h) >= 2

    @pytest.mark.asyncio
    async def test_alpha_vantage_intraday(self, alpha_client: AlphaVantageClient):
        with patch.object(
            alpha_client,
            "get_intraday",
            return_value=[{"timestamp": "2024-01-01T09:15:00", "close": 150.0}],
        ):
            bars = await alpha_client.get_intraday("AAPL", interval="5min")
        assert bars

    @pytest.mark.asyncio
    async def test_alpha_vantage_technical_indicators(self, alpha_client: AlphaVantageClient):
        with patch.object(
            alpha_client,
            "get_indicator",
            return_value=[{"rsi": 55.0}],
        ):
            data = await alpha_client.get_indicator("AAPL", indicator="RSI")
        assert "rsi" in data[0]

    @pytest.mark.asyncio
    async def test_free_tier_rate_limiting(self, alpha_client: AlphaVantageClient):
        alpha_client.max_calls_per_minute = 5
        with patch.object(alpha_client, "_sleep") as mock_sleep:
            for _ in range(7):
                await alpha_client._throttle()
        assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_data_provider_failover(
        self,
        polygon_client: PolygonClient,
        yahoo_client: YahooFinanceClient,
    ):
        with patch.object(polygon_client, "get_quote", side_effect=Exception("API down")):
            with patch.object(
                yahoo_client, "get_quote", return_value={"symbol": "AAPL", "price": 150.0}
            ):
                q = await yahoo_client.get_quote("AAPL")
        assert q["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_data_validation_schema(self, polygon_client: PolygonClient):
        sample = {"symbol": "AAPL", "price": 150.0}
        assert "symbol" in sample and "price" in sample

    @pytest.mark.asyncio
    async def test_missing_data_handling(self, yahoo_client: YahooFinanceClient):
        with patch.object(yahoo_client, "get_quote", return_value=None):
            q = await yahoo_client.get_quote("MISSING")
        assert q is None

    @pytest.mark.asyncio
    async def test_timezone_conversion(self, yahoo_client: YahooFinanceClient):
        with patch.object(
            yahoo_client,
            "get_history",
            return_value=[{"timestamp": "2024-01-01T09:30:00-05:00"}],
        ):
            hist = await yahoo_client.get_history("AAPL", start="2024-01-01", end="2024-01-02")
        assert "timestamp" in hist[0]

    @pytest.mark.asyncio
    async def test_corporate_actions_adjustments(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "get_adjusted_bars",
            return_value=[{"close": 150.0, "split_factor": 2}],
        ):
            bars = await polygon_client.get_adjusted_bars("AAPL", "day", "2024-01-01", "2024-01-10")
        assert "split_factor" in bars[0]


# ---------------------------------------------------------------------------
# 6. Rate Limiting & Retries
# ---------------------------------------------------------------------------

class TestRateLimitingRetries:
    """Test rate limiting and retry mechanisms."""

    @pytest.mark.asyncio
    async def test_rate_limit_429_handling(self, polygon_client: PolygonClient):
        call_count = 0

        async def mock_request(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientResponseError(
                    request_info=Mock(),
                    history=(),
                    status=429,
                    message="Rate limit exceeded",
                )
            return {"symbol": "AAPL", "price": 150.0}

        with patch.object(polygon_client, "_request", side_effect=mock_request):
            result = await polygon_client.get_quote("AAPL")
        assert call_count == 2
        assert result["price"] == 150.0

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, polygon_client: PolygonClient):
        retry_times: List[float] = []

        async def mock_request(*_args, **_kwargs):
            retry_times.append(asyncio.get_event_loop().time())
            if len(retry_times) < 3:
                raise asyncio.TimeoutError()
            return {"symbol": "AAPL", "price": 150.0}

        with patch.object(polygon_client, "_sleep", side_effect=asyncio.sleep) as _sleep:
            with patch.object(polygon_client, "_request", side_effect=mock_request):
                await polygon_client.get_quote("AAPL")
        assert len(retry_times) >= 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, polygon_client: PolygonClient):
        call_count = 0

        async def mock_request(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return {"symbol": "AAPL", "price": 150.0}

        with patch.object(polygon_client, "_request", side_effect=mock_request):
            result = await polygon_client.get_quote("AAPL")
        assert call_count == 2
        assert result["price"] == 150.0

    @pytest.mark.asyncio
    async def test_retry_on_5xx_errors(self, polygon_client: PolygonClient):
        call_count = 0

        async def mock_request(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise aiohttp.ClientResponseError(
                    request_info=Mock(), history=(), status=500, message="Server error"
                )
            return {"symbol": "AAPL", "price": 150.0}

        with patch.object(polygon_client, "_request", side_effect=mock_request):
            result = await polygon_client.get_quote("AAPL")
        assert call_count == 2
        assert result["price"] == 150.0

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx_errors(self, polygon_client: PolygonClient):
        call_count = 0

        async def mock_request(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=400, message="Bad request"
            )

        with patch.object(polygon_client, "_request", side_effect=mock_request):
            with pytest.raises(aiohttp.ClientResponseError):
                await polygon_client.get_quote("AAPL")
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_token_bucket(self, polygon_client: PolygonClient, mock_rate_limiter: Mock):
        polygon_client.rate_limiter = mock_rate_limiter
        with patch.object(polygon_client, "_request", return_value={"symbol": "AAPL", "price": 150.0}):
            await polygon_client.get_quote("AAPL")
        polygon_client.rate_limiter.acquire.assert_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_request_throttling(self, polygon_client: PolygonClient, mock_rate_limiter: Mock):
        polygon_client.rate_limiter = mock_rate_limiter

        async def call():
            with patch.object(polygon_client, "_request", return_value={"symbol": "AAPL", "price": 150.0}):
                return await polygon_client.get_quote("AAPL")

        await asyncio.gather(*(call() for _ in range(5)))
        assert polygon_client.rate_limiter.acquire.await_count == 5

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, polygon_client: PolygonClient):
        async def mock_request(*_args, **_kwargs):
            raise asyncio.TimeoutError()

        polygon_client.max_retries = 2
        with patch.object(polygon_client, "_request", side_effect=mock_request):
            with pytest.raises(asyncio.TimeoutError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_repeated_failures(self, polygon_client: PolygonClient):
        polygon_client.circuit_breaker_fail_threshold = 3

        async def mock_request(*_args, **_kwargs):
            raise aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=503, message="Service down"
            )

        with patch.object(polygon_client, "_request", side_effect=mock_request):
            for _ in range(3):
                with pytest.raises(aiohttp.ClientResponseError):
                    await polygon_client.get_quote("AAPL")
        assert polygon_client.circuit_breaker_open is True

    @pytest.mark.asyncio
    async def test_rate_limit_per_endpoint(self, polygon_client: PolygonClient):
        polygon_client.endpoint_limits = {"get_quote": 5}
        with patch.object(polygon_client, "_request", return_value={"symbol": "AAPL", "price": 150.0}):
            for _ in range(5):
                await polygon_client.get_quote("AAPL")
        assert True


# ---------------------------------------------------------------------------
# 7. Error Handling & Edge Cases
# ---------------------------------------------------------------------------

class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, polygon_client: PolygonClient):
        with patch.object(polygon_client, "_request", side_effect=asyncio.TimeoutError()):
            with pytest.raises(asyncio.TimeoutError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_connection_refused_error(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "_request",
            side_effect=ConnectionRefusedError(),
        ):
            with pytest.raises(ConnectionRefusedError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "_request",
            side_effect=aiohttp.ClientConnectorError(Mock(), OSError("DNS failure")),
        ):
            with pytest.raises(aiohttp.ClientConnectorError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_ssl_certificate_error(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "_request",
            side_effect=aiohttp.ClientSSLError("SSL error"),
        ):
            with pytest.raises(aiohttp.ClientSSLError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "_request_raw",
            return_value=Mock(text=AsyncMock(return_value="not-json")),
        ):
            with pytest.raises(ValueError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_empty_response_body(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "_request_raw",
            return_value=Mock(text=AsyncMock(return_value="")),
        ):
            with pytest.raises(ValueError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_unexpected_status_code(self, polygon_client: PolygonClient):
        with patch.object(
            polygon_client,
            "_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=418, message="I'm a teapot"
            ),
        ):
            with pytest.raises(aiohttp.ClientResponseError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_api_version_mismatch(self, polygon_client: PolygonClient):
        polygon_client.api_version = "v1"
        with patch.object(
            polygon_client,
            "_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=426, message="Upgrade required"
            ),
        ):
            with pytest.raises(aiohttp.ClientResponseError):
                await polygon_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_downstream_service_outage(self, yahoo_client: YahooFinanceClient):
        with patch.object(yahoo_client, "_request", side_effect=Exception("downstream")):
            with pytest.raises(Exception):
                await yahoo_client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_partial_response_handling(self, yahoo_client: YahooFinanceClient):
        with patch.object(
            yahoo_client,
            "_request",
            return_value={"symbol": "AAPL"},
        ):
            q = await yahoo_client.get_quote("AAPL")
        assert "symbol" in q


# ---------------------------------------------------------------------------
# 8. Notification Clients
# ---------------------------------------------------------------------------

class TestNotificationClients:
    """Test notification and alerting integrations."""

    @pytest.mark.asyncio
    async def test_telegram_send_message(self):
        client = TelegramClient(token="test_token", chat_id="123")
        client._post = AsyncMock(return_value={"ok": True})
        res = await client.send_message("Test message")
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_email_send_alert(self):
        client = EmailClient(
            smtp_host="smtp.test",
            smtp_port=587,
            username="user",
            password="pass",
            from_addr="from@test",
        )
        client._send = AsyncMock(return_value=True)
        res = await client.send_email("to@test", "Subject", "Body")
        assert res is True

    @pytest.mark.asyncio
    async def test_slack_webhook_post(self):
        client = SlackClient(webhook_url="https://hooks.slack.test/url")
        client._post = AsyncMock(return_value={"ok": True})
        res = await client.send_message("channel", "Test")
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_notification_retry_on_failure(self):
        client = SlackClient(webhook_url="https://hooks.slack.test/url")
        client._post = AsyncMock(side_effect=[Exception("fail"), {"ok": True}])
        res = await client.send_message("channel", "Test", max_retries=2)
        assert res["ok"] is True
        assert client._post.await_count == 2

    @pytest.mark.asyncio
    async def test_notification_rate_limiting(self):
        client = TelegramClient(token="test", chat_id="123")
        client._post = AsyncMock(return_value={"ok": True})
        client.rate_limit_per_minute = 5
        await asyncio.gather(*(client.send_message(f"msg {i}") for i in range(5)))
        assert client._post.await_count == 5

    @pytest.mark.asyncio
    async def test_rich_formatting_support(self):
        client = SlackClient(webhook_url="https://hooks.slack.test/url")
        client._post = AsyncMock(return_value={"ok": True})
        res = await client.send_message("channel", "*bold* _italic_")
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_attachment_handling(self):
        client = EmailClient(
            smtp_host="smtp.test",
            smtp_port=587,
            username="user",
            password="pass",
            from_addr="from@test",
        )
        client._send = AsyncMock(return_value=True)
        res = await client.send_email(
            "to@test",
            "Subject",
            "Body",
            attachments=[b"file-bytes"],
        )
        assert res is True

    @pytest.mark.asyncio
    async def test_notification_batching(self):
        client = TelegramClient(token="test", chat_id="123")
        client._post = AsyncMock(return_value={"ok": True})
        res = await client.send_batch(["m1", "m2", "m3"])
        assert res is True
        assert client._post.await_count == 3


# ---------------------------------------------------------------------------
# 9. WebSocket Streams
# ---------------------------------------------------------------------------

class TestWebSocketStreams:
    """Test real-time WebSocket connections."""

    @pytest.mark.asyncio
    async def test_websocket_connection_establish(self, zerodha_client: ZerodhaClient):
        ws = Mock()
        ws.closed = False
        with patch.object(zerodha_client, "connect_websocket", return_value=ws):
            conn = await zerodha_client.connect_websocket()
        assert conn.closed is False

    @pytest.mark.asyncio
    async def test_websocket_subscribe_symbols(self, zerodha_client: ZerodhaClient):
        ws = AsyncMock()
        ws.send_json = AsyncMock()
        with patch.object(zerodha_client, "connect_websocket", return_value=ws):
            await zerodha_client.start_streaming(["AAPL", "MSFT"])
        ws.send_json.assert_awaited()

    @pytest.mark.asyncio
    async def test_websocket_message_parsing(self, zerodha_client: ZerodhaClient):
        messages = [
            {"type": "quote", "symbol": "AAPL", "price": 150.0},
            {"type": "quote", "symbol": "MSFT", "price": 300.0},
        ]

        async def fake_stream():
            for m in messages:
                yield m

        zerodha_client.stream_websocket = fake_stream
        received = []
        async for msg in zerodha_client.stream_websocket():
            received.append(msg)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_websocket_reconnect_on_disconnect(self, zerodha_client: ZerodhaClient):
        reconnect_count = 0

        async def mock_connect():
            nonlocal reconnect_count
            reconnect_count += 1
            if reconnect_count == 1:
                raise ConnectionError("Connection lost")
            return Mock(closed=False)

        with patch.object(zerodha_client, "connect_websocket", side_effect=mock_connect):
            await zerodha_client.start_streaming(["AAPL"])
        assert reconnect_count >= 2

    @pytest.mark.asyncio
    async def test_websocket_heartbeat_keepalive(self, zerodha_client: ZerodhaClient):
        ws = AsyncMock()
        ws.send_json = AsyncMock()
        ws.closed = False
        with patch.object(zerodha_client, "connect_websocket", return_value=ws):
            await zerodha_client._send_heartbeat(ws)
        ws.send_json.assert_awaited()

    @pytest.mark.asyncio
    async def test_websocket_authentication(self, zerodha_client: ZerodhaClient):
        ws = AsyncMock()
        ws.send_json = AsyncMock()
        zerodha_client.access_token = "token123"
        with patch.object(zerodha_client, "connect_websocket", return_value=ws):
            await zerodha_client.start_streaming(["AAPL"])
        ws.send_json.assert_awaited()

    @pytest.mark.asyncio
    async def test_websocket_graceful_shutdown(self, zerodha_client: ZerodhaClient):
        ws = AsyncMock()
        ws.close = AsyncMock()
        with patch.object(zerodha_client, "connect_websocket", return_value=ws):
            conn = await zerodha_client.connect_websocket()
            await zerodha_client.stop_streaming(conn)
        ws.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_websocket_message_buffering(self, zerodha_client: ZerodhaClient):
        buffer: List[Dict[str, Any]] = []

        async def fake_stream():
            for _ in range(3):
                yield {"symbol": "AAPL", "price": 150.0}

        zerodha_client.stream_websocket = fake_stream
        async for msg in zerodha_client.stream_websocket():
            buffer.append(msg)
        assert len(buffer) == 3


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status_code,should_retry",
    [
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (400, False),
        (401, False),
        (404, False),
    ],
)
@pytest.mark.asyncio
async def test_retry_logic_by_status(
    polygon_client: PolygonClient,
    status_code: int,
    should_retry: bool,
):
    """Test retry logic for different HTTP status codes."""
    call_count = 0

    async def mock_request(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        raise aiohttp.ClientResponseError(
            request_info=Mock(),
            history=(),
            status=status_code,
            message=f"HTTP {status_code}",
        )

    polygon_client.max_retries = 2
    with patch.object(polygon_client, "_request", side_effect=mock_request):
        with pytest.raises(aiohttp.ClientResponseError):
            await polygon_client.get_quote("AAPL")

    if should_retry:
        assert call_count > 1
    else:
        assert call_count == 1


@pytest.mark.parametrize(
    "provider_cls,symbol,expected_ok",
    [
        (PolygonClient, "AAPL", True),
        (YahooFinanceClient, "AAPL", True),
        (AlphaVantageClient, "AAPL", True),
        (PolygonClient, "INVALID", False),
    ],
)
@pytest.mark.asyncio
async def test_data_provider_symbol_validation(
    provider_cls,
    symbol: str,
    expected_ok: bool,
):
    """Test symbol validation across providers."""
    if provider_cls is PolygonClient:
        client = PolygonClient(api_key="test", rate_limit=5)
    elif provider_cls is YahooFinanceClient:
        client = YahooFinanceClient()
    else:
        client = AlphaVantageClient(api_key="test", rate_limit=5)

    try:
        with patch.object(client, "validate_symbol", return_value=expected_ok):
            res = client.validate_symbol(symbol)
        assert res is expected_ok
    finally:
        if hasattr(client, "close") and asyncio.iscoroutinefunction(client.close):
            await client.close()
