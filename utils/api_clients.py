"""
============================================================
ALPHA-PRIME v2.0 - API Clients Layer
============================================================

Unified, rate-limit-aware, cache-first API clients for 15+ financial
data providers with automatic failover and structured error handling.

Key features:
- Unified async interface for OHLCV, quotes, and fundamentals.
- Token-bucket rate limiting with jitter and backoff. [web:395][web:397][web:399][web:401]
- Cache-first reads integrated with CacheManager.
- Multi-provider failover orchestration (MultiClient).
- Health checks, metrics hooks, and rich CLI tools.
- Graceful degradation when specific providers or deps are missing.

NOTE:
This module focuses on the integration patterns and abstractions.
Provider implementations are pragmatic but minimal — extend endpoints
as needed in project-specific code.

============================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import aiohttp
import numpy as np
import pandas as pd

from config import get_logger, get_settings
from utils.cache_manager import CacheManager, CacheConfig

logger = get_logger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────────────────
# DATA STRUCTURES & CONFIG
# ──────────────────────────────────────────────────────────

Price = float


@dataclass
class OHLCVBar:
    timestamp: datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: int
    symbol: str


@dataclass
class Quote:
    symbol: str
    price: Price
    change_pct: float
    volume: int
    timestamp: datetime


@dataclass
class Fundamentals:
    pe_ratio: float
    roe: float
    debt_to_equity: float
    market_cap: int
    sector: str


@dataclass
class ApiConfig:
    """
    Configuration for API clients and failover behaviour.

    Attributes:
        providers: Mapping of logical route to provider names in failover order.
        api_keys: Provider → API key mapping.
        cache_ttl_bars: Hours to cache OHLCV bars.
        cache_ttl_quote: Hours to cache quotes.
        cache_ttl_fundamentals: Hours to cache fundamentals.
        max_retries: Max HTTP retry attempts per request.
        request_timeout: Per-request timeout in seconds.
        rate_limit_tokens: Token bucket capacity.
        rate_limit_refill: Tokens per minute replenished.
    """

    providers: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "bars": ["polygon", "tiingo", "alphavantage", "yahoo"],
            "quote": ["polygon", "yahoo", "tiingo"],
            "fundamentals": ["fmp", "intrinio", "finnhub"],
        }
    )
    api_keys: Dict[str, str] = field(default_factory=dict)
    cache_ttl_bars: float = 0.25
    cache_ttl_quote: float = 0.083
    cache_ttl_fundamentals: float = 24.0
    max_retries: int = 3
    request_timeout: float = 10.0
    rate_limit_tokens: int = 300
    rate_limit_refill: float = 1.0  # per minute


# ──────────────────────────────────────────────────────────
# ERROR HIERARCHY
# ──────────────────────────────────────────────────────────


class ClientError(Exception):
    """Base error for all API client issues."""


class RateLimitError(ClientError):
    """Rate limit exceeded."""


class DataNotFoundError(ClientError):
    """Requested data not found (404)."""


class ServerError(ClientError):
    """5xx server error."""


class NetworkError(ClientError):
    """Network / connection error."""


class AllClientsFailedError(ClientError):
    """All providers failed for a given request."""


# ──────────────────────────────────────────────────────────
# TOKEN BUCKET RATE LIMITER (ASYNC)
# ──────────────────────────────────────────────────────────


class TokenBucket:
    """
    Async token bucket rate limiter. [web:395][web:397][web:399][web:401]

    capacity: max number of tokens.
    refill_rate: tokens per minute.
    """

    def __init__(self, capacity: int = 300, refill_rate: float = 1.0) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per minute
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            await self._refill()
            while self.tokens < tokens:
                wait_time = 60.0 / self.refill_rate
                jitter = random.uniform(0.0, 0.25 * wait_time)
                sleep_for = wait_time + jitter
                await asyncio.sleep(sleep_for)
                await self._refill()
            self.tokens -= tokens

    async def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        add = elapsed * (self.refill_rate / 60.0)
        if add > 0:
            self.tokens = min(self.capacity, self.tokens + add)
            self.last_refill = now


# ──────────────────────────────────────────────────────────
# BASE CLIENT INTERFACE & UTILITIES
# ──────────────────────────────────────────────────────────


class BaseClient:
    """
    Unified async interface for market data providers.

    Subclasses should implement:
        - _get_bars_uncached
        - _get_quote_uncached
        - _get_fundamentals_uncached

    All public methods are cache-first, rate-limited, and wrapped in
    structured retry logic.
    """

    name: str = "base"

    def __init__(
        self,
        config: ApiConfig,
        cache: CacheManager,
        session: aiohttp.ClientSession,
        bucket: TokenBucket,
    ) -> None:
        self.config = config
        self.cache = cache
        self.session = session
        self.bucket = bucket
        self.api_key = config.api_keys.get(self.name, "")
        self.healthy: bool = True
        self._circuit_open_until: float = 0.0

    # --- unified public interface ------------------------------------------

    async def get_bars(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        key = f"bars:{self.name}:{symbol}:{freq}:{start}:{end}"
        cached = self.cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return cached

        bars = await self._wrapped_request(self._get_bars_uncached, symbol, freq, start, end)
        if isinstance(bars, pd.DataFrame):
            self.cache.set(key, bars, ttl_hours=self.config.cache_ttl_bars)
        return bars

    async def get_quote(self, symbol: str) -> Quote:
        key = f"quote:{self.name}:{symbol}"
        cached = self.cache.get(key)
        if isinstance(cached, Quote):
            return cached
        quote = await self._wrapped_request(self._get_quote_uncached, symbol)
        self.cache.set(key, quote, ttl_hours=self.config.cache_ttl_quote)
        return quote

    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        key = f"fundamentals:{self.name}:{symbol}"
        cached = self.cache.get(key)
        if isinstance(cached, Fundamentals):
            return cached
        fundamentals = await self._wrapped_request(self._get_fundamentals_uncached, symbol)
        self.cache.set(key, fundamentals, ttl_hours=self.config.cache_ttl_fundamentals)
        return fundamentals

    # --- core implementation hooks -----------------------------------------

    async def _get_bars_uncached(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:  # pragma: no cover - to override
        raise NotImplementedError

    async def _get_quote_uncached(self, symbol: str) -> Quote:  # pragma: no cover
        raise NotImplementedError

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:  # pragma: no cover
        raise NotImplementedError

    # --- health & monitoring -----------------------------------------------

    async def health_check(self) -> bool:
        """
        Simple health check via canary quote.

        Returns:
            True if healthy, False otherwise.
        """
        if time.time() < self._circuit_open_until:
            return False
        try:
            await self.get_quote("AAPL")
            self.healthy = True
        except ClientError:
            self.healthy = False
        return self.healthy

    # --- retry wrapper ------------------------------------------------------

    async def _wrapped_request(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Wrap provider-specific calls with:
            - token bucket acquire
            - retries with backoff
            - structured error mapping
        """
        if time.time() < self._circuit_open_until:
            raise NetworkError(f"Circuit open for {self.name}")

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            await self.bucket.acquire()
            try:
                return await func(*args, **kwargs)
            except DataNotFoundError:
                raise
            except RateLimitError as exc:
                last_exc = exc
                backoff = 1.0 * attempt + random.uniform(0.0, 0.5)
                await asyncio.sleep(backoff)
            except ServerError as exc:
                last_exc = exc
                backoff = 1.0 * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                await asyncio.sleep(backoff)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt == self.config.max_retries:
                    self._circuit_open_until = time.time() + 30.0
                backoff = 1.0 * attempt
                await asyncio.sleep(backoff)
        raise NetworkError(f"{self.name} request failed after retries: {last_exc}")


# ──────────────────────────────────────────────────────────
# PROVIDER CLIENTS (SKELETON IMPLEMENTATIONS)
# ──────────────────────────────────────────────────────────


class AlphaVantageClient(BaseClient):
    name = "alphavantage"
    BASE_URL = "https://www.alphavantage.co/query"

    async def _get_bars_uncached(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        function = "TIME_SERIES_DAILY_ADJUSTED"
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        async with self.session.get(
            self.BASE_URL, params=params, timeout=self.config.request_timeout
        ) as resp:
            if resp.status == 404:
                raise DataNotFoundError("AlphaVantage: data not found")
            if resp.status == 429:
                raise RateLimitError("AlphaVantage: rate limited")
            if 500 <= resp.status < 600:
                raise ServerError(f"AlphaVantage server error {resp.status}")
            data = await resp.json()
        key = "Time Series (Daily)"
        if key not in data:
            raise DataNotFoundError("AlphaVantage: time series missing")
        rows = []
        for dt_str, vals in data[key].items():
            ts = datetime.fromisoformat(dt_str)
            if ts.date() < start or ts.date() > end:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": int(vals["6. volume"]),
                    "symbol": symbol,
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key}
        async with self.session.get(
            self.BASE_URL, params=params, timeout=self.config.request_timeout
        ) as resp:
            if resp.status == 429:
                raise RateLimitError("AlphaVantage: rate limited")
            if 500 <= resp.status < 600:
                raise ServerError("AlphaVantage server error")
            data = await resp.json()
        q = data.get("Global Quote", {})
        if not q:
            raise DataNotFoundError("AlphaVantage: quote missing")
        price = float(q["05. price"])
        change_pct = float(q["10. change percent"].rstrip("%"))
        volume = int(q.get("06. volume", "0"))
        return Quote(
            symbol=symbol,
            price=price,
            change_pct=change_pct,
            volume=volume,
            timestamp=datetime.utcnow(),
        )

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        # AV has limited fundamentals; placeholder mapping.
        return Fundamentals(
            pe_ratio=0.0,
            roe=0.0,
            debt_to_equity=0.0,
            market_cap=0,
            sector="",
        )


class YahooFinanceClient(BaseClient):
    name = "yahoo"
    BASE_URL = "https://query1.finance.yahoo.com"

    async def _get_bars_uncached(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        interval = "1d" if freq == "1d" else "1h"
        url = f"{self.BASE_URL}/v8/finance/chart/{symbol}"
        params = {
            "interval": interval,
            "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
            "period2": int(datetime.combine(end, datetime.max.time()).timestamp()),
        }
        async with self.session.get(url, params=params, timeout=self.config.request_timeout) as resp:
            if resp.status == 404:
                raise DataNotFoundError("Yahoo: data not found")
            if 500 <= resp.status < 600:
                raise ServerError("Yahoo server error")
            data = await resp.json()
        res = data.get("chart", {}).get("result")
        if not res:
            raise DataNotFoundError("Yahoo: chart result missing")
        res0 = res[0]
        timestamps = res0["timestamp"]
        indicators = res0["indicators"]["quote"][0]
        rows = []
        for i, ts in enumerate(timestamps):
            dt = datetime.utcfromtimestamp(ts)
            if dt.date() < start or dt.date() > end:
                continue
            rows.append(
                {
                    "timestamp": dt,
                    "open": indicators["open"][i],
                    "high": indicators["high"][i],
                    "low": indicators["low"][i],
                    "close": indicators["close"][i],
                    "volume": indicators["volume"][i],
                    "symbol": symbol,
                }
            )
        return pd.DataFrame(rows).set_index("timestamp").sort_index()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        url = f"{self.BASE_URL}/v7/finance/quote"
        params = {"symbols": symbol}
        async with self.session.get(url, params=params, timeout=self.config.request_timeout) as resp:
            if resp.status == 404:
                raise DataNotFoundError("Yahoo: quote not found")
            if 500 <= resp.status < 600:
                raise ServerError("Yahoo server error")
            data = await resp.json()
        res = data.get("quoteResponse", {}).get("result", [])
        if not res:
            raise DataNotFoundError("Yahoo: quote result empty")
        q = res[0]
        price = float(q.get("regularMarketPrice", 0.0))
        change_pct = float(q.get("regularMarketChangePercent", 0.0))
        volume = int(q.get("regularMarketVolume", 0))
        ts = datetime.utcfromtimestamp(q.get("regularMarketTime", int(time.time())))
        return Quote(symbol=symbol, price=price, change_pct=change_pct, volume=volume, timestamp=ts)

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        # Yahoo fundamentals via /v10/finance/quoteSummary; simplified.
        return Fundamentals(
            pe_ratio=0.0,
            roe=0.0,
            debt_to_equity=0.0,
            market_cap=0,
            sector="",
        )


class PolygonClient(BaseClient):
    name = "polygon"
    BASE_URL = "https://api.polygon.io"

    async def _get_bars_uncached(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        """
        Uses Polygon aggregates endpoint. [web:398][web:398][web:400][web:406]
        """
        if not self.api_key:
            raise ClientError("Polygon API key missing.")
        multiplier = 1
        timespan = "day" if freq == "1d" else "minute"
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(url, headers=headers, timeout=self.config.request_timeout) as resp:
            if resp.status == 404:
                raise DataNotFoundError("Polygon: data not found")
            if resp.status == 429:
                raise RateLimitError("Polygon: rate limited")
            if 500 <= resp.status < 600:
                raise ServerError("Polygon server error")
            data = await resp.json()
        results = data.get("results", [])
        rows = []
        for r in results:
            ts = datetime.utcfromtimestamp(r["t"] / 1000.0)
            rows.append(
                {
                    "timestamp": ts,
                    "open": r["o"],
                    "high": r["h"],
                    "low": r["l"],
                    "close": r["c"],
                    "volume": r["v"],
                    "symbol": symbol,
                }
            )
        return pd.DataFrame(rows).set_index("timestamp").sort_index()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        url = f"{self.BASE_URL}/v2/last/trade/{symbol}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(url, headers=headers, timeout=self.config.request_timeout) as resp:
            if resp.status == 404:
                raise DataNotFoundError("Polygon: quote not found")
            if resp.status == 429:
                raise RateLimitError("Polygon: rate limited")
            if 500 <= resp.status < 600:
                raise ServerError("Polygon server error")
            data = await resp.json()
        trade = data.get("results", {})
        price = float(trade.get("p", 0.0))
        ts = datetime.utcfromtimestamp(trade.get("t", int(time.time() * 1000)) / 1000.0)
        return Quote(symbol=symbol, price=price, change_pct=0.0, volume=int(trade.get("s", 0)), timestamp=ts)

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(
            pe_ratio=0.0,
            roe=0.0,
            debt_to_equity=0.0,
            market_cap=0,
            sector="",
        )


# For brevity, other clients share similar pattern stubs with minimal logic.
# In production, implement actual endpoints as needed.


class TiingoClient(BaseClient):
    name = "tiingo"

    async def _get_bars_uncached(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class FMPClient(BaseClient):
    name = "fmp"

    async def _get_bars_uncached(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class TwelveDataClient(BaseClient):
    name = "twelvedata"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class EODHistoricalClient(BaseClient):
    name = "eodhistorical"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class IntrinioClient(BaseClient):
    name = "intrinio"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class FinnHubClient(BaseClient):
    name = "finnhub"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class QuandlClient(BaseClient):
    name = "quandl"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        return Fundamentals(0.0, 0.0, 0.0, 0, "")


class NewsAPIClient(BaseClient):
    name = "newsapi"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        raise DataNotFoundError("News API does not provide quotes.")

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        raise DataNotFoundError("News API does not provide fundamentals.")

    async def get_sentiment(self, symbol: str, days: int = 7) -> float:
        """Return sentiment score stub."""
        return 0.0

    async def get_earnings_calendar(self, symbol: str) -> List[Dict[str, Any]]:
        return []


class AlphaVantageNewsClient(BaseClient):
    name = "av_news"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        raise DataNotFoundError("AlphaVantageNews does not provide quotes.")

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        raise DataNotFoundError("AlphaVantageNews does not provide fundamentals.")


class SECEdgarClient(BaseClient):
    name = "sec_edgar"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        raise DataNotFoundError("EDGAR does not provide quotes.")

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        raise DataNotFoundError("EDGAR fundamentals via parsing; stubbed.")


class OpenFIGIClient(BaseClient):
    name = "openfigi"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        raise DataNotFoundError("OpenFIGI is for symbol resolution, not bars.")

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        raise DataNotFoundError("OpenFIGI does not provide quotes.")

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        raise DataNotFoundError("OpenFIGI does not provide fundamentals.")


class ZerodhaClient(BaseClient):
    """
    Zerodha (Kite) brokerage client stub for India-specific operations.

    In production, integrate with Kite Connect SDK.
    """

    name = "zerodha"

    async def _get_bars_uncached(self, symbol: str, freq: str, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame()

    async def _get_quote_uncached(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=0.0, change_pct=0.0, volume=0, timestamp=datetime.utcnow())

    async def _get_fundamentals_uncached(self, symbol: str) -> Fundamentals:
        raise DataNotFoundError("Zerodha does not provide fundamentals.")

    async def get_positions(self) -> List[Dict[str, Any]]:
        return []

    async def place_order(self, symbol: str, side: str, qty: int, price: float) -> str:
        return "ORDER_ID"

    async def get_historical(self, symbol: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────
# MULTI-CLIENT FAILOVER ORCHESTRATOR
# ──────────────────────────────────────────────────────────


CLIENT_CLASSES: Dict[str, Any] = {
    "alphavantage": AlphaVantageClient,
    "yahoo": YahooFinanceClient,
    "polygon": PolygonClient,
    "tiingo": TiingoClient,
    "fmp": FMPClient,
    "twelvedata": TwelveDataClient,
    "eodhistorical": EODHistoricalClient,
    "intrinio": IntrinioClient,
    "finnhub": FinnHubClient,
    "quandl": QuandlClient,
    "newsapi": NewsAPIClient,
    "av_news": AlphaVantageNewsClient,
    "sec_edgar": SECEdgarClient,
    "openfigi": OpenFIGIClient,
    "zerodha": ZerodhaClient,
}


class MultiClient:
    """
    Orchestrator that manages multiple provider clients with failover.

    Example:
        api_config = ApiConfig(api_keys={...})
        cache = CacheManager(CacheConfig())
        async with MultiClient(api_config, cache) as mc:
            bars = await mc.get_bars("AAPL", "1d", start, end)
    """

    def __init__(self, config: ApiConfig, cache: CacheManager) -> None:
        self.config = config
        self.cache = cache
        self.session: Optional[aiohttp.ClientSession] = None
        self.bucket = TokenBucket(
            capacity=config.rate_limit_tokens, refill_rate=config.rate_limit_refill
        )
        self.clients: Dict[str, BaseClient] = {}

    async def __aenter__(self) -> "MultiClient":
        self.session = aiohttp.ClientSession()
        for name, cls in CLIENT_CLASSES.items():
            self.clients[name] = cls(self.config, self.cache, self.session, self.bucket)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            await self.session.close()

    # ---- generic helper ----------------------------------------------------

    async def _route_with_failover(
        self,
        route: str,
        func_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        providers = self.config.providers.get(route, [])
        last_exc: Optional[Exception] = None
        for name in providers:
            client = self.clients.get(name)
            if client is None:
                continue
            try:
                healthy = await client.health_check()
                if not healthy:
                    logger.warning("Client %s unhealthy; skipping.", name)
                    continue
                fn = getattr(client, func_name)
                return await fn(*args, **kwargs)
            except ClientError as exc:
                logger.warning("%s failed for %s: %s", name, func_name, exc)
                last_exc = exc
                continue
        raise AllClientsFailedError(f"All providers failed for route {route}: {last_exc}")

    # ---- high-level API ----------------------------------------------------

    async def get_bars(
        self, symbol: str, freq: str, start: date, end: date
    ) -> pd.DataFrame:
        return await self._route_with_failover("bars", "get_bars", symbol, freq, start, end)

    async def get_quote(self, symbol: str) -> Quote:
        return await self._route_with_failover("quote", "get_quote", symbol)

    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        return await self._route_with_failover("fundamentals", "get_fundamentals", symbol)

    async def warm_symbols(self, symbols: Sequence[str], freq: str, start: date, end: date) -> None:
        tasks = [self.get_bars(sym, freq, start, end) for sym in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def health_report(self) -> Dict[str, bool]:
        report: Dict[str, bool] = {}
        for name, client in self.clients.items():
            try:
                ok = await client.health_check()
            except Exception:  # noqa: BLE001
                ok = False
            report[name] = ok
        return report


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def _default_api_config() -> ApiConfig:
    keys = {
        "polygon": os.getenv("POLYGON_API_KEY", ""),
        "alphavantage": os.getenv("ALPHAVANTAGE_API_KEY", ""),
        "tiingo": os.getenv("TIINGO_API_KEY", ""),
        "fmp": os.getenv("FMP_API_KEY", ""),
        "finnhub": os.getenv("FINNHUB_API_KEY", ""),
    }
    return ApiConfig(api_keys=keys)


async def _cli_test() -> None:
    api_config = _default_api_config()
    cache = CacheManager(CacheConfig())
    async with MultiClient(api_config, cache) as mc:
        report = await mc.health_report()
        print("API CLIENT HEALTH CHECK:")
        for name in ["polygon", "tiingo", "alphavantage", "yahoo"]:
            if name not in report:
                continue
            healthy = report[name]
            icon = "✅" if healthy else "❌"
            msg = "healthy" if healthy else "unhealthy"
            print(f"{icon} {name.capitalize()}: {msg}")


async def _cli_warm(symbols: List[str], freq: str) -> None:
    api_config = _default_api_config()
    cache = CacheManager(CacheConfig())
    start = date.today().replace(year=date.today().year - 1)
    end = date.today()
    async with MultiClient(api_config, cache) as mc:
        print(f"Warming cache for {len(symbols)} symbols...")
        await mc.warm_symbols(symbols, freq, start, end)
        print("Done.")


async def _cli_quote(symbols: List[str]) -> None:
    api_config = _default_api_config()
    cache = CacheManager(CacheConfig())
    async with MultiClient(api_config, cache) as mc:
        parts = []
        for sym in symbols:
            try:
                q = await mc.get_quote(sym)
                parts.append(f"{sym}: ${q.price:.2f} ({q.change_pct:+.1f}%)")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"{sym}: ERROR({exc})")
        print("  |  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - API Clients CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Run health checks for all API clients.")

    warm_p = sub.add_parser("warm", help="Warm cache for symbols.")
    warm_p.add_argument("symbols", nargs="+", help="Symbols to warm.")
    warm_p.add_argument("--freq", type=str, default="1d")

    quote_p = sub.add_parser("quote", help="Fetch quotes for symbols.")
    quote_p.add_argument("--symbols", type=str, required=True, help="Colon-separated symbols (AAPL:MSFT).")

    args = parser.parse_args()
    if args.command == "test":
        asyncio.run(_cli_test())
    elif args.command == "warm":
        asyncio.run(_cli_warm(args.symbols, args.freq))
    elif args.command == "quote":
        syms = args.symbols.split(":")
        asyncio.run(_cli_quote(syms))


if __name__ == "__main__":
    main()
