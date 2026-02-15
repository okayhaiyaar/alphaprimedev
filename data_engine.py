"""
============================================================
ALPHA-PRIME v2.0 - Data Engine (The Mathematician)
============================================================
Module 2: Fetches market data and computes technical indicators.

NO AI/ML in this module - pure mathematical analysis only.

Data Flow:
1. Fetch OHLCV data from yfinance
2. Validate data quality (missing bars, outliers, splits)
3. Compute 20+ technical indicators using pandas_ta
4. Return structured, validated output

Usage:
    from data_engine import (
        get_market_data,
        calculate_hard_technicals,
        get_multi_timeframe_data,
        validate_data_quality,
    )

    df = get_market_data("AAPL", period="3mo", interval="1d")
    technicals = calculate_hard_technicals(df, ticker="AAPL", timeframe="1d")

    print(technicals["momentum"]["rsi"])
    print(technicals["trend"]["trend"])

Output Schema (simplified view):
    HardTechnicals {
        ticker: str
        timeframe: str
        price_action: PriceAction
        trend: TrendIndicators
        momentum: MomentumIndicators
        volatility: VolatilityIndicators
        volume: VolumeIndicators
        macd: MACDData
        support_resistance: SupportResistance
        data_quality: DataQualityReport
        computed_at_utc: str
    }
============================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_logger, get_settings

try:
    import pandas_ta as ta  # type: ignore

    HAS_PANDAS_TA = True
except ImportError:  # pragma: no cover - optional dependency path
    ta = None
    HAS_PANDAS_TA = False

# Initialize module-level objects
settings = get_settings()
logger = get_logger(__name__)
cache = Cache(f"{settings.cache_dir}/market_data")

if not HAS_PANDAS_TA:
    logger.warning("pandas_ta not installed; using built-in indicator fallback.")

# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class DataQualityReport:
    """Data quality assessment for a time series."""

    is_valid: bool
    quality_score: float  # 0-100
    issues: List[str]
    missing_bars: int
    outliers_detected: int
    split_adjusted: bool
    last_update_utc: str


@dataclass
class PriceAction:
    """Current price action summary."""

    last_price: float
    change_pct_1d: float
    change_pct_1w: float
    change_pct_1m: float
    volume: int
    avg_volume_20d: float
    high_52w: float
    low_52w: float
    distance_from_52w_high_pct: float


@dataclass
class TrendIndicators:
    """Trend-following indicators."""

    ema_9: float
    ema_20: float
    ema_50: float
    ema_200: Optional[float]
    sma_50: float
    sma_200: Optional[float]
    trend: str  # UP | DOWN | SIDEWAYS | UNKNOWN
    trend_strength: float  # 0-100


@dataclass
class MomentumIndicators:
    """Momentum oscillators."""

    rsi: float
    rsi_signal: str  # OVERSOLD | NEUTRAL | OVERBOUGHT | UNKNOWN
    stoch_k: float
    stoch_d: float
    stoch_signal: str  # OVERSOLD | NEUTRAL | OVERBOUGHT | UNKNOWN


@dataclass
class VolatilityIndicators:
    """Volatility measures."""

    atr: float
    atr_pct: float  # ATR as % of price
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width_pct: float
    bb_position: float  # 0-1 (0=lower band,1=upper)


@dataclass
class VolumeIndicators:
    """Volume-based indicators."""

    obv: float
    obv_trend: str  # INCREASING | DECREASING | FLAT | UNKNOWN
    volume_ratio: float  # current_volume / avg_volume_20d
    vwap: float


@dataclass
class MACDData:
    """MACD indicator components."""

    macd_line: float
    signal_line: float
    histogram: float
    signal: str  # BULLISH | BEARISH | NEUTRAL


@dataclass
class SupportResistance:
    """Key support and resistance levels."""

    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    pivot: float


@dataclass
class HardTechnicals:
    """Complete technical analysis package (NO AI)."""

    ticker: str
    timeframe: str
    price_action: PriceAction
    trend: TrendIndicators
    momentum: MomentumIndicators
    volatility: VolatilityIndicators
    volume: VolumeIndicators
    macd: MACDData
    support_resistance: SupportResistance
    data_quality: DataQualityReport
    computed_at_utc: str


# ──────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────


def _ensure_ohlcv(df: pd.DataFrame) -> None:
    """Ensure OHLCV columns exist, raising ValueError if not."""
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _safe_pct_change(current: float, past: float) -> float:
    """Safe percentage change calculation."""
    if past is None or past == 0 or np.isnan(past):
        return 0.0
    return (current - past) / past * 100.0


# ──────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_yfinance(
    ticker: str, period: str, interval: str
) -> pd.DataFrame:
    """
    Internal helper to fetch data from yfinance.

    Args:
        ticker: Stock symbol.
        period: History period.
        interval: Data interval.

    Returns:
        DataFrame with OHLCV.
    """
    logger.info("Fetching market data via yfinance: %s %s %s", ticker, period, interval)
    t = yf.Ticker(ticker)
    timeout_s = int(getattr(settings, "request_timeout_seconds", 30))
    df = t.history(period=period, interval=interval, auto_adjust=True, timeout=timeout_s)
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}.")
    return df


def _fallback_fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fallback data fetcher returning empty data when primary providers fail.

    This keeps UI/scheduler paths resilient under restricted-network conditions.
    """
    logger.warning(
        "Returning empty fallback market data for %s (%s, %s).",
        ticker,
        period,
        interval,
    )
    return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


def _cache_key(ticker: str, period: str, interval: str) -> str:
    """Build a cache key for OHLCV data."""
    return f"market::{ticker.upper()}::{period}::{interval}"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def get_market_data(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV market data from yfinance with disk caching.

    Args:
        ticker: Stock symbol.
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m,
                  1h, 1d, 5d, 1wk, 1mo, 3mo).

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Raises:
        ValueError: If no usable data is returned.
    """
    ticker_u = ticker.upper()
    key = _cache_key(ticker_u, period, interval)

    ttl = 300 if any(i in interval for i in ("m", "h")) else 3600
    cached = cache.get(key)
    if cached is not None:
        try:
            df = pd.DataFrame(cached)
            if not df.empty:
                logger.debug("Market data cache hit for %s (%s, %s)", ticker_u, period, interval)
                return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to deserialize cached data for %s: %s", ticker_u, exc)

    try:
        df = _fetch_yfinance(ticker_u, period, interval)
    except Exception as exc:  # noqa: BLE001
        logger.error("yfinance fetch failed for %s: %s", ticker_u, exc)
        df = _fallback_fetch(ticker_u, period, interval)

    if df.empty:
        raise ValueError(f"No data available for {ticker_u} with period={period}, interval={interval}.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    _ensure_ohlcv(df)

    df = df.reset_index().rename(columns={"index": "Date"})
    try:
        cache.set(key, df.to_dict(orient="list"), expire=ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cache market data for %s: %s", ticker_u, exc)

    logger.info("Fetched %d bars for %s (%s, %s)", len(df), ticker_u, period, interval)
    return df


def get_multi_timeframe_data(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch data across multiple timeframes for confluence analysis.

    Timeframes:
        1h : last 5 days, 1-hour bars
        4h : last 60 days, resampled from 1-hour bars
        1d : last 6 months, daily bars

    Args:
        ticker: Stock symbol.

    Returns:
        Dict[str, DataFrame]: Keys '1h', '4h', '1d'.
    """
    symbol = ticker.upper()
    logger.info("Fetching multi-timeframe data for %s", symbol)

    timeframes = {
        "1h": {"period": "5d", "interval": "1h"},
        "4h": {"period": "60d", "interval": "1h"},  # resample
        "1d": {"period": "6mo", "interval": "1d"},
    }

    result: Dict[str, pd.DataFrame] = {}

    for name, params in timeframes.items():
        try:
            df = get_market_data(symbol, params["period"], params["interval"])
            if name == "4h":
                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"])
                df = (
                    df.set_index("Date")
                    .resample("4H")
                    .agg(
                        {
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        }
                    )
                    .dropna()
                    .reset_index()
                )
            result[name] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s data for %s: %s", name, symbol, exc)
            result[name] = pd.DataFrame()

    return result


# ──────────────────────────────────────────────────────────
# DATA QUALITY CHECKS
# ──────────────────────────────────────────────────────────


def validate_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """
    Comprehensive data quality assessment.

    Checks:
    - Missing bars (NaNs in OHLC).
    - Outliers (3 std dev rule on Close).
    - Zero/negative prices.
    - Volume anomalies (many zero-volume bars).
    - Large price gaps (potential splits/events).

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataQualityReport with quality score and issues.
    """
    if df.empty:
        return DataQualityReport(
            is_valid=False,
            quality_score=0.0,
            issues=["Empty DataFrame"],
            missing_bars=0,
            outliers_detected=0,
            split_adjusted=False,
            last_update_utc=datetime.now(timezone.utc).isoformat(),
        )

    issues: List[str] = []
    quality_score = 100.0

    # Missing price values
    missing_bars = int(df[["Open", "High", "Low", "Close"]].isnull().sum().sum())
    if missing_bars > 0:
        issues.append(f"{missing_bars} missing price values")
        quality_score -= min(20.0, missing_bars * 2.0)

    outliers_detected = 0
    split_adjusted = False

    if "Close" in df.columns:
        close = df["Close"]
        close_mean = float(close.mean())
        close_std = float(close.std(ddof=1)) if len(close) > 1 else 0.0

        if close_std > 0:
            mask = (close > close_mean + 3 * close_std) | (close < close_mean - 3 * close_std)
            outliers_detected = int(mask.sum())
            if outliers_detected > 0:
                issues.append(f"{outliers_detected} price outliers detected")
                quality_score -= min(15.0, outliers_detected * 1.5)

        # Non-positive prices
        invalid_prices = int((close <= 0).sum())
        if invalid_prices > 0:
            issues.append(f"{invalid_prices} invalid prices (<= 0)")
            quality_score -= 25.0

        # Large gaps (potential splits)
        if len(close) > 2:
            pct_changes = close.pct_change().abs()
            large_gaps = int((pct_changes > 0.5).sum())
            if large_gaps > 0:
                issues.append(f"{large_gaps} large price gaps (>50%)")
                split_adjusted = True
                quality_score -= 10.0

    # Volume anomalies
    if "Volume" in df.columns:
        volume = df["Volume"]
        zero_volume = int((volume == 0).sum())
        if zero_volume > len(df) * 0.1:
            issues.append(f"{zero_volume} bars with zero volume")
            quality_score -= 10.0

    quality_score = max(0.0, quality_score)
    is_valid = quality_score >= 70.0

    return DataQualityReport(
        is_valid=is_valid,
        quality_score=quality_score,
        issues=issues,
        missing_bars=missing_bars,
        outliers_detected=outliers_detected,
        split_adjusted=split_adjusted,
        last_update_utc=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (PURE MATH, NO AI)
# ──────────────────────────────────────────────────────────


def _compute_trend(df: pd.DataFrame) -> TrendIndicators:
    """Compute EMA/SMA-based trend indicators."""
    df = df.copy()

    if HAS_PANDAS_TA:
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        if len(df) >= 200:
            df.ta.ema(length=200, append=True)

        df.ta.sma(length=50, append=True)
        if len(df) >= 200:
            df.ta.sma(length=200, append=True)
    else:
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        if len(df) >= 200:
            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        if len(df) >= 200:
            df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()

    last = df.iloc[-1]
    close = float(last["Close"])

    ema_9 = float(last.get("EMA_9", np.nan))
    ema_20 = float(last.get("EMA_20", np.nan))
    ema_50 = float(last.get("EMA_50", np.nan))
    ema_200 = float(last.get("EMA_200", np.nan)) if "EMA_200" in df.columns else None

    sma_50 = float(last.get("SMA_50", np.nan))
    sma_200 = float(last.get("SMA_200", np.nan)) if "SMA_200" in df.columns else None

    trend = "UNKNOWN"
    trend_strength = 0.0

    if not np.isnan(ema_20) and not np.isnan(ema_50):
        if close > ema_20 > ema_50:
            trend = "UP"
            trend_strength = min(100.0, abs(close - ema_50) / max(1e-9, ema_50) * 1000.0)
        elif close < ema_20 < ema_50:
            trend = "DOWN"
            trend_strength = min(100.0, abs(ema_50 - close) / max(1e-9, ema_50) * 1000.0)
        else:
            trend = "SIDEWAYS"
            trend_strength = 50.0

    return TrendIndicators(
        ema_9=ema_9,
        ema_20=ema_20,
        ema_50=ema_50,
        ema_200=ema_200,
        sma_50=sma_50,
        sma_200=sma_200,
        trend=trend,
        trend_strength=trend_strength,
    )


def _compute_momentum(df: pd.DataFrame) -> MomentumIndicators:
    """Compute RSIs and Stochastics."""
    df = df.copy()

    if HAS_PANDAS_TA:
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(append=True)
    else:
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        lowest_low = df["Low"].rolling(window=14, min_periods=14).min()
        highest_high = df["High"].rolling(window=14, min_periods=14).max()
        stoch_k = ((df["Close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)) * 100
        df["STOCHk_14_3_3"] = stoch_k
        df["STOCHd_14_3_3"] = stoch_k.rolling(window=3, min_periods=3).mean()

    last = df.iloc[-1]
    rsi = float(last.get("RSI_14", np.nan))

    if not np.isnan(rsi):
        if rsi <= 30:
            rsi_signal = "OVERSOLD"
        elif rsi >= 70:
            rsi_signal = "OVERBOUGHT"
        else:
            rsi_signal = "NEUTRAL"
    else:
        rsi_signal = "UNKNOWN"

    stoch_k = float(last.get("STOCHk_14_3_3", np.nan))
    stoch_d = float(last.get("STOCHd_14_3_3", np.nan))

    if not np.isnan(stoch_k):
        if stoch_k <= 20:
            stoch_signal = "OVERSOLD"
        elif stoch_k >= 80:
            stoch_signal = "OVERBOUGHT"
        else:
            stoch_signal = "NEUTRAL"
    else:
        stoch_signal = "UNKNOWN"

    return MomentumIndicators(
        rsi=rsi if not np.isnan(rsi) else 50.0,
        rsi_signal=rsi_signal,
        stoch_k=stoch_k if not np.isnan(stoch_k) else 50.0,
        stoch_d=stoch_d if not np.isnan(stoch_d) else 50.0,
        stoch_signal=stoch_signal,
    )


def _compute_volatility(df: pd.DataFrame) -> VolatilityIndicators:
    """Compute ATR and Bollinger Bands."""
    df = df.copy()

    if HAS_PANDAS_TA:
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
    else:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATRr_14"] = tr.rolling(window=14, min_periods=14).mean()

        bb_mid = df["Close"].rolling(window=20, min_periods=20).mean()
        bb_std = df["Close"].rolling(window=20, min_periods=20).std()
        df["BBM_20_2.0"] = bb_mid
        df["BBU_20_2.0"] = bb_mid + 2 * bb_std
        df["BBL_20_2.0"] = bb_mid - 2 * bb_std

    last = df.iloc[-1]
    close = float(last["Close"])

    atr = float(last.get("ATRr_14", np.nan))
    if np.isnan(atr):
        atr = 0.0
    atr_pct = (atr / close * 100.0) if close > 0 else 0.0

    bb_upper = float(last.get("BBU_20_2.0", np.nan))
    bb_middle = float(last.get("BBM_20_2.0", np.nan))
    bb_lower = float(last.get("BBL_20_2.0", np.nan))

    if any(np.isnan(x) for x in (bb_upper, bb_middle, bb_lower)):
        bb_upper = close * 1.02
        bb_middle = close
        bb_lower = close * 0.98

    bb_width_pct = (bb_upper - bb_lower) / max(1e-9, bb_middle) * 100.0
    bb_position = (close - bb_lower) / max(1e-9, (bb_upper - bb_lower))

    return VolatilityIndicators(
        atr=atr,
        atr_pct=atr_pct,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_width_pct=bb_width_pct,
        bb_position=bb_position,
    )


def _compute_volume(df: pd.DataFrame) -> VolumeIndicators:
    """Compute OBV, VWAP, and volume ratios."""
    df = df.copy()

    if HAS_PANDAS_TA:
        df.ta.obv(append=True)
        df.ta.vwap(append=True)
    else:
        direction = np.sign(df["Close"].diff().fillna(0.0))
        df["OBV"] = (direction * df["Volume"]).cumsum()
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        cumulative_vol = df["Volume"].cumsum().replace(0, np.nan)
        df["VWAP_D"] = (typical_price * df["Volume"]).cumsum() / cumulative_vol

    last = df.iloc[-1]
    close = float(last["Close"])

    obv = float(last.get("OBV", 0.0))
    vwap = float(last.get("VWAP", last.get("VWAP_D", close)))

    if len(df) >= 20:
        obv_20 = float(df.iloc[-20].get("OBV", 0.0))
        if obv > obv_20 * 1.1:
            obv_trend = "INCREASING"
        elif obv < obv_20 * 0.9:
            obv_trend = "DECREASING"
        else:
            obv_trend = "FLAT"
    else:
        obv_trend = "UNKNOWN"

    avg_volume = float(df["Volume"].tail(20).mean())
    volume_ratio = float(last["Volume"]) / avg_volume if avg_volume > 0 else 1.0

    return VolumeIndicators(
        obv=obv,
        obv_trend=obv_trend,
        volume_ratio=volume_ratio,
        vwap=vwap,
    )


def _compute_macd(df: pd.DataFrame) -> MACDData:
    """Compute MACD line, signal, and histogram."""
    df = df.copy()

    if HAS_PANDAS_TA:
        df.ta.macd(append=True)
    else:
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["MACD_12_26_9"] = macd_line
        df["MACDs_12_26_9"] = signal_line
        df["MACDh_12_26_9"] = macd_line - signal_line

    last = df.iloc[-1]
    macd_line = float(last.get("MACD_12_26_9", 0.0))
    signal_line = float(last.get("MACDs_12_26_9", 0.0))
    histogram = float(last.get("MACDh_12_26_9", 0.0))

    if histogram > 0 and macd_line > signal_line:
        signal = "BULLISH"
    elif histogram < 0 and macd_line < signal_line:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return MACDData(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram,
        signal=signal,
    )


def _compute_support_resistance(df: pd.DataFrame) -> SupportResistance:
    """Compute simple support/resistance and pivot from recent highs/lows."""
    high_20 = float(df["High"].tail(20).max())
    low_20 = float(df["Low"].tail(20).min())
    last_close = float(df["Close"].iloc[-1])

    pivot = (high_20 + low_20 + last_close) / 3.0

    support_1 = low_20
    support_2 = low_20 * 0.98
    resistance_1 = high_20
    resistance_2 = high_20 * 1.02

    return SupportResistance(
        support_1=support_1,
        support_2=support_2,
        resistance_1=resistance_1,
        resistance_2=resistance_2,
        pivot=pivot,
    )


def _compute_price_action(df: pd.DataFrame) -> PriceAction:
    """Compute price action metrics."""
    last = df.iloc[-1]
    close = float(last["Close"])
    vol = int(last["Volume"])

    # 1d, 1w (5 bars), 1m (20 bars) performance
    change_1d = _safe_pct_change(close, float(df.iloc[-2]["Close"])) if len(df) >= 2 else 0.0
    change_1w = _safe_pct_change(close, float(df.iloc[-5]["Close"])) if len(df) >= 5 else 0.0
    change_1m = _safe_pct_change(close, float(df.iloc[-20]["Close"])) if len(df) >= 20 else 0.0

    if len(df) >= 252:
        high_52w = float(df["High"].tail(252).max())
        low_52w = float(df["Low"].tail(252).min())
    else:
        high_52w = float(df["High"].max())
        low_52w = float(df["Low"].min())

    distance_from_52w_high_pct = _safe_pct_change(high_52w, close) * -1.0

    avg_volume_20d = float(df["Volume"].tail(20).mean())

    return PriceAction(
        last_price=close,
        change_pct_1d=change_1d,
        change_pct_1w=change_1w,
        change_pct_1m=change_1m,
        volume=vol,
        avg_volume_20d=avg_volume_20d,
        high_52w=high_52w,
        low_52w=low_52w,
        distance_from_52w_high_pct=distance_from_52w_high_pct,
    )


def calculate_hard_technicals(
    df: pd.DataFrame,
    ticker: str = "",
    timeframe: str = "1d",
) -> Dict[str, object]:
    """
    Compute 20+ technical indicators using pandas_ta.

    PURE MATHEMATICAL ANALYSIS – no AI/ML.

    Args:
        df: OHLCV DataFrame.
        ticker: Stock symbol (for metadata).
        timeframe: Timeframe label (for metadata).

    Returns:
        dict: HardTechnicals as a JSON-serializable dict.

    Raises:
        ValueError: If data is insufficient or malformed.
    """
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data: require >= 50 bars, got {len(df)}.")

    _ensure_ohlcv(df)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    trend = _compute_trend(df)
    momentum = _compute_momentum(df)
    volatility = _compute_volatility(df)
    volume = _compute_volume(df)
    macd = _compute_macd(df)
    support_resistance = _compute_support_resistance(df)
    price_action = _compute_price_action(df)
    data_quality = validate_data_quality(df)

    ht = HardTechnicals(
        ticker=ticker.upper() if ticker else "",
        timeframe=timeframe,
        price_action=price_action,
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        volume=volume,
        macd=macd,
        support_resistance=support_resistance,
        data_quality=data_quality,
        computed_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Technicals computed for %s (tf=%s): trend=%s, RSI=%.1f, ATR%%=%.2f, quality=%.1f",
        ht.ticker,
        timeframe,
        trend.trend,
        momentum.rsi,
        volatility.atr_pct,
        data_quality.quality_score,
    )

    return asdict(ht)


# ──────────────────────────────────────────────────────────
# CLI TOOL
# ──────────────────────────────────────────────────────────


def _cli() -> None:
    """
    Simple CLI for quick debugging of The Mathematician.

    Usage:
        python data_engine.py TICKER [PERIOD] [INTERVAL]

    Example:
        python data_engine.py AAPL 3mo 1d
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_engine.py TICKER [PERIOD] [INTERVAL]")
        print("Example: python data_engine.py AAPL 3mo 1d")
        print()
        print("Periods:   1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
        print("Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
        raise SystemExit(1)

    ticker = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "3mo"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"

    print(f"\nFetching data for {ticker} (period={period}, interval={interval})...\n")
    try:
        df = get_market_data(ticker, period=period, interval=interval)
        print(f"✅ Fetched {len(df)} bars")
        print(f"Date range: {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}\n")

        print("Computing technicals...\n")
        tech = calculate_hard_technicals(df, ticker=ticker, timeframe=interval)

        print(json.dumps(tech, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
