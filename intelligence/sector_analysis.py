"""
============================================================
ALPHA-PRIME v2.0 - Sector Analysis
============================================================

Analyzes sector performance, rotations, and correlations:

Why Sector Analysis?
- Stock performance reflects a mix of stock-specific, sector, and market factors. [web:311][web:317]
- Strong stocks in strong sectors tend to trend more persistently than isolated winners.
- Weak stocks in weak sectors face persistent headwinds.
- Sector rotation helps infer the current phase of the market cycle. [web:314][web:318][web:322]
- Capital systematically flows between cyclical and defensive sectors over time.

11 S&P Sectors (GICS Classification via SPDR Sector ETFs): [web:311][web:313][web:317][web:319]
1. Technology (XLK):
   - Software, semiconductors, hardware.
   - High growth, high volatility.
   - Often leads in bull markets; can lag in recessions.

2. Financials (XLF):
   - Banks, insurance, asset managers, brokers.
   - Sensitive to interest rates and credit conditions.
   - Tends to lead in early economic recoveries.

3. Healthcare (XLV):
   - Pharma, biotech, healthcare providers.
   - Defensive, relatively stable through cycles.

4. Consumer Discretionary (XLY):
   - Retail, autos, leisure, e-commerce.
   - Cyclical; benefits from strong consumer spending.

5. Communication Services (XLC):
   - Telecom, media, internet platforms.
   - Growth-tilted with some defensive characteristics.

6. Industrials (XLI):
   - Manufacturing, transportation, aerospace, defense.
   - Cyclical; often early-cycle leader.

7. Consumer Staples (XLP):
   - Food, beverages, household goods.
   - Defensive; demand stable across cycles.

8. Energy (XLE):
   - Oil, gas, energy equipment and services.
   - Tied to commodity and inflation dynamics.

9. Utilities (XLU):
   - Electric, gas, water utilities.
   - Defensive, low growth, rate-sensitive.

10. Real Estate (XLRE):
    - REITs and listed property companies.
    - Sensitive to rates and property cycles.

11. Materials (XLB):
    - Chemicals, metals, mining, basic materials.
    - Cyclical; benefits from industrial demand.

Sector Rotation Model (economic cycle view): [web:314][web:318][web:320][web:322]
1. RECOVERY PHASE:
   - Leaders: Financials, Industrials, some Consumer Discretionary.
   - Laggards: Utilities, Consumer Staples.
   - Rationale: Risk appetite returning, early-cycle cyclicals bid.

2. EXPANSION PHASE:
   - Leaders: Technology, Consumer Discretionary, growth sectors.
   - Laggards: Defensives.
   - Rationale: Earnings growth broad-based, credit conditions easy.

3. SLOWDOWN PHASE:
   - Leaders: Energy, Materials, late-cycle cyclicals.
   - Laggards: High-multiple growth.
   - Rationale: Inflation and input-cost sectors outperform as growth decelerates.

4. RECESSION PHASE:
   - Leaders: Utilities, Consumer Staples, Healthcare.
   - Laggards: Cyclicals (Tech, Discretionary, Industrials, Financials).
   - Rationale: Flight to safety and earnings resilience.

Relative Strength:
- RS = (1 + stock return) / (1 + sector return).
- RS > 1 → Outperforming sector.
- RS < 1 → Underperforming sector.
- RS > 1.1 → Strong outperformance.
- RS < 0.9 → Meaningful underperformance. [web:320]

Usage:
    from intelligence.sector_analysis import (
        get_sector_performance,
        detect_sector_rotation,
        calculate_relative_strength,
        get_ticker_sector_context,
    )

    sectors = get_sector_performance(period="1y")
    rotation = detect_sector_rotation()
    rs = calculate_relative_strength("AAPL", "XLK")
    context = get_ticker_sector_context("AAPL")

Integration:
- research_engine.py uses sector metrics for context and risk filters.
- brain.py adjusts conviction based on sector strength and rotation.
- Dashboard displays sector heatmaps, rotation state, and ticker context.
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from diskcache import Cache

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()
cache = Cache(f"{settings.cache_dir}/sector")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & SECTOR DEFINITIONS
# ──────────────────────────────────────────────────────────


class MarketPhase(str, Enum):
    """Market cycle phases used in sector rotation analysis."""

    RECOVERY = "RECOVERY"
    EXPANSION = "EXPANSION"
    SLOWDOWN = "SLOWDOWN"
    RECESSION = "RECESSION"
    UNKNOWN = "UNKNOWN"


SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

SECTOR_CHARACTERISTICS: Dict[str, Dict[str, str]] = {
    "Technology": {"type": "Growth", "cyclical": "True", "volatility": "High"},
    "Financials": {"type": "Cyclical", "cyclical": "True", "volatility": "Medium"},
    "Healthcare": {"type": "Defensive", "cyclical": "False", "volatility": "Low"},
    "Consumer Discretionary": {"type": "Cyclical", "cyclical": "True", "volatility": "Medium"},
    "Communication Services": {"type": "Mixed", "cyclical": "True", "volatility": "Medium"},
    "Industrials": {"type": "Cyclical", "cyclical": "True", "volatility": "Medium"},
    "Consumer Staples": {"type": "Defensive", "cyclical": "False", "volatility": "Low"},
    "Energy": {"type": "Cyclical", "cyclical": "True", "volatility": "High"},
    "Utilities": {"type": "Defensive", "cyclical": "False", "volatility": "Very Low"},
    "Real Estate": {"type": "Rate-Sensitive", "cyclical": "True", "volatility": "Medium"},
    "Materials": {"type": "Cyclical", "cyclical": "True", "volatility": "Medium"},
}


@dataclass
class SectorPerformance:
    """
    Performance metrics for a sector ETF over multiple horizons.

    Attributes:
        sector_name: Human-readable sector name.
        ticker: Sector ETF ticker (e.g. XLK, XLF).
        current_price: Latest closing price.
        return_1d: 1‑day percent return.
        return_1w: 1‑week (~5 trading days) percent return.
        return_1m: 1‑month (~21 trading days) percent return.
        return_3m: 3‑month percent return.
        return_ytd: Year‑to‑date percent return.
        return_1y: 1‑year percent return.
        volume: Latest daily volume.
        relative_strength_spy: Relative strength vs SPY on 1‑month horizon.
        momentum_score: Heuristic 0–100 momentum score.
        rank: Rank within sectors (1 = strongest on 1‑month performance).
        characteristics: Static metadata (type, cyclical flag, volatility).
    """

    sector_name: str
    ticker: str
    current_price: float
    return_1d: float
    return_1w: float
    return_1m: float
    return_3m: float
    return_ytd: float
    return_1y: float
    volume: int
    relative_strength_spy: float
    momentum_score: float
    rank: int
    characteristics: Dict[str, str]


@dataclass
class SectorRotation:
    """
    Sector rotation state derived from cross‑sector performance.

    Attributes:
        market_phase: MarketPhase classification.
        leading_sectors: Top 3 sectors by recent returns.
        lagging_sectors: Bottom 3 sectors by recent returns.
        rotation_strength: 0–100 measure of dispersion between leaders and laggards.
        defensive_exposure: % of leaders that are defensive sectors.
        cyclical_exposure: % of leaders that are cyclical sectors.
        rotation_detected: Whether rotation_strength exceeds a minimum threshold.
        rationale: Explanatory bullet points for current classification.
        analyzed_at_utc: ISO timestamp when computed.
    """

    market_phase: MarketPhase
    leading_sectors: List[str]
    lagging_sectors: List[str]
    rotation_strength: float
    defensive_exposure: float
    cyclical_exposure: float
    rotation_detected: bool
    rationale: List[str]
    analyzed_at_utc: str


@dataclass
class SectorMomentum:
    """
    Momentum view for a single sector ETF.

    Attributes:
        sector_name: Sector name.
        momentum_score: 0–100 momentum score using recent returns.
        trend: "UP", "DOWN", or "SIDEWAYS".
        strength: "STRONG", "MODERATE", or "WEAK".
        ema_alignment: True if EMA9 > EMA20 > EMA50 (bullish structure).
        rsi: Latest 14‑day RSI value.
        volume_trend: "INCREASING" or "DECREASING" 20‑day volume trend.
    """

    sector_name: str
    momentum_score: float
    trend: str
    strength: str
    ema_alignment: bool
    rsi: float
    volume_trend: str


@dataclass
class SectorContext:
    """
    Sector context for an individual ticker.

    Attributes:
        ticker: Stock symbol.
        sector: Mapped sector name.
        sector_etf: Sector ETF ticker.
        sector_performance_1m: Sector 1‑month return.
        ticker_performance_1m: Stock 1‑month return.
        relative_strength: RS = (1+stock) / (1+sector).
        sector_rank: Sector rank (1–11).
        outperforming_sector: True if RS > 1.
        sector_momentum: Sector momentum score.
        recommendation: "FAVORABLE", "NEUTRAL", or "UNFAVORABLE".
    """

    ticker: str
    sector: str
    sector_etf: str
    sector_performance_1m: float
    ticker_performance_1m: float
    relative_strength: float
    sector_rank: int
    outperforming_sector: bool
    sector_momentum: float
    recommendation: str


# ──────────────────────────────────────────────────────────
# SECTOR PERFORMANCE TRACKING
# ──────────────────────────────────────────────────────────


def fetch_sector_data(sector_etf: str, period: str = "1y") -> pd.DataFrame:
    """
    Download daily OHLCV history for a sector ETF via yfinance. [web:309][web:311][web:313]

    Args:
        sector_etf: Sector ETF ticker (e.g. "XLK").
        period: History window, e.g. "6mo", "1y".

    Returns:
        DataFrame indexed by date with OHLCV columns.
    """
    try:
        df = yf.download(sector_etf, period=period, interval="1d", progress=False)
        if df.empty:
            logger.warning("Empty data for sector ETF %s.", sector_etf)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching data for %s: %s", sector_etf, exc, exc_info=True)
        return pd.DataFrame()


def _safe_return(current: float, past: float) -> float:
    """Compute percent return guarding against zero prices."""
    if past <= 0:
        return 0.0
    return (current / past - 1.0) * 100.0


def calculate_returns(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute multi‑horizon returns from a price history.

    Periods:
        - 1d, 1w (5 days), 1m (21 days), 3m (63 days), YTD, 1y.

    Args:
        df: OHLCV DataFrame.

    Returns:
        Mapping from period label to percent return.
    """
    if df.empty or "Close" not in df.columns:
        return {"1d": 0.0, "1w": 0.0, "1m": 0.0, "3m": 0.0, "ytd": 0.0, "1y": 0.0}

    closes = df["Close"].dropna()
    if len(closes) < 2:
        return {"1d": 0.0, "1w": 0.0, "1m": 0.0, "3m": 0.0, "ytd": 0.0, "1y": 0.0}

    current_price = float(closes.iloc[-1])
    idx = closes.index

    res: Dict[str, float] = {}

    if len(closes) >= 2:
        res["1d"] = _safe_return(current_price, float(closes.iloc[-2]))
    else:
        res["1d"] = 0.0

    if len(closes) >= 6:
        res["1w"] = _safe_return(current_price, float(closes.iloc[-6]))
    else:
        res["1w"] = 0.0

    if len(closes) >= 22:
        res["1m"] = _safe_return(current_price, float(closes.iloc[-22]))
    else:
        res["1m"] = 0.0

    if len(closes) >= 64:
        res["3m"] = _safe_return(current_price, float(closes.iloc[-64]))
    else:
        res["3m"] = 0.0

    try:
        year_mask = idx.year == idx[-1].year
        year_data = closes[year_mask]
        if not year_data.empty:
            year_start_price = float(year_data.iloc[0])
            res["ytd"] = _safe_return(current_price, year_start_price)
        else:
            res["ytd"] = 0.0
    except Exception:  # noqa: BLE001
        res["ytd"] = 0.0

    if len(closes) >= 252:
        res["1y"] = _safe_return(current_price, float(closes.iloc[-252]))
    else:
        res["1y"] = res["ytd"]

    return res


def get_sector_performance(period: str = "1y") -> Dict[str, SectorPerformance]:
    """
    Retrieve performance metrics for all defined sectors.

    Args:
        period: yfinance period for historical data (e.g. "6mo", "1y").

    Returns:
        Mapping from sector name to SectorPerformance.

    Example:
        >>> perf = get_sector_performance("1y")
        >>> perf["Technology"].return_1m
    """
    cache_key = f"sector_perf_{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Sector performance cache hit for period %s.", period)
        return cached

    logger.info("Fetching sector performance data for period %s.", period)

    spy_df = fetch_sector_data("SPY", period=period)
    spy_returns = calculate_returns(spy_df)

    performances: Dict[str, SectorPerformance] = {}

    for sector_name, etf_ticker in SECTOR_ETFS.items():
        try:
            df = fetch_sector_data(etf_ticker, period=period)
            if df.empty:
                logger.warning("No data for sector %s (%s).", sector_name, etf_ticker)
                continue

            rets = calculate_returns(df)
            closes = df["Close"].dropna()
            vols = df["Volume"].dropna()
            current_price = float(closes.iloc[-1])
            volume = int(vols.iloc[-1]) if not vols.empty else 0

            spy_1m = spy_returns.get("1m", 0.0)
            if spy_1m > -99.0:
                rs_spy = (1.0 + rets.get("1m", 0.0) / 100.0) / (1.0 + spy_1m / 100.0)
            else:
                rs_spy = 1.0

            momentum_score = float(
                max(0.0, min(100.0, 50.0 + rets.get("1m", 0.0) + rets.get("3m", 0.0) / 2.0))
            )

            perf = SectorPerformance(
                sector_name=sector_name,
                ticker=etf_ticker,
                current_price=current_price,
                return_1d=float(rets.get("1d", 0.0)),
                return_1w=float(rets.get("1w", 0.0)),
                return_1m=float(rets.get("1m", 0.0)),
                return_3m=float(rets.get("3m", 0.0)),
                return_ytd=float(rets.get("ytd", 0.0)),
                return_1y=float(rets.get("1y", 0.0)),
                volume=volume,
                relative_strength_spy=float(rs_spy),
                momentum_score=momentum_score,
                rank=0,
                characteristics=SECTOR_CHARACTERISTICS.get(sector_name, {}),
            )
            performances[sector_name] = perf
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing sector %s: %s", sector_name, exc, exc_info=True)

    sorted_perf = sorted(performances.values(), key=lambda s: s.return_1m, reverse=True)
    for rank, sp in enumerate(sorted_perf, start=1):
        sp.rank = rank

    logger.info("Fetched performance for %d sectors.", len(performances))
    cache.set(cache_key, performances, expire=60 * 60)
    return performances


# ──────────────────────────────────────────────────────────
# SECTOR ROTATION DETECTION
# ──────────────────────────────────────────────────────────


def detect_sector_rotation(lookback_days: int = 30) -> SectorRotation:
    """
    Infer current sector rotation and market phase from sector returns. [web:314][web:318][web:320][web:322]

    The current implementation uses 1‑month performance (≈ lookback_days)
    as a proxy for recent leadership.

    Args:
        lookback_days: Window in calendar days (currently advisory only).

    Returns:
        SectorRotation object describing leaders, laggards, and phase.
    """
    _ = lookback_days  # reserved; 1m mapping is handled via get_sector_performance
    sectors = get_sector_performance(period="1y")
    if not sectors:
        return SectorRotation(
            market_phase=MarketPhase.UNKNOWN,
            leading_sectors=[],
            lagging_sectors=[],
            rotation_strength=0.0,
            defensive_exposure=0.0,
            cyclical_exposure=0.0,
            rotation_detected=False,
            rationale=["Insufficient sector data."],
            analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    sorted_perf = sorted(sectors.values(), key=lambda s: s.return_1m, reverse=True)
    leading = [s.sector_name for s in sorted_perf[:3]]
    lagging = [s.sector_name for s in sorted_perf[-3:]]

    defensive = {"Healthcare", "Consumer Staples", "Utilities"}
    cyclical = {
        "Technology",
        "Financials",
        "Consumer Discretionary",
        "Communication Services",
        "Industrials",
        "Energy",
        "Materials",
        "Real Estate",
    }

    defensive_returns = [s.return_1m for s in sectors.values() if s.sector_name in defensive]
    cyclical_returns = [s.return_1m for s in sectors.values() if s.sector_name in cyclical]

    avg_def = float(np.mean(defensive_returns)) if defensive_returns else 0.0
    avg_cyc = float(np.mean(cyclical_returns)) if cyclical_returns else 0.0

    rationale: List[str] = []
    phase = MarketPhase.UNKNOWN

    if avg_cyc > avg_def + 2.0:
        if "Financials" in leading[:2] and "Industrials" in leading[:3]:
            phase = MarketPhase.RECOVERY
            rationale.append("Financials and Industrials leading → Recovery / early‑cycle.")
        elif "Technology" in leading[:2] or "Consumer Discretionary" in leading[:2]:
            phase = MarketPhase.EXPANSION
            rationale.append("Technology/Discretionary leading → Expansion / growth phase.")
        else:
            phase = MarketPhase.EXPANSION
            rationale.append("Cyclicals broadly outperforming defensives → Risk‑on / Expansion.")
    elif avg_def > avg_cyc + 2.0:
        phase = MarketPhase.RECESSION
        rationale.append("Defensive sectors outperform cyclicals → Recession / risk‑off.")
    elif "Energy" in leading or "Materials" in leading:
        phase = MarketPhase.SLOWDOWN
        rationale.append("Energy/Materials leadership → Late‑cycle / Slowdown characteristics.")
    else:
        phase = MarketPhase.UNKNOWN
        rationale.append("Mixed sector signals; no clear rotation regime.")

    top3_avg = float(np.mean([s.return_1m for s in sorted_perf[:3]]))
    bottom3_avg = float(np.mean([s.return_1m for s in sorted_perf[-3:]]))
    rotation_strength = float(min(100.0, abs(top3_avg - bottom3_avg) * 5.0))
    rotation_detected = rotation_strength > 30.0

    defensive_count = sum(1 for s in leading if s in defensive)
    cyclical_count = sum(1 for s in leading if s in cyclical)
    defensive_exposure = defensive_count / 3.0 * 100.0 if leading else 0.0
    cyclical_exposure = cyclical_count / 3.0 * 100.0 if leading else 0.0

    rotation = SectorRotation(
        market_phase=phase,
        leading_sectors=leading,
        lagging_sectors=lagging,
        rotation_strength=rotation_strength,
        defensive_exposure=defensive_exposure,
        cyclical_exposure=cyclical_exposure,
        rotation_detected=rotation_detected,
        rationale=rationale,
        analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Sector rotation: phase=%s, leaders=%s, laggards=%s, strength=%.1f.",
        phase.value,
        ", ".join(leading),
        ", ".join(lagging),
        rotation_strength,
    )
    return rotation


# ──────────────────────────────────────────────────────────
# RELATIVE STRENGTH & CORRELATIONS
# ──────────────────────────────────────────────────────────


def calculate_relative_strength(
    ticker: str,
    sector_etf: str,
    period: str = "3mo",
) -> float:
    """
    Calculate relative strength (RS) of a stock vs its sector ETF. [web:320]

    RS = (1 + stock_return) / (1 + sector_return).

    Args:
        ticker: Stock symbol.
        sector_etf: Sector ETF ticker.
        period: yfinance period string.

    Returns:
        RS ratio (float). 1.0 on failure.
    """
    try:
        stock_df = yf.download(ticker, period=period, interval="1d", progress=False)
        sector_df = yf.download(sector_etf, period=period, interval="1d", progress=False)
        if stock_df.empty or sector_df.empty:
            return 1.0

        stock_closes = stock_df["Close"].dropna()
        sector_closes = sector_df["Close"].dropna()

        if len(stock_closes) < 2 or len(sector_closes) < 2:
            return 1.0

        stock_ret = float(stock_closes.iloc[-1] / stock_closes.iloc[0] - 1.0)
        sector_ret = float(sector_closes.iloc[-1] / sector_closes.iloc[0] - 1.0)

        if sector_ret <= -1.0:
            return 1.0

        rs = (1.0 + stock_ret) / (1.0 + sector_ret)
        logger.debug("Relative strength %s vs %s over %s: %.3f.", ticker, sector_etf, period, rs)
        return rs
    except Exception as exc:  # noqa: BLE001
        logger.error("Error calculating relative strength %s vs %s: %s", ticker, sector_etf, exc, exc_info=True)
        return 1.0


def get_sector_correlation_matrix(sectors: List[str]) -> pd.DataFrame:
    """
    Compute correlation matrix between selected sector ETFs. [web:309][web:313]

    Args:
        sectors: List of sector names (keys from SECTOR_ETFS) or tickers.

    Returns:
        Pandas DataFrame of daily return correlations.
    """
    tickers: List[str] = []
    for s in sectors:
        if s in SECTOR_ETFS:
            tickers.append(SECTOR_ETFS[s])
        else:
            tickers.append(s)

    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(tickers, period="6mo", interval="1d", progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        corr = returns.corr()
        logger.info("Computed sector correlation matrix for %d series.", len(tickers))
        return corr
    except Exception as exc:  # noqa: BLE001
        logger.error("Error computing sector correlation matrix: %s", exc, exc_info=True)
        return pd.DataFrame()


def identify_leading_sectors(lookback_days: int = 21) -> List[str]:
    """
    Identify top‑performing sectors over a short lookback window.

    Args:
        lookback_days: Approx number of trading days (defaults to ~1 month).

    Returns:
        List of sector names sorted by performance descending.
    """
    period = "1mo" if lookback_days <= 22 else "3mo"
    perf = get_sector_performance(period="1y")
    sorted_sectors = sorted(perf.values(), key=lambda s: s.return_1m, reverse=True)
    leaders = [s.sector_name for s in sorted_sectors[:5]]
    logger.info("Leading sectors over %d‑day lookback: %s.", lookback_days, ", ".join(leaders))
    return leaders


# ──────────────────────────────────────────────────────────
# SECTOR MOMENTUM
# ──────────────────────────────────────────────────────────


def analyze_sector_momentum(sector: str) -> SectorMomentum:
    """
    Analyze momentum profile for a given sector ETF.

    Uses EMA alignment, RSI, recent volume trend, and returns for scoring.

    Args:
        sector: Sector name (key in SECTOR_ETFS).

    Returns:
        SectorMomentum instance.

    Raises:
        ValueError if sector is unknown or data insufficient.
    """
    etf = SECTOR_ETFS.get(sector)
    if not etf:
        raise ValueError(f"Unknown sector: {sector}")

    df = fetch_sector_data(etf, period="6mo")
    if df.empty or len(df) < 60:
        raise ValueError(f"Insufficient data for sector {sector} ({etf}).")

    closes = df["Close"].copy()
    df["EMA9"] = closes.ewm(span=9).mean()
    df["EMA20"] = closes.ewm(span=20).mean()
    df["EMA50"] = closes.ewm(span=50).mean()

    last_row = df.iloc[-1]
    ema_alignment = bool(last_row["EMA9"] > last_row["EMA20"] > last_row["EMA50"])

    delta = closes.diff()
    gain = delta.clip(lower=0.0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    current_rsi = float(rsi_series.iloc[-1])

    if ema_alignment and current_rsi > 50.0:
        trend = "UP"
        strength = "STRONG" if current_rsi > 60.0 else "MODERATE"
    elif (not ema_alignment) and current_rsi < 50.0:
        trend = "DOWN"
        strength = "STRONG" if current_rsi < 40.0 else "MODERATE"
    else:
        trend = "SIDEWAYS"
        strength = "WEAK"

    vol = df["Volume"].dropna()
    avg_volume_20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    recent_volume = float(vol.iloc[-5:].mean()) if len(vol) >= 5 else float(vol.mean())
    volume_trend = "INCREASING" if recent_volume > avg_volume_20 * 1.1 else "DECREASING"

    rets = calculate_returns(df)
    momentum_score = float(
        max(0.0, min(100.0, 50.0 + rets.get("1m", 0.0) + rets.get("3m", 0.0) / 2.0))
    )

    momentum = SectorMomentum(
        sector_name=sector,
        momentum_score=momentum_score,
        trend=trend,
        strength=strength,
        ema_alignment=ema_alignment,
        rsi=current_rsi,
        volume_trend=volume_trend,
    )

    logger.info(
        "Sector momentum %s (%s): trend=%s, strength=%s, RSI=%.1f.",
        sector,
        etf,
        trend,
        strength,
        current_rsi,
    )
    return momentum


# ──────────────────────────────────────────────────────────
# TICKER SECTOR CONTEXT
# ──────────────────────────────────────────────────────────


def get_ticker_sector(ticker: str) -> Tuple[str, str]:
    """
    Map a stock ticker to one of the canonical sectors and ETFs. [web:317][web:320]

    Uses Yahoo Finance metadata to infer sector, then maps to ALPHA‑PRIME
    sector names and associated ETF.

    Args:
        ticker: Stock symbol.

    Returns:
        (sector_name, sector_etf) tuple. Falls back to Technology/XLK.
    """
    try:
        info = yf.Ticker(ticker).info
        raw_sector = str(info.get("sector", "Technology"))

        mapping: Dict[str, str] = {
            "Technology": "Technology",
            "Financial Services": "Financials",
            "Financial": "Financials",
            "Healthcare": "Healthcare",
            "Consumer Cyclical": "Consumer Discretionary",
            "Consumer Discretionary": "Consumer Discretionary",
            "Communication Services": "Communication Services",
            "Industrials": "Industrials",
            "Consumer Defensive": "Consumer Staples",
            "Consumer Staples": "Consumer Staples",
            "Energy": "Energy",
            "Utilities": "Utilities",
            "Real Estate": "Real Estate",
            "Basic Materials": "Materials",
            "Materials": "Materials",
        }
        sector_name = mapping.get(raw_sector, "Technology")
        sector_etf = SECTOR_ETFS.get(sector_name, "XLK")
        return sector_name, sector_etf
    except Exception as exc:  # noqa: BLE001
        logger.error("Error resolving sector for %s: %s", ticker, exc, exc_info=True)
        return "Technology", "XLK"


def get_ticker_sector_context(ticker: str) -> SectorContext:
    """
    Build a sector context snapshot for a given stock.

    Combines sector rank, sector returns, ticker returns, and RS to
    flag whether the setup is favorable, neutral, or unfavorable.

    Args:
        ticker: Stock symbol.

    Returns:
        SectorContext object.
    """
    sector_name, sector_etf = get_ticker_sector(ticker)
    sectors = get_sector_performance()
    sector_perf = sectors.get(sector_name)
    if sector_perf is None:
        raise ValueError(f"Sector performance not available for {sector_name}.")

    try:
        stock_df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        stock_returns = calculate_returns(stock_df)
        ticker_1m = float(stock_returns.get("1m", 0.0))
    except Exception:  # noqa: BLE001
        ticker_1m = 0.0

    rs = calculate_relative_strength(ticker, sector_etf, period="3mo")

    if sector_perf.rank <= 4 and rs > 1.05:
        recommendation = "FAVORABLE"
    elif sector_perf.rank >= 8 or rs < 0.95:
        recommendation = "UNFAVORABLE"
    else:
        recommendation = "NEUTRAL"

    context = SectorContext(
        ticker=ticker.upper(),
        sector=sector_name,
        sector_etf=sector_etf,
        sector_performance_1m=sector_perf.return_1m,
        ticker_performance_1m=ticker_1m,
        relative_strength=rs,
        sector_rank=sector_perf.rank,
        outperforming_sector=bool(rs > 1.0),
        sector_momentum=sector_perf.momentum_score,
        recommendation=recommendation,
    )

    logger.info(
        "Sector context %s: sector=%s (rank %d), RS=%.2f, recommendation=%s.",
        ticker,
        sector_name,
        sector_perf.rank,
        rs,
        recommendation,
    )
    return context


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Sector Analysis")
    print("=" * 70 + "\n")

    print("SECTOR PERFORMANCE (1‑Month)")
    print("-" * 70)

    try:
        perf_dict = get_sector_performance(period="1y")
        ordered = sorted(perf_dict.values(), key=lambda s: s.return_1m, reverse=True)

        for sp in ordered:
            icon = "🟢" if sp.return_1m > 0 else "🔴"
            print(f"#{sp.rank:2d} {icon} {sp.sector_name:24} ({sp.ticker}): {sp.return_1m:+6.2f}%")
            print(
                f"    RS vs SPY: {sp.relative_strength_spy:.2f}x, "
                f"Type: {sp.characteristics.get('type','N/A')}, "
                f"Vol: {sp.volume:,}"
            )
            print()

        print("=" * 70)
        print("SECTOR ROTATION")
        print("-" * 70)

        rotation = detect_sector_rotation()
        print(f"Market Phase      : {rotation.market_phase.value}")
        print(f"Rotation Strength : {rotation.rotation_strength:.0f}/100")
        print(f"Defensive Exposure: {rotation.defensive_exposure:.0f}%")
        print(f"Cyclical Exposure : {rotation.cyclical_exposure:.0f}%\n")

        print("Leading Sectors:")
        for s_name in rotation.leading_sectors:
            print(f"  ✅ {s_name}")
        print("\nLagging Sectors:")
        for s_name in rotation.lagging_sectors:
            print(f"  ❌ {s_name}")
        print("\nRationale:")
        for r in rotation.rationale:
            print(f"  - {r}")

        if len(sys.argv) > 1:
            ticker_cli = sys.argv[1].upper()
            print("\n" + "=" * 70)
            print(f"SECTOR CONTEXT: {ticker_cli}")
            print("-" * 70)

            try:
                ctx = get_ticker_sector_context(ticker_cli)
                print(f"Sector               : {ctx.sector}")
                print(f"Sector ETF           : {ctx.sector_etf}")
                print(f"Sector Rank          : #{ctx.sector_rank}/11")
                print(f"Sector Perf (1M)     : {ctx.sector_performance_1m:+.2f}%")
                print(f"{ticker_cli} Perf (1M)      : {ctx.ticker_performance_1m:+.2f}%")
                print(f"Relative Strength    : {ctx.relative_strength:.2f}x")
                print(f"Outperforming Sector : {'✅ Yes' if ctx.outperforming_sector else '❌ No'}")
                print(f"Sector Momentum Score: {ctx.sector_momentum:.0f}/100")
                print(f"\n📊 Recommendation    : {ctx.recommendation}")
            except Exception as exc:  # noqa: BLE001
                print(f"Error computing sector context for {ticker_cli}: {exc}")
                traceback.print_exc()

        print("\n" + "=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal error in sector analysis CLI: {exc}")
        traceback.print_exc()