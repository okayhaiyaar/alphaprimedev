"""
============================================================
ALPHA-PRIME v2.0 - Regime Detector (Strategy Module)
============================================================

Classifies current market regime to adapt trading strategy:

Regime Types:
1. BULL (Strong Uptrend):
   - SMA50 > SMA200 (Golden Cross)  [web:206][web:212]
   - ADX > 25 (strong trend)  [web:207][web:218]
   - VIX < 20 (low fear)  [web:210]
   - Higher highs and higher lows
   - Strategy: Long bias, full position sizing

2. BEAR (Strong Downtrend):
   - SMA50 < SMA200 (Death Cross)  [web:206][web:212]
   - ADX > 25 (strong downtrend)
   - VIX elevated
   - Lower highs and lower lows
   - Strategy: Avoid longs, prefer shorts/cash

3. SIDEWAYS (Range-bound):
   - SMA50 ≈ SMA200 (flat)
   - ADX < 25 (weak trend)  [web:207][web:211]
   - VIX moderate
   - Choppy price action
   - Strategy: Reduce frequency, wait for breakout

4. HIGH_VOL (Extreme Volatility):
   - VIX > 30 (fear/panic)  [web:210]
   - Large daily swings (> 2%)
   - Unpredictable direction
   - Strategy: Reduce size or pause trading

5. UNKNOWN:
   - Insufficient data
   - Conflicting signals

Why It Matters:
- Trend-following strategies underperform in sideways markets.
- Regime filters help reduce whipsaws and improve risk-adjusted returns. [web:211][web:212]
- Volatility-aware filters using VIX and ATR stabilize performance. [web:210][web:211][web:212]

Usage:
    from strategy.regime_detector import get_current_regime

    regime = get_current_regime(benchmark="SPY")
    print(f"Current regime: {regime}")

    # Pass to Oracle
    decision = consult_oracle(ticker, intel, technicals, regime=regime)

Integration:
- Called by scheduler before analyzing tickers
- Passed to brain.py for context-aware decisions
- Displayed on dashboard
- Logged for performance analysis
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from diskcache import Cache

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()
cache = Cache(f"{settings.cache_dir}/regime")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & ENUMS
# ──────────────────────────────────────────────────────────


class RegimeType(str, Enum):
    """Market regime types."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeIndicators:
    """Technical indicators used for regime detection."""
    sma_50: float
    sma_200: float
    sma_ratio: float  # SMA50 / SMA200
    adx: float
    rsi: float
    vix: Optional[float]
    atr_percentile: float
    price_vs_sma50: float  # % above/below SMA50
    price_vs_sma200: float  # % above/below SMA200
    daily_change_pct: float
    daily_range_pct: float  # (High - Low) / Close * 100


@dataclass
class RegimeCharacteristics:
    """Characteristics and recommendations for each regime."""
    regime: RegimeType
    description: str
    confidence: float  # 0-100
    trade_bias: str  # LONG | SHORT | NEUTRAL | AVOID
    recommended_position_size_multiplier: float  # 0-1.5
    recommended_confidence_threshold: int  # Min Oracle confidence
    indicators: RegimeIndicators
    detected_at_utc: str


@dataclass
class RegimeTransition:
    """Detected regime change."""
    from_regime: RegimeType
    to_regime: RegimeType
    transition_date: str
    confidence: float


# ──────────────────────────────────────────────────────────
# TECHNICAL INDICATOR CALCULATION
# ──────────────────────────────────────────────────────────


def fetch_vix() -> Optional[float]:
    """
    Fetch current VIX (CBOE Volatility Index).

    Returns:
        Latest VIX close value as float, or None if unavailable.
    """
    try:
        vix_ticker = yf.Ticker("^VIX")
        hist = vix_ticker.history(period="1d")
        if hist.empty or "Close" not in hist.columns:
            return None
        vix_value = float(hist["Close"].iloc[-1])
        return vix_value
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not fetch VIX: %s", exc)
        return None


def calculate_regime_indicators(
    df: pd.DataFrame,
    benchmark: str = "SPY",
) -> RegimeIndicators:
    """
    Calculate technical indicators required for regime detection.

    Indicators:
        - SMA(50), SMA(200): trend direction.
        - ADX(14): trend strength.
        - RSI(14): momentum.
        - ATR(14) percentile: volatility regime. [web:211][web:212]
        - VIX: market-wide volatility regime. [web:210]
        - Daily change and range: short-term volatility.

    Args:
        df: OHLCV DataFrame with columns ["Open","High","Low","Close","Volume"].
        benchmark: Ticker symbol (for logging only).

    Returns:
        RegimeIndicators instance.

    Raises:
        ValueError: if insufficient history (< 200 bars).
    """
    if df.empty or len(df) < 200:
        raise ValueError("Insufficient data for regime detection (need >= 200 bars).")

    df = df.copy()

    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=200).mean()

    df.ta.adx(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)

    atr_col = [c for c in df.columns if c.upper().startswith("ATR")][-1]
    atr_values = df[atr_col].dropna()
    if atr_values.empty:
        atr_percentile = 0.0
        current_atr = 0.0
    else:
        current_atr = float(atr_values.iloc[-1])
        atr_percentile = float((atr_values < current_atr).sum() / len(atr_values) * 100.0)

    last = df.iloc[-1]

    sma_50 = float(last.get("SMA_50", np.nan))
    sma_200 = float(last.get("SMA_200", np.nan))
    sma_ratio = sma_50 / sma_200 if sma_200 and sma_200 != 0 else 1.0

    adx_col = [c for c in df.columns if c.upper().startswith("ADX")][-1]
    adx_val = float(last.get(adx_col, 0.0))

    rsi_col = [c for c in df.columns if c.upper().startswith("RSI")][-1]
    rsi_val = float(last.get(rsi_col, 50.0))

    close = float(last["Close"])
    price_vs_sma50 = (close - sma_50) / sma_50 * 100.0 if sma_50 else 0.0
    price_vs_sma200 = (close - sma_200) / sma_200 * 100.0 if sma_200 else 0.0

    if len(df) >= 2:
        prev_close = float(df.iloc[-2]["Close"])
        daily_change_pct = (close - prev_close) / prev_close * 100.0 if prev_close else 0.0
    else:
        daily_change_pct = 0.0

    high = float(last["High"])
    low = float(last["Low"])
    daily_range_pct = (high - low) / close * 100.0 if close else 0.0

    vix_val = fetch_vix()

    indicators = RegimeIndicators(
        sma_50=sma_50,
        sma_200=sma_200,
        sma_ratio=sma_ratio,
        adx=adx_val,
        rsi=rsi_val,
        vix=vix_val,
        atr_percentile=atr_percentile,
        price_vs_sma50=price_vs_sma50,
        price_vs_sma200=price_vs_sma200,
        daily_change_pct=daily_change_pct,
        daily_range_pct=daily_range_pct,
    )

    logger.debug(
        "Regime indicators for %s → SMA_ratio=%.3f, ADX=%.1f, RSI=%.1f, "
        "VIX=%s, ATR_pct=%.0f, Δ=%.2f%%, range=%.2f%%.",
        benchmark,
        indicators.sma_ratio,
        indicators.adx,
        indicators.rsi,
        f"{indicators.vix:.2f}" if indicators.vix is not None else "N/A",
        indicators.atr_percentile,
        indicators.daily_change_pct,
        indicators.daily_range_pct,
    )

    return indicators


# ──────────────────────────────────────────────────────────
# REGIME CLASSIFICATION LOGIC
# ──────────────────────────────────────────────────────────


def detect_regime(
    price_data: pd.DataFrame,
    benchmark: str = "SPY",
) -> RegimeType:
    """
    Detect current market regime from OHLCV data.

    Simple decision tree, combining:
        1. Volatility regime (VIX, daily change).
        2. Trend strength (ADX).
        3. Trend direction (SMA50 vs SMA200). [web:210][web:211][web:212]

    Args:
        price_data: OHLCV DataFrame.
        benchmark: Benchmark ticker for logging.

    Returns:
        RegimeType value.
    """
    try:
        indicators = calculate_regime_indicators(price_data, benchmark)

        if indicators.vix is not None and indicators.vix > 30.0:
            logger.info("HIGH_VOL regime detected (VIX=%.1f).", indicators.vix)
            return RegimeType.HIGH_VOL

        if abs(indicators.daily_change_pct) > 3.0 or indicators.daily_range_pct > 4.0:
            logger.info(
                "HIGH_VOL regime detected (Δ=%.2f%%, range=%.2f%%).",
                indicators.daily_change_pct,
                indicators.daily_range_pct,
            )
            return RegimeType.HIGH_VOL

        sma_ratio = indicators.sma_ratio
        adx = indicators.adx

        if adx > 25.0:
            if sma_ratio > 1.01:
                logger.info(
                    "BULL regime detected (SMA_ratio=%.3f, ADX=%.1f).", sma_ratio, adx
                )
                return RegimeType.BULL
            if sma_ratio < 0.99:
                logger.info(
                    "BEAR regime detected (SMA_ratio=%.3f, ADX=%.1f).", sma_ratio, adx
                )
                return RegimeType.BEAR

            logger.info(
                "SIDEWAYS regime (transitional strong trend; SMA50~SMA200, ADX=%.1f).",
                adx,
            )
            return RegimeType.SIDEWAYS

        logger.info("SIDEWAYS regime detected (ADX=%.1f < 25).", adx)
        return RegimeType.SIDEWAYS

    except Exception as exc:  # noqa: BLE001
        logger.error("Error detecting regime for %s: %s", benchmark, exc, exc_info=True)
        return RegimeType.UNKNOWN


# ──────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ──────────────────────────────────────────────────────────


def get_regime_confidence(
    regime: RegimeType,
    indicators: RegimeIndicators,
) -> float:
    """
    Compute a heuristic confidence score for a detected regime.

    Score components:
        - SMA separation and ADX strength.
        - VIX bucket.
        - Volatility percentile.

    Args:
        regime: Detected regime type.
        indicators: RegimeIndicators instance.

    Returns:
        Confidence in [0, 100].
    """
    confidence = 50.0

    if regime == RegimeType.BULL:
        if indicators.sma_ratio > 1.02:
            confidence += 15.0
        if indicators.adx > 30.0:
            confidence += 15.0
        if indicators.price_vs_sma200 > 5.0:
            confidence += 10.0
        if indicators.vix is not None and indicators.vix < 15.0:
            confidence += 10.0

    elif regime == RegimeType.BEAR:
        if indicators.sma_ratio < 0.98:
            confidence += 15.0
        if indicators.adx > 30.0:
            confidence += 15.0
        if indicators.price_vs_sma200 < -5.0:
            confidence += 10.0
        if indicators.vix is not None and indicators.vix > 25.0:
            confidence += 10.0

    elif regime == RegimeType.SIDEWAYS:
        if indicators.adx < 20.0:
            confidence += 20.0
        if 0.99 < indicators.sma_ratio < 1.01:
            confidence += 15.0
        if indicators.vix is not None and 15.0 < indicators.vix < 25.0:
            confidence += 10.0

    elif regime == RegimeType.HIGH_VOL:
        if indicators.vix is not None and indicators.vix > 35.0:
            confidence += 20.0
        if abs(indicators.daily_change_pct) > 4.0:
            confidence += 15.0
        if indicators.atr_percentile > 90.0:
            confidence += 15.0

    confidence = float(max(0.0, min(100.0, confidence)))
    return confidence


# ──────────────────────────────────────────────────────────
# REGIME CHARACTERISTICS & RECOMMENDATIONS
# ──────────────────────────────────────────────────────────


def get_regime_characteristics(
    regime: RegimeType,
    indicators: RegimeIndicators,
) -> RegimeCharacteristics:
    """
    Map a regime and its indicators to qualitative trading guidance.

    Args:
        regime: Detected regime type.
        indicators: RegimeIndicators instance.

    Returns:
        RegimeCharacteristics containing bias and risk parameters.
    """
    confidence = get_regime_confidence(regime, indicators)

    characteristics_map: Dict[RegimeType, Dict[str, object]] = {
        RegimeType.BULL: {
            "description": "Strong uptrend with constructive volatility.",
            "trade_bias": "LONG",
            "position_multiplier": 1.0,
            "confidence_threshold": 70,
        },
        RegimeType.BEAR: {
            "description": "Persistent downtrend with negative momentum.",
            "trade_bias": "SHORT",
            "position_multiplier": 0.5,
            "confidence_threshold": 80,
        },
        RegimeType.SIDEWAYS: {
            "description": "Range-bound, choppy conditions.",
            "trade_bias": "NEUTRAL",
            "position_multiplier": 0.7,
            "confidence_threshold": 80,
        },
        RegimeType.HIGH_VOL: {
            "description": "Extreme volatility and unstable direction.",
            "trade_bias": "AVOID",
            "position_multiplier": 0.3,
            "confidence_threshold": 90,
        },
        RegimeType.UNKNOWN: {
            "description": "Insufficient data or conflicting signals.",
            "trade_bias": "NEUTRAL",
            "position_multiplier": 0.5,
            "confidence_threshold": 85,
        },
    }

    cfg = characteristics_map.get(regime, characteristics_map[RegimeType.UNKNOWN])

    return RegimeCharacteristics(
        regime=regime,
        description=str(cfg["description"]),
        confidence=confidence,
        trade_bias=str(cfg["trade_bias"]),
        recommended_position_size_multiplier=float(cfg["position_multiplier"]),
        recommended_confidence_threshold=int(cfg["confidence_threshold"]),
        indicators=indicators,
        detected_at_utc=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────────────────────
# TRADING RULES BY REGIME
# ──────────────────────────────────────────────────────────


def should_trade_in_regime(
    regime: str,
    action: str,
    confidence: int,
) -> Tuple[bool, str]:
    """
    Decide whether a trade should be allowed given regime and signal strength.

    Policy:
        - HIGH_VOL: Only very high confidence trades.
        - BEAR: Restrict longs unless high confidence.
        - SIDEWAYS: Require higher confidence for any trade.
        - BULL: Favor longs; shorts need slightly higher confidence.
        - UNKNOWN: Conservative threshold.

    Args:
        regime: Regime string ("BULL","BEAR","SIDEWAYS","HIGH_VOL","UNKNOWN").
        action: Trade action ("BUY" or "SELL").
        confidence: Oracle confidence percentage.

    Returns:
        (allowed, message) pair.
    """
    try:
        regime_type = RegimeType(regime)
    except ValueError:
        regime_type = RegimeType.UNKNOWN

    action = action.upper()

    if regime_type == RegimeType.HIGH_VOL:
        if confidence < 90:
            return False, (
                f"HIGH_VOL regime: confidence {confidence}% < 90% threshold; "
                "market too volatile for this signal."
            )
        logger.warning("Trading in HIGH_VOL regime with confidence=%d%%.", confidence)
        return True, "HIGH_VOL regime: high-confidence signal allowed."

    if regime_type == RegimeType.BEAR:
        if action == "BUY" and confidence < 85:
            return False, (
                f"BEAR regime: BUY confidence {confidence}% < 85% threshold; "
                "avoid low-conviction longs against the trend."
            )
        return True, f"BEAR regime: {action} allowed (confidence={confidence}%)."

    if regime_type == RegimeType.SIDEWAYS:
        if confidence < 80:
            return False, (
                f"SIDEWAYS regime: confidence {confidence}% < 80% threshold; "
                "choppy tape, waiting for clearer setups."
            )
        return True, f"SIDEWAYS regime: high-confidence {action} allowed."

    if regime_type == RegimeType.BULL:
        if action == "BUY" and confidence >= 70:
            return True, "BULL regime: BUY signal aligned with trend."
        if action == "SELL" and confidence >= 75:
            return True, "BULL regime: SELL allowed as counter-trend with high conviction."
        return False, f"BULL regime: confidence {confidence}% below required threshold."

    if confidence < 80:
        return False, f"UNKNOWN regime: confidence {confidence}% < 80% threshold."
    return True, "UNKNOWN regime: proceeding cautiously with signal."


# ──────────────────────────────────────────────────────────
# MAIN REGIME DETECTION FUNCTION
# ──────────────────────────────────────────────────────────


def get_current_regime(
    benchmark: str = "SPY",
    period: str = "1y",
    use_cache: bool = True,
) -> str:
    """
    Fetch benchmark data and classify the current market regime.

    Uses a 1-year daily window by default to compute SMAs and volatility.
    Results are cached for 1 hour per benchmark.

    Args:
        benchmark: Benchmark ticker symbol (e.g. "SPY").
        period: Historical data period (e.g. "1y").
        use_cache: If True, uses 1-hour cache.

    Returns:
        Regime string: "BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", or "UNKNOWN".
    """
    cache_key = f"regime_{benchmark}_{period}"

    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            logger.debug("Regime cache hit for %s: %s.", benchmark, cached)
            return str(cached)

    logger.info("Detecting market regime for %s...", benchmark)

    try:
        data = yf.download(
            benchmark,
            period=period,
            interval="1d",
            progress=False,
        )
        if data.empty or len(data) < 200:
            logger.warning("Insufficient data for regime detection on %s.", benchmark)
            return RegimeType.UNKNOWN.value

        regime = detect_regime(data, benchmark)
        indicators = calculate_regime_indicators(data, benchmark)
        characteristics = get_regime_characteristics(regime, indicators)

        logger.info(
            "Regime for %s: %s (confidence=%.0f%%, bias=%s).",
            benchmark,
            characteristics.regime.value,
            characteristics.confidence,
            characteristics.trade_bias,
        )

        cache.set(cache_key, regime.value, expire=3600)
        return regime.value
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error in get_current_regime for %s: %s", benchmark, exc, exc_info=True
        )
        return RegimeType.UNKNOWN.value


# ──────────────────────────────────────────────────────────
# REGIME TRANSITION DETECTION
# ──────────────────────────────────────────────────────────


def detect_regime_transition(
    historical_regimes: List[Tuple[str, str]],
    current_regime: str,
) -> Optional[RegimeTransition]:
    """
    Detect whether the regime has changed compared to the last recorded value.

    Args:
        historical_regimes: List of (date_iso, regime_str) tuples sorted by date.
        current_regime: Newly detected regime string.

    Returns:
        RegimeTransition if a change is detected, otherwise None.
    """
    if not historical_regimes:
        return None

    last_date, last_regime = historical_regimes[-1]
    if last_regime == current_regime:
        return None

    try:
        from_regime = RegimeType(last_regime)
    except ValueError:
        from_regime = RegimeType.UNKNOWN

    try:
        to_regime = RegimeType(current_regime)
    except ValueError:
        to_regime = RegimeType.UNKNOWN

    transition = RegimeTransition(
        from_regime=from_regime,
        to_regime=to_regime,
        transition_date=datetime.now(timezone.utc).isoformat(),
        confidence=80.0,
    )
    logger.info(
        "Regime transition detected: %s → %s.",
        transition.from_regime.value,
        transition.to_regime.value,
    )
    return transition


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Regime Detector - Test Tool")
    print("=" * 70 + "\n")

    if len(sys.argv) > 1:
        benchmark_cli = sys.argv[1].upper()
    else:
        benchmark_cli = "SPY"

    print(f"Analyzing {benchmark_cli}...\n")

    regime_str = get_current_regime(benchmark=benchmark_cli, use_cache=False)
    print(f"Detected Regime: {regime_str}\n")

    try:
        data_cli = yf.download(
            benchmark_cli, period="1y", interval="1d", progress=False
        )
        indicators_cli = calculate_regime_indicators(data_cli, benchmark_cli)
        characteristics_cli = get_regime_characteristics(
            RegimeType(regime_str), indicators_cli
        )

        print(f"🎯 REGIME: {characteristics_cli.regime.value}")
        print(f"Description              : {characteristics_cli.description}")
        print(f"Confidence               : {characteristics_cli.confidence:.1f}%")
        print(f"Trade Bias               : {characteristics_cli.trade_bias}")
        print(
            "Position Size Multiplier : "
            f"{characteristics_cli.recommended_position_size_multiplier:.1f}x"
        )
        print(
            "Min Oracle Confidence    : "
            f"{characteristics_cli.recommended_confidence_threshold}%"
        )

        print("\n" + "-" * 70)
        print("Technical Indicators:")
        print(f"  SMA50          : {indicators_cli.sma_50:.2f}")
        print(f"  SMA200         : {indicators_cli.sma_200:.2f}")
        print(f"  SMA Ratio      : {indicators_cli.sma_ratio:.3f}")
        print(f"  ADX            : {indicators_cli.adx:.1f}")
        print(f"  RSI            : {indicators_cli.rsi:.1f}")
        print(
            f"  VIX            : "
            f"{indicators_cli.vix:.2f}" if indicators_cli.vix is not None else "  VIX            : N/A"
        )
        print(f"  ATR Percentile : {indicators_cli.atr_percentile:.0f}%")
        print(f"  Price vs SMA50 : {indicators_cli.price_vs_sma50:+.2f}%")
        print(f"  Price vs SMA200: {indicators_cli.price_vs_sma200:+.2f}%")
        print(f"  Daily Δ        : {indicators_cli.daily_change_pct:+.2f}%")
        print(f"  Daily Range    : {indicators_cli.daily_range_pct:.2f}%")

        print("\n" + "-" * 70)
        print("Trading Rules Smoke Test:")

        for action in ["BUY", "SELL"]:
            for conf in [60, 75, 85, 92]:
                allowed, reason = should_trade_in_regime(
                    regime_str, action, conf
                )
                status = "✅" if allowed else "❌"
                print(
                    f"  {status} {action} @ {conf}% confidence → {reason}"
                )

    except Exception as exc:  # noqa: BLE001
        print(f"Error running regime detector CLI: {exc}")

    print("\n" + "=" * 70 + "\n")
