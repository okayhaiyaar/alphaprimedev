"""
============================================================
ALPHA-PRIME v2.0 - Multi-Timeframe Analysis
============================================================

Analyzes multiple timeframes to find signal confluence:

Why Multi-Timeframe Analysis?
- Single timeframe = tunnel vision
- Multiple timeframes = context + confirmation
- Reduces false signals and whipsaws
- Improves win rate through confluence

Timeframe Hierarchy:
1. 1d (Daily): Primary trend, overall bias
   - "Zoom out to see the forest"
   - Sets directional bias for lower timeframes

2. 4h (4-hour): Swing trend, confirmation
   - "See the trees"
   - Confirms daily trend, identifies swings

3. 1h (1-hour): Entry timing, intraday momentum
   - "See the branches"
   - Precise entry/exit timing

Trading Rules:
1. PERFECT CONFLUENCE (100%):
   - All timeframes aligned (trend, RSI, MACD)
   - Highest probability setup
   - Full position size

2. STRONG CONFLUENCE (75%+):
   - 2 out of 3 timeframes aligned
   - 1 timeframe neutral (not opposing)
   - Good probability, standard position

3. MODERATE CONFLUENCE (50-75%):
   - Mixed signals
   - 1 timeframe opposing
   - Reduced position size or wait

4. WEAK CONFLUENCE (<50%):
   - Conflicting signals
   - Avoid trade

Example:
    Daily = Uptrend (MACD bullish, RSI 60)
    4h   = Uptrend (MACD bullish, RSI 55)
    1h   = Pullback (MACD bearish, RSI 40 - oversold)

    → Confluence: 75% (daily/4h agree, 1h shows entry opportunity)
    → Action: BUY on 1h oversold bounce

Usage:
    from strategy.multi_timeframe import analyze_multi_timeframe

    analysis = analyze_multi_timeframe("AAPL", ["1h", "4h", "1d"])

    if analysis.confluence.confluence_score >= 75:
        print("Strong confluence - proceed with trade")
    else:
        print("Weak confluence - skip trade")

Integration:
- Called by brain.py before Oracle decision
- Enhances signal quality
- Reduces whipsaws
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401  (used indirectly via calculate_hard_technicals)
import yfinance as yf  # noqa: F401  (may be used by get_market_data implementation)
from diskcache import Cache

from config import get_logger, get_settings
from data_engine import get_market_data, calculate_hard_technicals

logger = get_logger(__name__)
settings = get_settings()
cache = Cache(f"{settings.cache_dir}/multi_timeframe")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & ENUMS
# ──────────────────────────────────────────────────────────


class TrendDirection(str, Enum):
    """Trend direction enum."""

    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class SignalStrength(str, Enum):
    """Signal strength levels."""

    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class TimeframeSignals:
    """Signals for a single timeframe."""
    timeframe: str
    trend: TrendDirection
    rsi: float
    rsi_signal: str  # OVERSOLD | NEUTRAL | OVERBOUGHT
    macd_signal: str  # BULLISH | BEARISH | NEUTRAL
    macd_histogram: float
    ema_alignment: bool  # True if EMA9 > EMA20 > EMA50 (bullish stack)
    volume_trend: str  # INCREASING | DECREASING | FLAT
    price_vs_ema20: float  # % above/below EMA20
    signal_strength: SignalStrength
    last_price: float


@dataclass
class Divergence:
    """Detected divergence between timeframes."""
    timeframe_1: str
    timeframe_2: str
    divergence_type: str  # TREND | RSI | MACD | VOLUME
    description: str
    severity: str  # MAJOR | MINOR


@dataclass
class ConfluenceResult:
    """Result of confluence analysis."""
    confluence_score: float  # 0-100
    aligned_timeframes: List[str]
    conflicting_timeframes: List[str]
    recommended_action: str  # BUY | SELL | WAIT
    confidence_boost: int  # +0 to +20 confidence points
    rationale: List[str]
    divergences: List[Divergence]


@dataclass
class MultiTimeframeAnalysis:
    """Complete multi-timeframe analysis across 1h / 4h / 1d."""
    ticker: str
    timeframe_signals: Dict[str, TimeframeSignals]
    confluence: ConfluenceResult
    primary_trend: TrendDirection  # From highest timeframe (1d)
    strongest_timeframe: str
    trade_recommendation: str  # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    analyzed_at_utc: str


# ──────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────


_TIMEFRAME_WEIGHT = {"1h": 1, "4h": 4, "1d": 24}


def _timeframe_key(tf: str) -> int:
    """Map timeframe string to numeric weight (for sorting)."""
    return _TIMEFRAME_WEIGHT.get(tf, 999)


def _signal_strength_score(strength: SignalStrength) -> int:
    """Map SignalStrength to numeric score for comparison."""
    mapping = {
        SignalStrength.STRONG_BULLISH: 5,
        SignalStrength.BULLISH: 4,
        SignalStrength.NEUTRAL: 3,
        SignalStrength.BEARISH: 2,
        SignalStrength.STRONG_BEARISH: 1,
    }
    return mapping.get(strength, 3)


# ──────────────────────────────────────────────────────────
# SINGLE TIMEFRAME ANALYSIS
# ──────────────────────────────────────────────────────────


def analyze_single_timeframe(
    ticker: str,
    timeframe: str,
    period: str = "3mo",
) -> TimeframeSignals:
    """
    Analyze a single timeframe and derive a compact signal summary.

    Pulls data via `get_market_data`, computes technicals via
    `calculate_hard_technicals`, and distills them into a TimeframeSignals
    object suitable for multi-timeframe confluence logic.

    Args:
        ticker: Symbol (e.g. "AAPL").
        timeframe: Timeframe string accepted by get_market_data (e.g. "1h","4h","1d").
        period: Historical period (default "3mo").

    Returns:
        TimeframeSignals instance.

    Raises:
        ValueError: if there is insufficient data.
    """
    logger.debug("Analyzing %s on %s timeframe...", ticker, timeframe)

    df = get_market_data(ticker, period=period, interval=timeframe)
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker} on {timeframe} (len={len(df)}).")

    technicals = calculate_hard_technicals(df, ticker, timeframe)

    trend_raw = technicals["trend"]["trend"]
    try:
        trend = TrendDirection(trend_raw)
    except ValueError:
        trend = TrendDirection.UNKNOWN

    rsi = float(technicals["momentum"].get("rsi", 50.0))
    rsi_signal = str(technicals["momentum"].get("rsi_signal", "NEUTRAL"))

    macd = technicals.get("macd", {})
    macd_signal = str(macd.get("signal", "NEUTRAL"))
    macd_histogram = float(macd.get("histogram", 0.0))

    ema_9 = float(technicals["trend"].get("ema_9", 0.0))
    ema_20 = float(technicals["trend"].get("ema_20", 0.0))
    ema_50 = float(technicals["trend"].get("ema_50", 0.0))
    ema_alignment = bool(ema_9 > ema_20 > ema_50) if all([ema_9, ema_20, ema_50]) else False

    volume_indicators = technicals.get("volume", {})
    volume_trend = str(volume_indicators.get("obv_trend", "FLAT"))

    last_price = float(technicals["price_action"].get("last_price", df["Close"].iloc[-1]))
    price_vs_ema20 = (last_price - ema_20) / ema_20 * 100.0 if ema_20 else 0.0

    signal_strength = determine_signal_strength(
        trend=trend,
        rsi=rsi,
        macd_signal=macd_signal,
        ema_alignment=ema_alignment,
    )

    return TimeframeSignals(
        timeframe=timeframe,
        trend=trend,
        rsi=rsi,
        rsi_signal=rsi_signal,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        ema_alignment=ema_alignment,
        volume_trend=volume_trend,
        price_vs_ema20=price_vs_ema20,
        signal_strength=signal_strength,
        last_price=last_price,
    )


def determine_signal_strength(
    trend: TrendDirection,
    rsi: float,
    macd_signal: str,
    ema_alignment: bool,
) -> SignalStrength:
    """
    Determine aggregated signal strength for a single timeframe.

    Scoring:
        - Trend UP/DOWN counts double.
        - MACD and EMA alignment contribute directional bias.
        - RSI extremes reinforce trend bias.

    Args:
        trend: TrendDirection for timeframe.
        rsi: RSI value.
        macd_signal: "BULLISH" | "BEARISH" | "NEUTRAL".
        ema_alignment: True if bullish EMA stack (9 > 20 > 50).

    Returns:
        SignalStrength enum.
    """
    bullish = 0
    bearish = 0

    if trend == TrendDirection.UP:
        bullish += 2
    elif trend == TrendDirection.DOWN:
        bearish += 2

    if macd_signal.upper() == "BULLISH":
        bullish += 1
    elif macd_signal.upper() == "BEARISH":
        bearish += 1

    if ema_alignment:
        bullish += 1
    elif not ema_alignment and trend == TrendDirection.DOWN:
        bearish += 1

    if rsi > 60.0:
        bullish += 1
    elif rsi < 40.0:
        bearish += 1

    if bullish >= 4:
        return SignalStrength.STRONG_BULLISH
    if bullish >= 2:
        return SignalStrength.BULLISH
    if bearish >= 4:
        return SignalStrength.STRONG_BEARISH
    if bearish >= 2:
        return SignalStrength.BEARISH
    return SignalStrength.NEUTRAL


# ──────────────────────────────────────────────────────────
# DIVERGENCE DETECTION
# ──────────────────────────────────────────────────────────


def detect_divergences(
    signals: Dict[str, TimeframeSignals],
) -> List[Divergence]:
    """
    Detect divergences between adjacent timeframes.

    Types:
        - TREND: One timeframe UP, another DOWN.
        - RSI: One overbought, other oversold.
        - MACD: Bullish vs bearish.

    Args:
        signals: Mapping timeframe → TimeframeSignals.

    Returns:
        List of Divergence objects.
    """
    divergences: List[Divergence] = []

    tfs = sorted(signals.keys(), key=_timeframe_key)
    for i in range(len(tfs) - 1):
        tf1 = tfs[i]
        tf2 = tfs[i + 1]
        s1 = signals[tf1]
        s2 = signals[tf2]

        if (
            s1.trend == TrendDirection.UP
            and s2.trend == TrendDirection.DOWN
        ) or (
            s1.trend == TrendDirection.DOWN
            and s2.trend == TrendDirection.UP
        ):
            divergences.append(
                Divergence(
                    timeframe_1=tf1,
                    timeframe_2=tf2,
                    divergence_type="TREND",
                    description=(
                        f"{tf1} trend {s1.trend.value}, "
                        f"{tf2} trend {s2.trend.value}"
                    ),
                    severity="MAJOR",
                )
            )

        if (s1.rsi < 30.0 and s2.rsi > 70.0) or (s1.rsi > 70.0 and s2.rsi < 30.0):
            divergences.append(
                Divergence(
                    timeframe_1=tf1,
                    timeframe_2=tf2,
                    divergence_type="RSI",
                    description=(
                        f"{tf1} RSI={s1.rsi:.1f}, "
                        f"{tf2} RSI={s2.rsi:.1f}"
                    ),
                    severity="MAJOR",
                )
            )

        if (
            s1.macd_signal.upper() == "BULLISH"
            and s2.macd_signal.upper() == "BEARISH"
        ) or (
            s1.macd_signal.upper() == "BEARISH"
            and s2.macd_signal.upper() == "BULLISH"
        ):
            divergences.append(
                Divergence(
                    timeframe_1=tf1,
                    timeframe_2=tf2,
                    divergence_type="MACD",
                    description=(
                        f"{tf1} MACD={s1.macd_signal}, "
                        f"{tf2} MACD={s2.macd_signal}"
                    ),
                    severity="MINOR",
                )
            )

    return divergences


# ──────────────────────────────────────────────────────────
# CONFLUENCE ANALYSIS
# ──────────────────────────────────────────────────────────


def check_timeframe_confluence(
    signals: Dict[str, TimeframeSignals],
) -> ConfluenceResult:
    """
    Evaluate confluence across multiple timeframes.

    Components:
        - Trend alignment (max 40 points).
        - RSI agreement (max 20 points).
        - MACD confluence (max 20 points).
        - Volume confirmation (max 20 points).

    Args:
        signals: Mapping timeframe string → TimeframeSignals.

    Returns:
        ConfluenceResult summarizing score, action, and rationale.
    """
    if len(signals) < 2:
        raise ValueError("Need at least 2 timeframes for confluence analysis.")

    total_tfs = len(signals)
    trends = {tf: sig.trend for tf, sig in signals.items()}

    up_trends = sum(1 for t in trends.values() if t == TrendDirection.UP)
    down_trends = sum(1 for t in trends.values() if t == TrendDirection.DOWN)
    sideways_trends = sum(1 for t in trends.values() if t == TrendDirection.SIDEWAYS)

    confluence_score = 0.0
    aligned_tfs: List[str] = []
    conflicting_tfs: List[str] = []
    rationale: List[str] = []

    if up_trends == total_tfs:
        confluence_score += 40.0
        aligned_tfs = list(signals.keys())
        rationale.append("✅ All timeframes in uptrend.")
    elif down_trends == total_tfs:
        confluence_score += 40.0
        aligned_tfs = list(signals.keys())
        rationale.append("✅ All timeframes in downtrend.")
    elif up_trends >= total_tfs * 0.66:
        confluence_score += 30.0
        aligned_tfs = [tf for tf, t in trends.items() if t == TrendDirection.UP]
        rationale.append(f"✓ {up_trends}/{total_tfs} timeframes in uptrend.")
    elif down_trends >= total_tfs * 0.66:
        confluence_score += 30.0
        aligned_tfs = [tf for tf, t in trends.items() if t == TrendDirection.DOWN]
        rationale.append(f"✓ {down_trends}/{total_tfs} timeframes in downtrend.")
    else:
        conflicting_tfs = list(signals.keys())
        rationale.append(
            f"⚠️ Mixed trends: {up_trends} up, {down_trends} down, {sideways_trends} sideways."
        )

    rsi_vals = [sig.rsi for sig in signals.values()]
    rsi_oversold = sum(1 for r in rsi_vals if r < 30.0)
    rsi_overbought = sum(1 for r in rsi_vals if r > 70.0)

    if rsi_oversold == total_tfs:
        confluence_score += 20.0
        rationale.append("✅ All timeframes oversold (RSI < 30).")
    elif rsi_overbought == total_tfs:
        confluence_score += 20.0
        rationale.append("✅ All timeframes overbought (RSI > 70).")
    elif rsi_oversold >= total_tfs * 0.66:
        confluence_score += 15.0
        rationale.append(f"✓ {rsi_oversold}/{total_tfs} timeframes oversold.")
    elif rsi_overbought >= total_tfs * 0.66:
        confluence_score += 15.0
        rationale.append(f"✓ {rsi_overbought}/{total_tfs} timeframes overbought.")

    macd_bullish = sum(
        1 for sig in signals.values() if sig.macd_signal.upper() == "BULLISH"
    )
    macd_bearish = sum(
        1 for sig in signals.values() if sig.macd_signal.upper() == "BEARISH"
    )

    if macd_bullish == total_tfs:
        confluence_score += 20.0
        rationale.append("✅ All timeframes MACD bullish.")
    elif macd_bearish == total_tfs:
        confluence_score += 20.0
        rationale.append("✅ All timeframes MACD bearish.")
    elif macd_bullish >= total_tfs * 0.66:
        confluence_score += 15.0
        rationale.append(f"✓ {macd_bullish}/{total_tfs} timeframes MACD bullish.")
    elif macd_bearish >= total_tfs * 0.66:
        confluence_score += 15.0
        rationale.append(f"✓ {macd_bearish}/{total_tfs} timeframes MACD bearish.")

    volume_increasing = sum(
        1 for sig in signals.values() if sig.volume_trend.upper() == "INCREASING"
    )
    if volume_increasing >= total_tfs * 0.66:
        confluence_score += 20.0
        rationale.append(
            f"✅ Volume increasing across {volume_increasing}/{total_tfs} timeframes."
        )
    elif volume_increasing >= total_tfs * 0.5:
        confluence_score += 10.0
        rationale.append(
            f"✓ Volume supportive in {volume_increasing}/{total_tfs} timeframes."
        )

    if confluence_score >= 80.0 and up_trends >= total_tfs * 0.66:
        recommended_action = "BUY"
        confidence_boost = 20
    elif confluence_score >= 80.0 and down_trends >= total_tfs * 0.66:
        recommended_action = "SELL"
        confidence_boost = 20
    elif confluence_score >= 60.0:
        recommended_action = "BUY" if up_trends >= down_trends else "SELL"
        confidence_boost = 10
    else:
        recommended_action = "WAIT"
        confidence_boost = 0
        rationale.append("⚠️ Insufficient confluence – waiting for clearer setup.")

    divergences = detect_divergences(signals)

    return ConfluenceResult(
        confluence_score=float(confluence_score),
        aligned_timeframes=aligned_tfs,
        conflicting_timeframes=conflicting_tfs,
        recommended_action=recommended_action,
        confidence_boost=confidence_boost,
        rationale=rationale,
        divergences=divergences,
    )


def calculate_confluence_score(trend_alignment: Dict[str, TrendDirection]) -> float:
    """
    Convenience wrapper to score pure trend alignment only.

    Args:
        trend_alignment: Mapping timeframe → TrendDirection.

    Returns:
        Trend-only confluence score (0–40).
    """
    total = len(trend_alignment)
    if total == 0:
        return 0.0

    up = sum(1 for t in trend_alignment.values() if t == TrendDirection.UP)
    down = sum(1 for t in trend_alignment.values() if t == TrendDirection.DOWN)

    if up == total or down == total:
        return 40.0
    if up >= total * 0.66 or down >= total * 0.66:
        return 30.0
    return 10.0


def get_strongest_timeframe(
    analyses: Dict[str, TimeframeSignals],
) -> Tuple[str, TimeframeSignals]:
    """
    Identify the timeframe with the strongest directional signal.

    Args:
        analyses: Mapping timeframe → TimeframeSignals.

    Returns:
        (timeframe, TimeframeSignals) pair.
    """
    if not analyses:
        raise ValueError("No timeframe analyses provided.")

    return max(
        analyses.items(),
        key=lambda kv: _signal_strength_score(kv[1].signal_strength),
    )


def should_trade_with_confluence(
    confluence: float,
    min_threshold: float = 60.0,
) -> Tuple[bool, str]:
    """
    Decide whether confluence is sufficient to permit a trade.

    Args:
        confluence: Confluence score in [0, 100].
        min_threshold: Minimum acceptable score (default 60).

    Returns:
        (allowed, message) pair summarizing decision.
    """
    if confluence >= 80.0:
        return True, f"Excellent confluence ({confluence:.0f}% ≥ 80%)."
    if confluence >= min_threshold:
        return True, f"Good confluence ({confluence:.0f}% ≥ {min_threshold:.0f}%)."
    return False, f"Insufficient confluence ({confluence:.0f}% < {min_threshold:.0f}%)."


# ──────────────────────────────────────────────────────────
# MAIN MULTI-TIMEFRAME ANALYSIS
# ──────────────────────────────────────────────────────────


def analyze_multi_timeframe(
    ticker: str,
    timeframes: Optional[List[str]] = None,
) -> MultiTimeframeAnalysis:
    """
    Run full multi-timeframe analysis for a ticker.

    Default timeframes:
        - 1h (short-term entry timing),
        - 4h (swing confirmation),
        - 1d (primary trend).

    Args:
        ticker: Symbol to analyze.
        timeframes: Optional list of timeframe strings. Defaults to ["1h","4h","1d"].

    Returns:
        MultiTimeframeAnalysis object with signals, confluence, and recommendation.
    """
    if timeframes is None:
        timeframes = ["1h", "4h", "1d"]

    key = f"{ticker.upper()}_{','.join(sorted(timeframes, key=_timeframe_key))}"
    cache_key = f"mta_{key}"
    cached = cache.get(cache_key)
    if cached:
        logger.debug("Multi-timeframe cache hit for %s.", key)
        return cached

    logger.info("Starting multi-timeframe analysis for %s on %s.", ticker, timeframes)

    tf_signals: Dict[str, TimeframeSignals] = {}
    for tf in timeframes:
        try:
            sig = analyze_single_timeframe(ticker, tf)
            tf_signals[tf] = sig
            logger.debug(
                "%s %s → trend=%s, strength=%s, RSI=%.1f.",
                ticker,
                tf,
                sig.trend.value,
                sig.signal_strength.value,
                sig.rsi,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Error analyzing %s on timeframe %s: %s", ticker, tf, exc, exc_info=True
            )

    if len(tf_signals) < 2:
        raise ValueError(f"Insufficient timeframe data for {ticker} (got {len(tf_signals)}).")

    confluence = check_timeframe_confluence(tf_signals)

    longest_tf = max(tf_signals.keys(), key=_timeframe_key)
    primary_trend = tf_signals[longest_tf].trend

    strongest_tf_name, _strongest_sig = get_strongest_timeframe(tf_signals)

    if confluence.confluence_score >= 80.0:
        if confluence.recommended_action == "BUY":
            trade_recommendation = "STRONG_BUY"
        elif confluence.recommended_action == "SELL":
            trade_recommendation = "STRONG_SELL"
        else:
            trade_recommendation = "HOLD"
    elif confluence.confluence_score >= 60.0:
        if confluence.recommended_action == "BUY":
            trade_recommendation = "BUY"
        elif confluence.recommended_action == "SELL":
            trade_recommendation = "SELL"
        else:
            trade_recommendation = "HOLD"
    else:
        trade_recommendation = "HOLD"

    analysis = MultiTimeframeAnalysis(
        ticker=ticker.upper(),
        timeframe_signals=tf_signals,
        confluence=confluence,
        primary_trend=primary_trend,
        strongest_timeframe=strongest_tf_name,
        trade_recommendation=trade_recommendation,
        analyzed_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Multi-TF analysis %s → confluence=%.0f%%, recommendation=%s.",
        ticker,
        confluence.confluence_score,
        trade_recommendation,
    )

    cache.set(cache_key, analysis, expire=300)
    return analysis


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Multi-Timeframe Analysis - Test Tool")
    print("=" * 70 + "\n")

    if len(sys.argv) < 2:
        print("Usage: python multi_timeframe.py TICKER")
        print("Example: python multi_timeframe.py AAPL\n")
        sys.exit(1)

    ticker_cli = sys.argv[1].upper()
    print(f"Analyzing {ticker_cli} across 1h / 4h / 1d...\n")

    try:
        analysis_cli = analyze_multi_timeframe(ticker_cli)

        print("=" * 70)
        print(f"MULTI-TIMEFRAME ANALYSIS: {ticker_cli}")
        print("=" * 70 + "\n")

        print("TIMEFRAME BREAKDOWN")
        print("-" * 70)
        for tf, sig in sorted(
            analysis_cli.timeframe_signals.items(), key=lambda kv: _timeframe_key(kv[0])
        ):
            print(f"\n{tf} timeframe:")
            print(f"  Trend           : {sig.trend.value}")
            print(f"  Signal Strength : {sig.signal_strength.value}")
            print(f"  RSI             : {sig.rsi:.1f} ({sig.rsi_signal})")
            print(f"  MACD            : {sig.macd_signal}")
            print(f"  EMA Alignment   : {'✅' if sig.ema_alignment else '❌'}")
            print(f"  Volume Trend    : {sig.volume_trend}")
            print(f"  Price vs EMA20  : {sig.price_vs_ema20:+.2f}%")
            print(f"  Last Price      : {sig.last_price:.2f}")

        print("\n" + "=" * 70)
        print("CONFLUENCE ANALYSIS")
        print("-" * 70)
        print(f"Confluence Score  : {analysis_cli.confluence.confluence_score:.0f}%")
        aligned = analysis_cli.confluence.aligned_timeframes or ["None"]
        conflicting = analysis_cli.confluence.conflicting_timeframes or ["None"]
        print(f"Aligned TFs       : {', '.join(aligned)}")
        print(f"Conflicting TFs   : {', '.join(conflicting)}")
        print(f"Confidence Boost  : +{analysis_cli.confluence.confidence_boost}")

        print("\nRationale:")
        for line in analysis_cli.confluence.rationale:
            print(f"  {line}")

        if analysis_cli.confluence.divergences:
            print("\n⚠️  Divergences Detected:")
            for div in analysis_cli.confluence.divergences:
                print(
                    f"  - {div.severity}: {div.divergence_type} → "
                    f"{div.timeframe_1} vs {div.timeframe_2}: {div.description}"
                )

        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("-" * 70)
        print(f"Primary Trend (1d)      : {analysis_cli.primary_trend.value}")
        print(f"Strongest Timeframe     : {analysis_cli.strongest_timeframe}")
        print(f"Trade Recommendation    : {analysis_cli.trade_recommendation}")
        print(f"Suggested Action        : {analysis_cli.confluence.recommended_action}")

        allowed, reason = should_trade_with_confluence(
            analysis_cli.confluence.confluence_score, min_threshold=60.0
        )
        print(f"\nConfluence Gate         : {'✅ ALLOW' if allowed else '❌ BLOCK'}")
        print(f"Reason                  : {reason}")

        print("\n" + "=" * 70 + "\n")

    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error: {exc}")
        traceback.print_exc()
