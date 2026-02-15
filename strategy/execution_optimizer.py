"""
============================================================
ALPHA-PRIME v2.0 - Execution Optimizer
============================================================

Optimizes trade execution to minimize slippage and costs:

Why Execution Matters:
- Slippage is the difference between expected and actual fill price. [web:250][web:253]
- Market orders prioritize speed and often suffer higher slippage. [web:252][web:253]
- Limit orders cap worst-case price and reduce negative slippage risk. [web:251][web:252]
- Poor execution can turn an otherwise profitable strategy unprofitable over time.

Execution Quality Metrics:
1. Slippage: Difference between expected and actual fill price.
   - < 0.1%: Excellent
   - 0.1–0.3%: Good
   - 0.3–0.5%: Acceptable
   - > 0.5%: Poor (needs improvement)

2. Fill Rate: % of orders filled.
   - Market orders: Usually ~100% fill.
   - Limit orders: 60–90% depending on distance to market.
   - Patient limits: 40–70%, may miss entries entirely. [web:252][web:258]

3. Market Impact: Price movement caused by order.
   - Larger orders relative to volume can “walk the book”. [web:252][web:261]
   - Splitting into smaller clips reduces impact.

Order Type Selection:
1. HIGH LIQUIDITY (volume > 1M, spread < 0.2%):
   - Use limit or marketable limit orders.
   - Low slippage risk; saves spread. [web:255][web:261]

2. MEDIUM LIQUIDITY (volume 100k–1M, spread 0.2–0.5%):
   - Use marketable limits.
   - Balance speed and price.

3. LOW LIQUIDITY (volume < 100k, spread > 0.5%):
   - Use patient limits or avoid entirely.
   - Market orders face high slippage risk. [web:255][web:261]

Entry Strategies:
1. BREAKOUT TRADES:
   - Aggressive execution (market or marketable limit).
   - Speed > price (do not miss move).

2. MEAN REVERSION:
   - Patient execution (limits inside entry zone).
   - Price > speed.

3. SWING TRADES:
   - Standard execution (limit near mid-spread).
   - Balanced approach.

Timing Optimization:
- AVOID: First 15 minutes (9:30–9:45 ET) – opening volatility.
- AVOID: Last 15 minutes (15:45–16:00 ET) – closing volatility.
- BEST: Mid-morning (10:00–11:00) or afternoon (14:00–15:00).

Usage:
    from strategy.execution_optimizer import optimize_entry

    plan = optimize_entry(
        ticker="AAPL",
        action="BUY",
        target_price=150.00,
        quantity=100
    )

    print(f"Order Type: {plan.order_type}")
    print(f"Limit Price: {plan.limit_price}")
    print(f"Expected Slippage: {plan.expected_slippage:.2f}%")

Integration:
- Called by portfolio.py before trade execution.
- Adjusts order parameters for optimal fills.
- Designed to support later tracking of realized execution quality.
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & ENUMS
# ──────────────────────────────────────────────────────────


class OrderType(str, Enum):
    """Order type classification."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    MARKETABLE_LIMIT = "MARKETABLE_LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class ExecutionUrgency(str, Enum):
    """Execution urgency levels."""

    AGGRESSIVE = "AGGRESSIVE"
    STANDARD = "STANDARD"
    PATIENT = "PATIENT"


@dataclass
class LiquidityMetrics:
    """
    Liquidity analysis snapshot for a ticker.

    Attributes:
        ticker: Symbol.
        avg_daily_volume: 30-day average daily volume.
        current_volume: Last daily volume.
        volume_percentile: Current volume rank vs last month (0–100).
        bid_ask_spread_dollars: Spread in currency units.
        bid_ask_spread_pct: Spread as percentage of price (0.3 = 0.3%).
        liquidity_score: Composite liquidity score (0–100).
        liquidity_tier: "HIGH" | "MEDIUM" | "LOW".
        order_book_depth: Optional placeholder for depth metrics.
    """

    ticker: str
    avg_daily_volume: int
    current_volume: int
    volume_percentile: float
    bid_ask_spread_dollars: float
    bid_ask_spread_pct: float
    liquidity_score: float
    liquidity_tier: str
    order_book_depth: Optional[Dict[str, float]] = None


@dataclass
class ExecutionPlan:
    """
    Optimized execution plan for a trade.

    Attributes:
        ticker: Symbol.
        action: "BUY" or "SELL".
        quantity: Order size in shares.
        order_type: Selected OrderType.
        limit_price: Limit price if applicable (None for pure market).
        stop_price: Stop trigger price if using stop-limit.
        expected_fill_price: Expected average fill.
        expected_slippage: Expected slippage in percent (e.g. 0.25 = 0.25%).
        fill_probability: Approximate probability of full fill (0–1).
        time_to_fill_estimate_seconds: Rough time-to-fill estimate.
        rationale: Bullet-point explanation of decisions.
        warnings: Potential issues (timing, liquidity, size).
    """

    ticker: str
    action: str
    quantity: int
    order_type: OrderType
    limit_price: Optional[float]
    stop_price: Optional[float]
    expected_fill_price: float
    expected_slippage: float
    fill_probability: float
    time_to_fill_estimate_seconds: int
    rationale: List[str]
    warnings: List[str]


@dataclass
class ExecutionResult:
    """
    Result of an actual order execution (for analytics).

    Attributes:
        ticker: Symbol.
        action: BUY/SELL.
        order_type: Order type used.
        target_price: Intended entry price.
        actual_fill_price: Realized average fill price.
        quantity: Executed size.
        slippage_pct: Realized slippage (%).
        slippage_dollars: Slippage in currency.
        execution_quality: "EXCELLENT" | "GOOD" | "ACCEPTABLE" | "POOR".
        timestamp_utc: ISO timestamp.
    """

    ticker: str
    action: str
    order_type: str
    target_price: float
    actual_fill_price: float
    quantity: int
    slippage_pct: float
    slippage_dollars: float
    execution_quality: str
    timestamp_utc: str


@dataclass
class SlippageEstimate:
    """
    Slippage estimation breakdown.

    Attributes:
        base_slippage_pct: Spread-driven slippage (%).
        volume_impact_pct: Market impact from order size (%).
        volatility_premium_pct: Extra slippage due to volatility (%).
        total_estimated_slippage_pct: Combined estimate (%).
        confidence: Confidence level (0–1).
    """

    base_slippage_pct: float
    volume_impact_pct: float
    volatility_premium_pct: float
    total_estimated_slippage_pct: float
    confidence: float


# ──────────────────────────────────────────────────────────
# LIQUIDITY ANALYSIS
# ──────────────────────────────────────────────────────────


def analyze_liquidity(ticker: str) -> LiquidityMetrics:
    """
    Analyze liquidity profile of a symbol using recent volume and price. [web:255][web:261][web:264]

    Uses 1-month daily history to estimate:
        - average and current volume,
        - bid-ask spread proxy,
        - composite liquidity score and tier.

    Args:
        ticker: Stock symbol.

    Returns:
        LiquidityMetrics instance. Defaults to LOW liquidity on failure.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="1mo", interval="1d")
        if hist.empty:
            raise ValueError(f"No historical data for {ticker}.")

        avg_volume = float(hist["Volume"].mean())
        current_volume = float(hist["Volume"].iloc[-1])
        volume_percentile = (
            float((hist["Volume"] < current_volume).sum()) / len(hist) * 100.0
        )

        current_price = float(hist["Close"].iloc[-1])

        if current_price >= 100.0:
            spread_pct = 0.05
        elif current_price >= 50.0:
            spread_pct = 0.10
        else:
            spread_pct = 0.20

        spread_dollars = current_price * (spread_pct / 100.0)

        volume_score = min(100.0, (avg_volume / 1_000_000.0) * 20.0)
        spread_score = max(0.0, 100.0 - spread_pct * 200.0)
        liquidity_score = volume_score * 0.6 + spread_score * 0.4

        if liquidity_score >= 70.0 and spread_pct < 0.20:
            tier = "HIGH"
        elif liquidity_score >= 40.0 and spread_pct < 0.50:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        metrics = LiquidityMetrics(
            ticker=ticker.upper(),
            avg_daily_volume=int(avg_volume),
            current_volume=int(current_volume),
            volume_percentile=volume_percentile,
            bid_ask_spread_dollars=spread_dollars,
            bid_ask_spread_pct=spread_pct,
            liquidity_score=liquidity_score,
            liquidity_tier=tier,
        )

        logger.info(
            "Liquidity %s: tier=%s, avg_vol=%s, spread=%.2f%%.",
            ticker,
            tier,
            f"{avg_volume:,.0f}",
            spread_pct,
        )
        return metrics
    except Exception as exc:  # noqa: BLE001
        logger.error("Error analyzing liquidity for %s: %s", ticker, exc, exc_info=True)
        return LiquidityMetrics(
            ticker=ticker.upper(),
            avg_daily_volume=0,
            current_volume=0,
            volume_percentile=0.0,
            bid_ask_spread_dollars=0.0,
            bid_ask_spread_pct=1.0,
            liquidity_score=0.0,
            liquidity_tier="LOW",
        )


# ──────────────────────────────────────────────────────────
# SLIPPAGE ESTIMATION
# ──────────────────────────────────────────────────────────


def calculate_slippage_estimate(
    ticker: str,
    quantity: int,
    order_type: OrderType,
    liquidity: Optional[LiquidityMetrics] = None,
) -> SlippageEstimate:
    """
    Estimate expected slippage for a given trade specification. [web:250][web:252][web:261]

    Components:
        - base_slippage: spread-based (depends on order type).
        - volume_impact: large orders relative to ADTV.
        - volatility_premium: for wide spreads / low liquidity.

    Args:
        ticker: Symbol.
        quantity: Order size in shares.
        order_type: OrderType.
        liquidity: Optional precomputed LiquidityMetrics.

    Returns:
        SlippageEstimate with percentage values.
    """
    if liquidity is None:
        liquidity = analyze_liquidity(ticker)

    if order_type == OrderType.MARKET:
        base_slippage = liquidity.bid_ask_spread_pct / 100.0
    elif order_type == OrderType.MARKETABLE_LIMIT:
        base_slippage = liquidity.bid_ask_spread_pct * 0.75 / 100.0
    else:
        base_slippage = liquidity.bid_ask_spread_pct * 0.25 / 100.0

    avg_vol = liquidity.avg_daily_volume
    if avg_vol <= 0:
        order_size_pct = 0.0
    else:
        order_size_pct = quantity / avg_vol * 100.0

    if order_size_pct > 5.0:
        volume_impact = 0.005
    elif order_size_pct > 2.0:
        volume_impact = 0.002
    elif order_size_pct > 0.5:
        volume_impact = 0.0005
    else:
        volume_impact = 0.0001

    if liquidity.bid_ask_spread_pct > 0.50:
        volatility_premium = 0.001
    else:
        volatility_premium = 0.0

    total = base_slippage + volume_impact + volatility_premium

    if liquidity.liquidity_tier == "HIGH":
        confidence = 0.9
    elif liquidity.liquidity_tier == "MEDIUM":
        confidence = 0.7
    else:
        confidence = 0.5

    est = SlippageEstimate(
        base_slippage_pct=base_slippage * 100.0,
        volume_impact_pct=volume_impact * 100.0,
        volatility_premium_pct=volatility_premium * 100.0,
        total_estimated_slippage_pct=total * 100.0,
        confidence=confidence,
    )

    logger.debug(
        "Slippage %s %s: total=%.3f%% (base=%.3f%%, vol=%.3f%%, vol_prem=%.3f%%).",
        ticker,
        order_type.value,
        est.total_estimated_slippage_pct,
        est.base_slippage_pct,
        est.volume_impact_pct,
        est.volatility_premium_pct,
    )
    return est


# ──────────────────────────────────────────────────────────
# ORDER TYPE SELECTION & LIMIT DECISIONS
# ──────────────────────────────────────────────────────────


def select_order_type(
    liquidity: LiquidityMetrics,
    urgency: ExecutionUrgency = ExecutionUrgency.STANDARD,
    strategy_type: str = "SWING",
) -> OrderType:
    """
    Select an order type based on liquidity, urgency, and strategy.

    Rules:
        - AGGRESSIVE / BREAKOUT:
            HIGH liquidity → MARKETABLE_LIMIT,
            otherwise → MARKET.
        - PATIENT / MEAN_REVERSION:
            Prefer LIMIT.
        - STANDARD:
            HIGH → LIMIT, MEDIUM → MARKETABLE_LIMIT, LOW → MARKET.

    Args:
        liquidity: LiquidityMetrics.
        urgency: ExecutionUrgency.
        strategy_type: Strategy label ("BREAKOUT","MEAN_REVERSION","SWING",...).

    Returns:
        OrderType value.
    """
    strategy_type = strategy_type.upper()

    if urgency == ExecutionUrgency.AGGRESSIVE or strategy_type == "BREAKOUT":
        if liquidity.liquidity_tier == "HIGH":
            return OrderType.MARKETABLE_LIMIT
        return OrderType.MARKET

    if urgency == ExecutionUrgency.PATIENT or strategy_type == "MEAN_REVERSION":
        return OrderType.LIMIT

    if liquidity.liquidity_tier == "HIGH":
        return OrderType.LIMIT
    if liquidity.liquidity_tier == "MEDIUM":
        return OrderType.MARKETABLE_LIMIT
    return OrderType.MARKET


def should_use_limit_order(
    ticker: str,
    spread_pct: float,
    urgency: ExecutionUrgency = ExecutionUrgency.STANDARD,
) -> Tuple[bool, float]:
    """
    Decide whether to use a limit order and suggest a limit offset.

    Heuristics:
        - Avoid limits when spread is extremely wide and urgency is not PATIENT.
        - Avoid limits when urgency is AGGRESSIVE.
        - For PATIENT, aim closer to mid-spread.
        - For STANDARD, slightly inside the spread.

    Args:
        ticker: Symbol (for logging only).
        spread_pct: Spread percentage (0.4 = 0.4%).
        urgency: ExecutionUrgency.

    Returns:
        (use_limit, offset_pct) where offset_pct is fraction of spread.
    """
    if spread_pct > 0.5 and urgency != ExecutionUrgency.PATIENT:
        return False, 0.0

    if urgency == ExecutionUrgency.AGGRESSIVE:
        return False, 0.0

    if urgency == ExecutionUrgency.PATIENT:
        offset = spread_pct * 0.5
    else:
        offset = spread_pct * 0.3

    logger.debug(
        "Limit decision %s: use_limit=%s, offset=%.3f%% of price.",
        ticker,
        True,
        offset,
    )
    return True, offset


# ──────────────────────────────────────────────────────────
# ENTRY ZONE & TIMING
# ──────────────────────────────────────────────────────────


def calculate_optimal_entry_zone(
    price: float,
    entry_range: Tuple[float, float],
    action: str = "BUY",
) -> Tuple[float, float]:
    """
    Calculate optimal entry price and limit for a given range.

    For BUY:
        aim near lower 30% of range, max at range_high.
    For SELL:
        aim near upper 70% of range, min at range_low.

    Args:
        price: Current reference price (unused in heuristic).
        entry_range: (low, high) acceptable range.
        action: "BUY" or "SELL".

    Returns:
        (optimal_price, max_acceptable_price_for_action).
    """
    low, high = entry_range
    action = action.upper()

    if low > high:
        low, high = high, low

    if action == "BUY":
        optimal = low + (high - low) * 0.3
        max_acc = high
    else:
        optimal = low + (high - low) * 0.7
        max_acc = low

    return float(optimal), float(max_acc)


def is_favorable_time_to_trade() -> Tuple[bool, str]:
    """
    Check if current time (US/Eastern) is favorable for execution.

    Returns:
        (is_favorable, reason) pair.
    """
    try:
        from pytz import timezone as tz  # type: ignore[import]

        market_tz = tz("US/Eastern")
        now = datetime.now(market_tz)
        current_time = now.time()

        open_time = dt_time(9, 30)
        close_time = dt_time(16, 0)

        if current_time < open_time or current_time >= close_time:
            return False, "Market closed."

        if current_time < dt_time(9, 45):
            return False, "First 15 minutes (opening volatility)."

        if current_time >= dt_time(15, 45):
            return False, "Last 15 minutes (closing volatility)."

        return True, "Favorable intraday window."
    except ImportError:
        return True, "pytz not available; skipping time filter."
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error during time check: %s", exc)
        return True, "Time check failed; proceeding."


def estimate_fill_probability(
    ticker: str,
    limit_price: float,
    current_price: float,
    action: str,
) -> float:
    """
    Estimate probability of fill for a limit order.

    Heuristics:
        - Limits at/through the market are assumed 100% fill.
        - Further away from price reduces probability.

    Args:
        ticker: Symbol (for potential future enhancements).
        limit_price: Limit level.
        current_price: Current market price.
        action: "BUY" or "SELL".

    Returns:
        float in [0, 1].
    """
    action = action.upper()
    if current_price <= 0:
        return 0.0

    diff_pct = abs((limit_price - current_price) / current_price * 100.0)

    if action == "BUY":
        if limit_price >= current_price:
            return 1.0
        if diff_pct < 0.5:
            return 0.85
        if diff_pct < 1.0:
            return 0.65
        if diff_pct < 2.0:
            return 0.40
        return 0.20

    if limit_price <= current_price:
        return 1.0
    if diff_pct < 0.5:
        return 0.85
    if diff_pct < 1.0:
        return 0.65
    if diff_pct < 2.0:
        return 0.40
    return 0.20


# ──────────────────────────────────────────────────────────
# MAIN EXECUTION OPTIMIZATION
# ──────────────────────────────────────────────────────────


def optimize_entry(
    ticker: str,
    action: str,
    target_price: float,
    quantity: int,
    entry_range: Optional[Tuple[float, float]] = None,
    urgency: ExecutionUrgency = ExecutionUrgency.STANDARD,
    strategy_type: str = "SWING",
) -> ExecutionPlan:
    """
    Build an execution plan optimizing order type, price, and timing.

    Steps:
        1. Analyze liquidity for spread and volume.
        2. Select order type from liquidity + urgency + strategy.
        3. Estimate slippage and expected fill.
        4. Suggest limit price if applicable.
        5. Assess fill probability and time-to-fill.
        6. Produce warnings about timing, liquidity, and order size.

    Args:
        ticker: Symbol.
        action: "BUY" or "SELL".
        target_price: Oracle / strategy preferred price.
        quantity: Shares.
        entry_range: Optional allowable entry band (low, high).
        urgency: ExecutionUrgency.
        strategy_type: Strategy label for context.

    Returns:
        ExecutionPlan describing recommended execution.
    """
    action = action.upper()
    logger.info(
        "Optimizing execution for %s %d %s @ %.4f (urgency=%s, strat=%s).",
        action,
        quantity,
        ticker,
        target_price,
        urgency.value,
        strategy_type,
    )

    rationale: List[str] = []
    warnings: List[str] = []

    liquidity = analyze_liquidity(ticker)
    rationale.append(
        f"Liquidity tier {liquidity.liquidity_tier} (spread={liquidity.bid_ask_spread_pct:.2f}%, "
        f"avg_vol={liquidity.avg_daily_volume:,})."
    )

    order_type = select_order_type(liquidity, urgency, strategy_type)
    rationale.append(f"Selected order type {order_type.value} based on liquidity and urgency.")

    slippage_est = calculate_slippage_estimate(
        ticker=ticker,
        quantity=quantity,
        order_type=order_type,
        liquidity=liquidity,
    )

    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    if order_type == OrderType.LIMIT:
        if entry_range is not None:
            optimal, _ = calculate_optimal_entry_zone(
                price=target_price,
                entry_range=entry_range,
                action=action,
            )
            limit_price = optimal
            rationale.append(
                f"Using limit in suggested entry zone: {entry_range[0]:.2f}–{entry_range[1]:.2f}, "
                f"optimal={optimal:.2f}."
            )
        else:
            _, offset_frac = should_use_limit_order(
                ticker, liquidity.bid_ask_spread_pct, urgency
            )
            if action == "BUY":
                limit_price = target_price * (1.0 - offset_frac / 100.0)
            else:
                limit_price = target_price * (1.0 + offset_frac / 100.0)
            rationale.append(f"Limit price anchored around target with offset={offset_frac:.3f}%.")

    elif order_type == OrderType.MARKETABLE_LIMIT:
        offset_frac = liquidity.bid_ask_spread_pct * 0.2 / 100.0
        if action == "BUY":
            limit_price = target_price * (1.0 + offset_frac)
        else:
            limit_price = target_price * (1.0 - offset_frac)
        rationale.append(
            f"Marketable limit price {limit_price:.2f} with offset {offset_frac*100:.3f}% around target."
        )

    if order_type == OrderType.MARKET:
        expected_fill = target_price * (
            1.0 + (slippage_est.total_estimated_slippage_pct / 100.0)
        )
    else:
        expected_fill = float(limit_price) if limit_price is not None else target_price

    if order_type == OrderType.MARKET:
        fill_prob = 1.0
    else:
        ref_price = target_price
        fill_prob = estimate_fill_probability(
            ticker=ticker,
            limit_price=limit_price if limit_price is not None else ref_price,
            current_price=ref_price,
            action=action,
        )

    rationale.append(f"Estimated fill probability {fill_prob:.0%}.")

    if order_type == OrderType.MARKET:
        ttf = 1
    elif fill_prob >= 0.8:
        ttf = 30
    elif fill_prob >= 0.6:
        ttf = 300
    else:
        ttf = 1800

    favorable, time_msg = is_favorable_time_to_trade()
    if not favorable:
        warnings.append(f"Timing caution: {time_msg}")
    else:
        rationale.append(f"Timing OK: {time_msg}")

    if liquidity.avg_daily_volume > 0:
        order_size_pct = quantity / liquidity.avg_daily_volume * 100.0
    else:
        order_size_pct = 0.0

    if order_size_pct > 5.0:
        warnings.append(
            f"Large order ({order_size_pct:.1f}% of ADTV) – consider splitting or VWAP-style execution."
        )
    elif order_size_pct > 2.0:
        warnings.append(
            f"Moderate order size ({order_size_pct:.1f}% of ADTV) – watch for market impact."
        )

    if liquidity.liquidity_tier == "LOW":
        warnings.append("Low liquidity symbol – expect wider slippage and slower fills.")

    plan = ExecutionPlan(
        ticker=ticker.upper(),
        action=action,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        expected_fill_price=float(expected_fill),
        expected_slippage=float(slippage_est.total_estimated_slippage_pct),
        fill_probability=float(fill_prob),
        time_to_fill_estimate_seconds=int(ttf),
        rationale=rationale,
        warnings=warnings,
    )

    logger.info(
        "Execution plan %s %s: type=%s, expected_slippage=%.3f%%, fill_prob=%.0f%%.",
        action,
        ticker,
        order_type.value,
        plan.expected_slippage,
        plan.fill_probability * 100.0,
    )
    return plan


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Execution Optimizer")
    print("=" * 70 + "\n")

    if len(sys.argv) < 5:
        print("Usage: python execution_optimizer.py TICKER ACTION PRICE QUANTITY")
        print("Example: python execution_optimizer.py AAPL BUY 150.00 100\n")
        sys.exit(1)

    try:
        ticker_cli = sys.argv[1].upper()
        action_cli = sys.argv[2].upper()
        target_price_cli = float(sys.argv[3])
        quantity_cli = int(sys.argv[4])

        print(
            f"Analyzing execution for: {action_cli} {quantity_cli} {ticker_cli} "
            f"@ {target_price_cli:.2f}\n"
        )

        print("LIQUIDITY ANALYSIS")
        print("-" * 70)
        liq_cli = analyze_liquidity(ticker_cli)
        print(f"Avg Daily Volume : {liq_cli.avg_daily_volume:,}")
        print(
            f"Bid-Ask Spread   : {liq_cli.bid_ask_spread_pct:.2f}% "
            f"(${liq_cli.bid_ask_spread_dollars:.4f})"
        )
        print(f"Liquidity Score  : {liq_cli.liquidity_score:.1f}/100")
        print(f"Tier             : {liq_cli.liquidity_tier}\n")

        print("EXECUTION PLANS BY URGENCY")
        print("-" * 70)
        for urg in ExecutionUrgency:
            plan_cli = optimize_entry(
                ticker_cli,
                action_cli,
                target_price_cli,
                quantity_cli,
                urgency=urg,
            )
            print(f"\nUrgency: {urg.value}")
            print(f"  Order Type    : {plan_cli.order_type.value}")
            if plan_cli.limit_price is not None:
                print(f"  Limit Price   : {plan_cli.limit_price:.4f}")
            print(f"  Exp. Fill     : {plan_cli.expected_fill_price:.4f}")
            print(f"  Exp. Slippage : {plan_cli.expected_slippage:.3f}%")
            print(f"  Fill Prob     : {plan_cli.fill_probability:.0%}")
            print(f"  Est. TTF      : {plan_cli.time_to_fill_estimate_seconds}s")
            if plan_cli.warnings:
                print("  Warnings:")
                for w in plan_cli.warnings:
                    print(f"    - {w}")

        print("\n" + "=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        traceback.print_exc()
