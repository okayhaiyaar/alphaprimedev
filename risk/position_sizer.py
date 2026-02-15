"""
============================================================
ALPHA-PRIME v2.0 - Position Sizer (Risk Management)
============================================================

Calculates optimal position sizes using multiple methodologies:

1. ATR-Based (Volatility-Adjusted):
   - Shares = Risk_Amount / (ATR * Multiplier)  [web:171][web:167]
   - Default method (recommended)

2. Fixed Percentage:
   - Shares = (Portfolio * Allocation_%) / Price  [web:171]

3. Kelly Criterion:
   - Kelly_Fraction ≈ (Win_Rate * Avg_Win - (1 - Win_Rate) * Avg_Loss) / Avg_Win  [web:162][web:163]
   - Shares = (Kelly * Portfolio) / Price
   - Aggressive, use with fractional Kelly (0.25-0.5)  [web:164][web:168]

Risk Controls:
- Maximum position size: 20% of portfolio (configurable)
- Minimum position size: 1 share
- Cash availability validation
- Stop loss validation

Usage:
    from risk.position_sizer import calculate_position_size

    result = calculate_position_size(
        portfolio_value=10000,
        risk_per_trade_pct=2.0,
        entry_price=150.00,
        stop_loss=145.00,
        method="ATR",
        atr=2.50,
    )

    print(f"Position size: {result.shares} shares")

Integration:
- Called by scheduler.py before trade execution
- Respects settings.max_risk_per_trade_pct
- Validates against portfolio constraints
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class PositionSizeResult:
    """
    Result of position size calculation.

    Attributes:
        shares: Final number of shares to trade (0 if invalid).
        method: Method used ("ATR", "FIXED", "KELLY").
        risk_amount: Dollar risk based on risk_per_trade_pct.
        position_value: Dollar value of the position at entry.
        position_pct: Position size as percentage of portfolio.
        validation_passed: True if all risk constraints passed.
        validation_message: Explanation of validation outcome.
    """

    shares: int
    method: str
    risk_amount: float
    position_value: float
    position_pct: float
    validation_passed: bool
    validation_message: str


# ──────────────────────────────────────────────────────────
# ATR-BASED POSITION SIZING (Volatility-Adjusted)
# ──────────────────────────────────────────────────────────


def calculate_atr_position_size(
    portfolio_value: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss: float,
    atr: Optional[float] = None,
) -> int:
    """
    Calculate position size using ATR/stop-distance based method.

    Core idea:
        Risk_Amount = Portfolio_Value * (Risk_Per_Trade_% / 100)
        Price_Risk_Per_Share = abs(Entry_Price - Stop_Loss)
        Shares = Risk_Amount / Price_Risk_Per_Share  [web:167][web:171]

    If ATR is provided, stop distance is sanity-checked against ATR:
        - Very tight (< 0.5 * ATR): may cause premature stop-outs.
        - Very wide (> 3 * ATR): may imply excessive per-share risk.

    Args:
        portfolio_value: Total portfolio value.
        risk_per_trade_pct: Risk percentage (e.g., 2.0 for 2%).
        entry_price: Planned entry price.
        stop_loss: Stop-loss price.
        atr: Average True Range (optional, for validation hints).

    Returns:
        Integer number of shares (0 if invalid).
    """
    if portfolio_value <= 0:
        logger.error("ATR sizing: invalid portfolio value %.2f.", portfolio_value)
        return 0

    if entry_price <= 0 or stop_loss <= 0:
        logger.error(
            "ATR sizing: invalid entry/stop (entry=%.4f, stop=%.4f).",
            entry_price,
            stop_loss,
        )
        return 0

    if risk_per_trade_pct <= 0 or risk_per_trade_pct > 10:
        logger.warning(
            "ATR sizing: risk_per_trade_pct %.2f outside recommended range (0–10%%).",
            risk_per_trade_pct,
        )

    risk_amount = portfolio_value * (risk_per_trade_pct / 100.0)
    price_risk_per_share = abs(entry_price - stop_loss)

    if price_risk_per_share <= 0:
        logger.error("ATR sizing: stop loss equals entry price (no distance).")
        return 0

    if atr is not None and atr > 0:
        actual_stop_distance = price_risk_per_share
        if actual_stop_distance < 0.5 * atr:
            logger.warning(
                "ATR sizing: stop loss too tight (%.2f < 0.5 * ATR=%.2f).",
                actual_stop_distance,
                0.5 * atr,
            )
        elif actual_stop_distance > 3.0 * atr:
            logger.warning(
                "ATR sizing: stop loss too wide (%.2f > 3 * ATR=%.2f).",
                actual_stop_distance,
                3.0 * atr,
            )

    shares = int(risk_amount / price_risk_per_share)
    if shares < 0:
        shares = 0

    logger.info(
        "ATR position size: %d shares (risk=$%.2f, distance=$%.2f/share).",
        shares,
        risk_amount,
        price_risk_per_share,
    )
    return shares


# ──────────────────────────────────────────────────────────
# KELLY CRITERION POSITION SIZING
# ──────────────────────────────────────────────────────────


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Compute Kelly fraction based on win rate and average win/loss.  [web:162][web:163]

    Approximate discrete-trade Kelly:
        Kelly = (p * Avg_Win - (1 - p) * Avg_Loss) / Avg_Win

    where:
        p          = win_rate (0–1),
        Avg_Win    = average winning trade amount (> 0),
        Avg_Loss   = average losing trade amount (> 0, absolute value).

    Args:
        win_rate: Historical win rate (0–1).
        avg_win: Average winning trade amount (positive).
        avg_loss: Average losing trade amount (positive).

    Returns:
        Kelly fraction in [0, 1]. Returns 0 if invalid or negative edge.
    """
    if not (0.0 <= win_rate <= 1.0):
        logger.error("Kelly fraction: invalid win_rate=%.3f (expected 0–1).", win_rate)
        return 0.0

    if avg_win <= 0 or avg_loss <= 0:
        logger.error(
            "Kelly fraction: invalid avg_win/avg_loss (win=%.2f, loss=%.2f).",
            avg_win,
            avg_loss,
        )
        return 0.0

    edge = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    kelly = edge / avg_win

    if kelly <= 0:
        logger.warning("Kelly fraction: non-positive edge (kelly=%.4f).", kelly)
        return 0.0

    kelly_clamped = max(0.0, min(1.0, kelly))
    logger.info("Kelly fraction computed: raw=%.4f, clamped=%.4f.", kelly, kelly_clamped)
    return kelly_clamped


def calculate_kelly_position_size(
    portfolio_value: float,
    entry_price: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fractional_kelly: float = 0.25,
) -> int:
    """
    Calculate position size using Kelly Criterion (fractional).  [web:162][web:164][web:168]

    WARNING:
        Full Kelly is aggressive and can lead to large drawdowns in practice.
        Common practice is to use fractional Kelly (0.25–0.5) for robustness.

    Args:
        portfolio_value: Total portfolio value.
        entry_price: Entry price per share (> 0).
        win_rate: Historical win rate (0–1, e.g., 0.60 for 60%).
        avg_win: Average winning trade P&L (> 0).
        avg_loss: Average losing trade P&L (> 0, absolute).
        fractional_kelly: Fraction of Kelly to use (e.g., 0.25 for quarter Kelly).

    Returns:
        Integer number of shares (0 if invalid).
    """
    if portfolio_value <= 0 or entry_price <= 0:
        logger.error(
            "Kelly sizing: invalid inputs (portfolio=%.2f, price=%.4f).",
            portfolio_value,
            entry_price,
        )
        return 0

    if fractional_kelly <= 0 or fractional_kelly > 1.0:
        logger.warning(
            "Kelly sizing: fractional_kelly %.3f outside recommended range (0–1).",
            fractional_kelly,
        )

    kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
    if kelly <= 0:
        logger.warning("Kelly sizing: zero Kelly fraction, returning 0 shares.")
        return 0

    effective_kelly = kelly * fractional_kelly
    allocation = portfolio_value * effective_kelly
    shares = int(allocation / entry_price)
    if shares < 0:
        shares = 0

    logger.info(
        "Kelly position size: %d shares (kelly=%.3f, fract=%.3f, alloc=$%.2f).",
        shares,
        kelly,
        fractional_kelly,
        allocation,
    )
    return shares


# ──────────────────────────────────────────────────────────
# FIXED PERCENTAGE POSITION SIZING
# ──────────────────────────────────────────────────────────


def calculate_fixed_position_size(
    portfolio_value: float,
    entry_price: float,
    allocation_pct: float = 10.0,
) -> int:
    """
    Calculate position size with fixed portfolio allocation percentage.

    Formula:
        Allocation_Amount = Portfolio_Value * (Allocation_% / 100)
        Shares = Allocation_Amount / Entry_Price

    Args:
        portfolio_value: Total portfolio value.
        entry_price: Entry price per share (> 0).
        allocation_pct: Allocation percentage (e.g., 10.0 = 10%).

    Returns:
        Integer number of shares (0 if invalid).
    """
    if portfolio_value <= 0 or entry_price <= 0:
        logger.error(
            "Fixed sizing: invalid portfolio/price (portfolio=%.2f, price=%.4f).",
            portfolio_value,
            entry_price,
        )
        return 0

    if allocation_pct <= 0 or allocation_pct > 100:
        logger.error("Fixed sizing: invalid allocation_pct %.2f (expected 0–100).", allocation_pct)
        return 0

    allocation_amount = portfolio_value * (allocation_pct / 100.0)
    shares = int(allocation_amount / entry_price)
    if shares < 0:
        shares = 0

    logger.info(
        "Fixed position size: %d shares (allocation=%.1f%%, amount=$%.2f).",
        shares,
        allocation_pct,
        allocation_amount,
    )
    return shares


# ──────────────────────────────────────────────────────────
# CASH-BASED LIMITS & VALIDATION
# ──────────────────────────────────────────────────────────


def calculate_max_shares_affordable(
    cash_available: float,
    price: float,
    commission: float = 0.0,
) -> int:
    """
    Calculate maximum shares affordable given available cash and commission.

    Args:
        cash_available: Available cash.
        price: Price per share (> 0).
        commission: Commission per trade (>= 0).

    Returns:
        Max integer number of shares (>= 0).
    """
    if cash_available <= commission or price <= 0:
        return 0

    max_shares = int((cash_available - commission) / price)
    return max(0, max_shares)


def validate_position_size(
    quantity: int,
    portfolio_value: float,
    price: float,
    max_position_pct: float = 20.0,
) -> Tuple[bool, str]:
    """
    Validate position size against portfolio-level risk constraints.

    Checks:
        1. quantity >= 1 share (minimum unit).
        2. position_value <= portfolio_value (cannot exceed account).
        3. position_pct <= max_position_pct (cap concentration).

    Args:
        quantity: Proposed number of shares.
        portfolio_value: Total portfolio value.
        price: Entry price per share.
        max_position_pct: Maximum allowed position as % of portfolio.

    Returns:
        (passed, message) pair indicating validation result.
    """
    if quantity < 1:
        return False, "Position size too small (< 1 share)."

    if portfolio_value <= 0 or price <= 0:
        return False, "Invalid portfolio value or price."

    position_value = quantity * price
    position_pct = position_value / portfolio_value * 100.0

    if position_value > portfolio_value:
        return (
            False,
            f"Position value ${position_value:.2f} exceeds portfolio ${portfolio_value:.2f}.",
        )

    if position_pct > max_position_pct:
        max_value = portfolio_value * max_position_pct / 100.0
        return (
            False,
            "Position %.1f%% exceeds max %.1f%% "
            "($%.2f > $%.2f)." % (position_pct, max_position_pct, position_value, max_value),
        )

    return True, (
        f"Valid position: {quantity} shares, "
        f"${position_value:.2f} ({position_pct:.1f}% of portfolio)."
    )


def adjust_position_for_cash(
    desired_shares: int,
    price: float,
    cash_available: float,
    commission: float = 0.0,
) -> int:
    """
    Adjust desired position size based on available cash.

    Args:
        desired_shares: Target number of shares from sizing method.
        price: Entry price per share.
        cash_available: Cash available for the trade.
        commission: Commission per trade.

    Returns:
        Adjusted shares that fit within cash constraint.
    """
    max_affordable = calculate_max_shares_affordable(cash_available, price, commission)
    if desired_shares <= max_affordable:
        return max(desired_shares, 0)

    logger.warning(
        "Adjusting position for cash: desired=%d, max_affordable=%d.",
        desired_shares,
        max_affordable,
    )
    return max_affordable


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> float:
    """
    Calculate reward:risk ratio (R multiple).

    Ratio:
        risk = |entry - stop|
        reward = |take_profit - entry|
        RR = reward / risk  (e.g., 2.0 = 1:2 risk:reward).

    Args:
        entry_price: Entry price.
        stop_loss: Stop-loss price.
        take_profit: Take-profit price.

    Returns:
        Reward-to-risk ratio. Returns 0.0 if risk is zero.
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)

    if risk <= 0:
        return 0.0

    return reward / risk


# ──────────────────────────────────────────────────────────
# MAIN POSITION SIZE CALCULATION (ENTRY POINT)
# ──────────────────────────────────────────────────────────


def calculate_position_size(
    portfolio_value: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss: float,
    method: Literal["ATR", "FIXED", "KELLY"] = "ATR",
    atr: Optional[float] = None,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    cash_available: Optional[float] = None,
) -> PositionSizeResult:
    """
    Main entry point for position sizing.

    Orchestrates method-specific sizing, then applies:
        - max_position_pct constraint,
        - min 1-share constraint,
        - cash-availability adjustment (if cash_available provided).

    Args:
        portfolio_value: Total portfolio value.
        risk_per_trade_pct: Risk % per trade (or allocation % for FIXED).
        entry_price: Entry price per share.
        stop_loss: Stop loss price.
        method: "ATR", "FIXED", or "KELLY".
        atr: Average True Range (for ATR method).
        win_rate: Win rate (0–1) for Kelly.
        avg_win: Average win amount for Kelly.
        avg_loss: Average loss amount for Kelly.
        cash_available: Optional cash cap; if provided, position shrinks as needed.

    Returns:
        PositionSizeResult with final shares and validation info.
    """
    logger.info("Calculating position size (method=%s).", method)

    max_position_pct = float(getattr(settings, "max_position_pct", 20.0))

    if portfolio_value <= 0 or entry_price <= 0:
        msg = "Invalid portfolio value or entry price."
        logger.error("Position sizing: %s", msg)
        return PositionSizeResult(
            shares=0,
            method=method,
            risk_amount=0.0,
            position_value=0.0,
            position_pct=0.0,
            validation_passed=False,
            validation_message=msg,
        )

    risk_amount = portfolio_value * (risk_per_trade_pct / 100.0)

    # Method-specific sizing
    if method == "ATR":
        shares_raw = calculate_atr_position_size(
            portfolio_value=portfolio_value,
            risk_per_trade_pct=risk_per_trade_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
            atr=atr,
        )
    elif method == "FIXED":
        shares_raw = calculate_fixed_position_size(
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            allocation_pct=risk_per_trade_pct,
        )
    elif method == "KELLY":
        if win_rate is None or avg_win is None or avg_loss is None:
            logger.error("Kelly sizing requires win_rate, avg_win, and avg_loss.")
            shares_raw = 0
        else:
            shares_raw = calculate_kelly_position_size(
                portfolio_value=portfolio_value,
                entry_price=entry_price,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                fractional_kelly=float(
                    getattr(settings, "fractional_kelly", 0.25)
                ),
            )
    else:
        msg = f"Unknown position sizing method: {method}"
        logger.error(msg)
        return PositionSizeResult(
            shares=0,
            method=method,
            risk_amount=risk_amount,
            position_value=0.0,
            position_pct=0.0,
            validation_passed=False,
            validation_message=msg,
        )

    if shares_raw < 0:
        shares_raw = 0

    # Apply cash constraint if provided
    if cash_available is not None:
        commission = float(getattr(settings, "commission_per_trade", 0.0))
        shares_adj = adjust_position_for_cash(
            desired_shares=shares_raw,
            price=entry_price,
            cash_available=cash_available,
            commission=commission,
        )
    else:
        shares_adj = shares_raw

    position_value = shares_adj * entry_price
    position_pct = position_value / portfolio_value * 100.0 if portfolio_value > 0 else 0.0

    passed, message = validate_position_size(
        quantity=shares_adj,
        portfolio_value=portfolio_value,
        price=entry_price,
        max_position_pct=max_position_pct,
    )

    shares_final = shares_adj if passed else 0

    logger.info(
        "Position sizing result: shares=%d, value=$%.2f (%.2f%%), valid=%s.",
        shares_final,
        position_value,
        position_pct,
        passed,
    )

    if not passed:
        logger.warning("Position sizing validation failed: %s", message)

    return PositionSizeResult(
        shares=shares_final,
        method=method,
        risk_amount=risk_amount,
        position_value=position_value,
        position_pct=position_pct,
        validation_passed=passed,
        validation_message=message,
    )


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ALPHA-PRIME Position Sizer - Test Tool")
    print("=" * 70 + "\n")

    portfolio_value = 10000.0
    entry_price = 150.00
    stop_loss = 145.00
    atr_val = 2.50

    print(f"Portfolio Value : ${portfolio_value:,.2f}")
    print(f"Entry Price     : ${entry_price:.2f}")
    print(f"Stop Loss       : ${stop_loss:.2f}")
    print(f"ATR             : ${atr_val:.2f}")
    print("\n" + "-" * 70 + "\n")

    # 1. ATR method
    print("1. ATR-Based Method (2% risk):")
    res_atr = calculate_position_size(
        portfolio_value=portfolio_value,
        risk_per_trade_pct=2.0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        method="ATR",
        atr=atr_val,
        cash_available=portfolio_value,
    )
    print(f"   Shares        : {res_atr.shares}")
    print(f"   Position Value: ${res_atr.position_value:.2f}")
    print(f"   Position %    : {res_atr.position_pct:.2f}%")
    print(f"   Validation    : {res_atr.validation_message}\n")

    # 2. Fixed method
    print("2. Fixed Percentage Method (10% allocation):")
    res_fixed = calculate_position_size(
        portfolio_value=portfolio_value,
        risk_per_trade_pct=10.0,  # used as allocation %
        entry_price=entry_price,
        stop_loss=stop_loss,
        method="FIXED",
        cash_available=portfolio_value,
    )
    print(f"   Shares        : {res_fixed.shares}")
    print(f"   Position Value: ${res_fixed.position_value:.2f}")
    print(f"   Position %    : {res_fixed.position_pct:.2f}%")
    print(f"   Validation    : {res_fixed.validation_message}\n")

    # 3. Kelly method
    print("3. Kelly Criterion Method (60% win rate, fractional=0.25):")
    res_kelly = calculate_position_size(
        portfolio_value=portfolio_value,
        risk_per_trade_pct=2.0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        method="KELLY",
        win_rate=0.60,
        avg_win=500.0,
        avg_loss=300.0,
        cash_available=portfolio_value,
    )
    print(f"   Shares        : {res_kelly.shares}")
    print(f"   Position Value: ${res_kelly.position_value:.2f}")
    print(f"   Position %    : {res_kelly.position_pct:.2f}%")
    print(f"   Validation    : {res_kelly.validation_message}\n")

    rr = calculate_risk_reward_ratio(entry_price, stop_loss, 160.00)
    print(f"Risk:Reward Ratio (TP=$160): 1:{rr:.2f}")

    print("\n" + "=" * 70 + "\n")
