"""
============================================================
ALPHA-PRIME v2.0 - Circuit Breakers (Risk Management)
============================================================

Automated trading halt conditions to protect capital:

1. Consecutive Loss Limit:
   - Stop after N losing trades in a row
   - Prevents revenge trading and cascading losses  [web:186][web:189]

2. Daily Loss Limit:
   - Halt if daily drawdown exceeds threshold %
   - Protects against catastrophic daily losses  [web:185][web:188][web:191]

3. VIX Spike Protection:
   - Halt when VIX > threshold (e.g., 35)
   - Avoids trading during extreme volatility/fear  [web:180][web:181]

4. Max Positions:
   - Limit total number of open positions
   - Prevents over-diversification and management issues

5. Portfolio Heat:
   - Calculate total risk exposure across all positions
   - Halt if aggregate risk exceeds threshold

6. Market Hours Check:
   - Only allow trades during market hours
   - Prevents after-hours execution issues

Usage:
    from risk.circuit_breakers import check_trade_allowed
    from portfolio import PaperTrader

    trader = PaperTrader()
    allowed, reason = check_trade_allowed(trader)

    if allowed:
        # Execute trade
        ...
    else:
        logger.warning(f"Trade blocked: {reason}")

Integration:
- Called by scheduler.py before every trade
- Overrides Oracle decisions when triggered
- Sends Discord alerts when tripped
============================================================
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dt_time, timezone
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# CONSECUTIVE LOSS PROTECTION
# ──────────────────────────────────────────────────────────


def check_consecutive_losses(
    trader,
    limit: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check whether recent consecutive losing trades exceed a limit.

    Logic:
        - Consider only SELL trades (assumed to realize P&L).
        - Walk backwards from most recent and count consecutive
          negative P&L values until a winning/non-losing trade breaks the streak.

    Purpose:
        Prevents revenge trading and emotional spirals after a losing streak. [web:186][web:189]

    Args:
        trader: PaperTrader instance (must expose trade_history_path).
        limit: Max allowed consecutive losing SELL trades (defaults to
               settings.consecutive_loss_limit if None).

    Returns:
        (allowed, reason) where:
            allowed = False if breaker is tripped,
            reason  = explanation string.
    """
    if limit is None:
        limit = int(getattr(settings, "consecutive_loss_limit", 3))

    try:
        path = trader.trade_history_path
        if not path.exists():
            return True, "No trade history yet."

        df = pd.read_csv(path)
        if df.empty:
            return True, "No trades recorded yet."

        sells = df[df["action"] == "SELL"].sort_values(
            "timestamp_utc", ascending=False
        )

        if sells.empty:
            return True, "No closed (SELL) trades yet."

        if len(sells) < limit:
            return True, f"Insufficient closed trades ({len(sells)} < {limit})."

        consecutive_losses = 0
        for pnl in sells["pnl"].values:
            try:
                if float(pnl) < 0:
                    consecutive_losses += 1
                else:
                    break
            except Exception:
                break

        if consecutive_losses >= limit:
            msg = (
                f"❌ CIRCUIT BREAKER: {consecutive_losses} consecutive losing trades "
                f"(limit={limit}). Stand down and reassess."
            )
            logger.warning(msg)
            return False, msg

        logger.debug(
            "Consecutive losses: %d (limit=%d) – OK.", consecutive_losses, limit
        )
        return True, f"Consecutive losses OK ({consecutive_losses}/{limit})."

    except Exception as exc:  # noqa: BLE001
        logger.error("Error checking consecutive losses: %s", exc, exc_info=True)
        return True, f"Consecutive loss check failed (allowing trade): {exc}"


# ──────────────────────────────────────────────────────────
# DAILY LOSS LIMIT
# ──────────────────────────────────────────────────────────


def _get_today_starting_value(trader) -> Optional[float]:
    """
    Infer today's starting portfolio value from trade history.

    Heuristic:
        - Get earliest trade today by timestamp_utc.
        - Take portfolio_value_after from previous trade; if none, use that trade.

    Returns:
        Starting value as float, or None if cannot be determined.
    """
    path = trader.trade_history_path
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty or "timestamp_utc" not in df.columns:
        return None

    df["dt"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"])
    if df.empty:
        return None

    today_date = datetime.now(timezone.utc).date()
    today_trades = df[df["dt"].dt.date == today_date]
    if today_trades.empty:
        return None

    today_trades = today_trades.sort_values("dt", ascending=True)
    first_idx = today_trades.index[0]

    prev_idx = first_idx - 1
    if prev_idx in df.index and "portfolio_value_after" in df.columns:
        try:
            return float(df.loc[prev_idx, "portfolio_value_after"])
        except Exception:
            pass

    try:
        return float(today_trades.iloc[0]["portfolio_value_after"])
    except Exception:
        return None


def check_daily_loss_limit(
    trader,
    limit_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Check whether today's equity drawdown exceeds a configured limit.

    Concept:
        daily_pnl_pct = (current_value - start_of_day_value) / start_of_day_value * 100
        If daily_pnl_pct <= -limit_pct → halt for the day. [web:185][web:188][web:191]

    Args:
        trader: PaperTrader instance.
        limit_pct: Daily max loss percentage (e.g., 3.0). Defaults to
                   settings.daily_loss_limit_pct if None.

    Returns:
        (allowed, reason) pair.
    """
    if limit_pct is None:
        limit_pct = float(getattr(settings, "daily_loss_limit_pct", 3.0))

    try:
        portfolio = trader.get_portfolio_state()
        starting_value = _get_today_starting_value(trader)

        if starting_value is None:
            return True, "No reliable start-of-day value; skipping daily loss check."

        current_value = float(portfolio.total_value)
        daily_pnl = current_value - starting_value
        daily_pnl_pct = (
            daily_pnl / starting_value * 100.0 if starting_value > 0 else 0.0
        )

        if daily_pnl_pct <= -limit_pct:
            msg = (
                f"❌ CIRCUIT BREAKER: Daily loss {daily_pnl_pct:.2f}% exceeds "
                f"limit {limit_pct:.2f}% (P&L=${daily_pnl:,.2f}). No more trades today."
            )
            logger.warning(msg)
            return False, msg

        logger.debug(
            "Daily P&L: %.2f%% (limit=-%.2f%%) – OK.", daily_pnl_pct, limit_pct
        )
        return True, f"Daily P&L OK ({daily_pnl_pct:+.2f}% / -{limit_pct:.2f}%)."

    except Exception as exc:  # noqa: BLE001
        logger.error("Error checking daily loss limit: %s", exc, exc_info=True)
        return True, f"Daily loss check failed (allowing trade): {exc}"


# ──────────────────────────────────────────────────────────
# VIX VOLATILITY THRESHOLD
# ──────────────────────────────────────────────────────────


def check_vix_threshold(
    threshold: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Check whether VIX (CBOE volatility index) exceeds a danger threshold.

    Rationale:
        - VIX above ~30–35 often coincides with high stress/volatile markets,
          where discretionary systems may choose to reduce or halt risk. [web:177][web:180][web:181]

    Args:
        threshold: Maximum allowed VIX level.
                   Defaults to settings.vix_shutdown_threshold (e.g., 35).

    Returns:
        (allowed, reason) pair. If data unavailable, allows trade but logs reason.
    """
    if threshold is None:
        threshold = float(getattr(settings, "vix_shutdown_threshold", 35.0))

    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if hist.empty or "Close" not in hist.columns:
            logger.warning("VIX data unavailable (allowing trade).")
            return True, "VIX check skipped (data unavailable)."

        current_vix = float(hist["Close"].iloc[-1])

        if current_vix > threshold:
            msg = (
                f"❌ CIRCUIT BREAKER: VIX at {current_vix:.2f} exceeds threshold "
                f"{threshold:.2f}. Extreme volatility – standing aside."
            )
            logger.warning(msg)
            return False, msg

        logger.debug(
            "VIX level: %.2f (threshold=%.2f) – OK.", current_vix, threshold
        )
        return True, f"VIX OK ({current_vix:.2f} <= {threshold:.2f})."

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error checking VIX: %s (allowing trade).", exc, exc_info=True)
        return True, f"VIX check failed (allowing trade): {exc}"


# ──────────────────────────────────────────────────────────
# MAX POSITIONS LIMIT
# ──────────────────────────────────────────────────────────


def check_max_positions(
    trader,
    max_positions: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check whether the number of open positions exceeds a limit.

    Purpose:
        Avoids over-diversification and operational overload, especially in
        smaller accounts or when monitoring capacity is constrained.

    Args:
        trader: PaperTrader instance.
        max_positions: Maximum allowed open positions. Defaults to
                       settings.max_open_positions (fallback=5).

    Returns:
        (allowed, reason) pair.
    """
    if max_positions is None:
        max_positions = int(getattr(settings, "max_open_positions", 5))

    portfolio = trader.get_portfolio_state()
    current_positions = int(portfolio.position_count)

    if current_positions >= max_positions:
        msg = (
            f"❌ CIRCUIT BREAKER: {current_positions} open positions "
            f"exceeds limit {max_positions}. Reduce exposure before adding."
        )
        logger.warning(msg)
        return False, msg

    logger.debug(
        "Open positions: %d/%d – OK.", current_positions, max_positions
    )
    return True, f"Open positions OK ({current_positions}/{max_positions})."


# ──────────────────────────────────────────────────────────
# PORTFOLIO HEAT (TOTAL RISK EXPOSURE)
# ──────────────────────────────────────────────────────────


def check_portfolio_heat(
    trader,
    max_heat_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Check whether aggregate risk exposure ("portfolio heat") exceeds a limit.

    Simplified implementation:
        - Assume each open position risks settings.max_risk_per_trade_pct.
        - Portfolioheat ≈ position_count * max_risk_per_trade_pct. [web:185][web:191]

    More advanced implementations could:
        - Sum per-position stop-based risk in dollars,
        - Convert to account %,
        - Compare against aggregate max heat threshold.

    Args:
        trader: PaperTrader instance.
        max_heat_pct: Maximum allowed aggregate risk percent.
                      Defaults to settings.max_portfolio_risk_pct (e.g., 15).

    Returns:
        (allowed, reason) pair.
    """
    if max_heat_pct is None:
        max_heat_pct = float(getattr(settings, "max_portfolio_risk_pct", 15.0))

    portfolio = trader.get_portfolio_state()
    if not portfolio.positions:
        return True, "No open positions (heat=0%)."

    per_trade_risk = float(getattr(settings, "max_risk_per_trade_pct", 2.0))
    estimated_heat_pct = portfolio.position_count * per_trade_risk

    if estimated_heat_pct >= max_heat_pct:
        msg = (
            f"❌ CIRCUIT BREAKER: Estimated portfolio heat {estimated_heat_pct:.1f}% "
            f"exceeds limit {max_heat_pct:.1f}%. Reduce risk before trading."
        )
        logger.warning(msg)
        return False, msg

    logger.debug(
        "Portfolio heat: %.1f%% (limit=%.1f%%) – OK.",
        estimated_heat_pct,
        max_heat_pct,
    )
    return True, f"Portfolio heat OK ({estimated_heat_pct:.1f}% / {max_heat_pct:.1f}%)."


# ──────────────────────────────────────────────────────────
# MARKET HOURS CHECK
# ──────────────────────────────────────────────────────────


def check_market_hours(
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
) -> Tuple[bool, str]:
    """
    Check whether current time is within permitted market hours.

    Defaults:
        - US regular session: 09:30–16:00 US/Eastern.
        - Optional pre-market: 04:00–09:30.
        - Optional after-hours: 16:00–20:00.

    Args:
        allow_premarket: If True, allow trades between 04:00–09:30.
        allow_afterhours: If True, allow trades between 16:00–20:00.

    Returns:
        (allowed, reason) pair. If timezone libs missing, allows trade.
    """
    try:
        from pytz import timezone as tz

        market_tz = tz("US/Eastern")
        now = datetime.now(market_tz)
        current_time = now.time()

        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        premarket_start = dt_time(4, 0)
        afterhours_end = dt_time(20, 0)

        if market_open <= current_time <= market_close:
            return True, f"Market open (current {current_time.strftime('%H:%M')} ET)."

        if allow_premarket and premarket_start <= current_time < market_open:
            return True, f"Pre-market (current {current_time.strftime('%H:%M')} ET)."

        if allow_afterhours and market_close < current_time <= afterhours_end:
            return True, f"After-hours (current {current_time.strftime('%H:%M')} ET)."

        msg = (
            f"⚠️ Market closed (current {current_time.strftime('%H:%M')} ET). "
            "Regular hours: 09:30–16:00 ET."
        )
        logger.info(msg)
        return False, msg

    except ImportError:
        logger.warning("pytz not installed – skipping market hours check.")
        return True, "Market hours check skipped (timezone support unavailable)."
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error checking market hours: %s (allowing trade).", exc)
        return True, f"Market hours check failed (allowing trade): {exc}"


# ──────────────────────────────────────────────────────────
# MASTER CHECK FUNCTION
# ──────────────────────────────────────────────────────────


def check_trade_allowed(
    trader,
    check_market_hours_flag: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Master circuit-breaker function deciding whether a trade is allowed.

    Runs all configured checks in sequence:
        1. Consecutive losses,
        2. Daily loss limit,
        3. VIX threshold,
        4. Max positions,
        5. Portfolio heat,
        6. Market hours (optional).

    First failing check halts trading and returns its reason.

    Args:
        trader: PaperTrader instance.
        check_market_hours_flag: Whether to enforce market hours.
        allow_premarket: Allow pre-market trading if True.
        allow_afterhours: Allow after-hours trading if True.

    Returns:
        (allowed, reason_if_blocked) where reason_if_blocked is None when allowed.
    """
    logger.info("Running circuit breaker checks before trade...")

    # 1. Consecutive losses
    allowed, msg = check_consecutive_losses(trader)
    if not allowed:
        return False, msg

    # 2. Daily loss limit
    allowed, msg = check_daily_loss_limit(trader)
    if not allowed:
        return False, msg

    # 3. VIX threshold
    allowed, msg = check_vix_threshold()
    if not allowed:
        return False, msg

    # 4. Max positions
    allowed, msg = check_max_positions(trader)
    if not allowed:
        return False, msg

    # 5. Portfolio heat
    allowed, msg = check_portfolio_heat(trader)
    if not allowed:
        return False, msg

    # 6. Market hours (optional)
    if check_market_hours_flag:
        allowed, msg = check_market_hours(
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
        )
        if not allowed:
            return False, msg

    logger.info("✅ All circuit breaker checks passed – trading allowed.")
    return True, None


# ──────────────────────────────────────────────────────────
# MANUAL CIRCUIT BREAKER RESET (PLACEHOLDER)
# ──────────────────────────────────────────────────────────


def manual_circuit_breaker_reset(trader) -> bool:
    """
    Manually reset circuit breaker state (conceptual placeholder).

    WARNING:
        Use only after:
            1. Reviewing recent performance issues,
            2. Adjusting strategy/parameters,
            3. Confirming that market conditions have normalized. [web:185][web:189]

    Current implementation:
        - Logs reset intent and rationale.
        - Real counters (e.g., daily loss, streaks) are implicit in trade history
          and do not require separate persisted state.

    Args:
        trader: PaperTrader instance (unused, reserved for future stateful logic).

    Returns:
        True to indicate the reset command was acknowledged.
    """
    logger.warning("⚠️ MANUAL CIRCUIT BREAKER RESET REQUESTED.")
    logger.info("Circuit breaker state conceptually reset (no persistent flags).")
    logger.info(
        "NOTE: Underlying conditions (losses, volatility) are unchanged; "
        "this only affects manual override logic."
    )
    return True


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    from portfolio import PaperTrader

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Circuit Breakers - Test Tool")
    print("=" * 70 + "\n")

    trader = PaperTrader()

    print("Running individual circuit breaker checks...\n")

    individual_checks = [
        ("Consecutive Losses", check_consecutive_losses, (trader,), {}),
        ("Daily Loss Limit", check_daily_loss_limit, (trader,), {}),
        ("VIX Threshold", check_vix_threshold, tuple(), {}),
        ("Max Positions", check_max_positions, (trader,), {}),
        ("Portfolio Heat", check_portfolio_heat, (trader,), {}),
        ("Market Hours", check_market_hours, tuple(), {}),
    ]

    for name, func, args, kwargs in individual_checks:
        allowed, msg = func(*args, **kwargs)
        status = "✅ PASS" if allowed else "❌ FAIL"
        print(f"{status} {name}")
        print(f"     {msg}\n")

    print("-" * 70)
    print("MASTER CHECK:\n")

    allowed, reason = check_trade_allowed(trader)
    if allowed:
        print("✅ ALL CHECKS PASSED – Trading allowed.")
    else:
        print("❌ TRADING HALTED")
        print(f"Reason: {reason}")

    print("\n" + "=" * 70 + "\n")
