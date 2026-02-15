"""
============================================================
ALPHA-PRIME v2.0 - Automated Scheduler
============================================================
Module 7: Daily market cycle orchestration.

Workflow:
1. Market Opens (09:30 IST)
2. Scan top movers (volume + price change)
3. For each ticker:
   a. Gather intelligence (research_engine)
   b. Fetch technicals (data_engine)
   c. Get market regime (regime_detector)
   d. Check events (event_calendar)
   e. Consult Oracle (brain)
   f. Check circuit breakers (risk)
   g. Execute trade if confidence >= threshold (portfolio)
   h. Send alert (alerts)
4. Log cycle completion
5. Wait for next day

Usage:
    # Scheduled mode (runs daily at market open)
    python scheduler.py scheduled

    # One-time run (immediate)
    python scheduler.py once

    # Process specific ticker
    python scheduler.py ticker AAPL

    # Test mode (mock data)
    python scheduler.py test

Error Handling:
- Individual ticker failures don't stop cycle
- All errors logged with full context
- System alerts on critical failures
- Automatic retry with exponential backoff

Integration:
- research_engine: Intelligence gathering
- data_engine: Technical analysis
- brain: Oracle decisions
- portfolio: Trade execution
- alerts: Notifications
- risk/circuit_breakers: Safety checks
- validation/drift_monitor: Performance tracking
============================================================
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import schedule
import yfinance as yf

from alerts import send_discord_alert, send_error_alert
from brain import consult_oracle
from config import get_logger, get_settings
from data_engine import calculate_hard_technicals, get_market_data
from portfolio import PaperTrader
from research_engine import get_god_tier_intel

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class ProcessResult:
    """
    Result of processing a single ticker through the full pipeline.

    Attributes:
        ticker: Symbol processed.
        success: True if pipeline completed without uncaught exceptions.
        action_taken: BUY | SELL | WAIT | SKIP | ERROR.
        confidence: Oracle confidence (0–100) if available.
        message: Human-readable summary.
        execution_time_seconds: Total processing time in seconds.
        error: Optional error message on failure.
    """

    ticker: str
    success: bool
    action_taken: str
    confidence: Optional[int]
    message: str
    execution_time_seconds: float
    error: Optional[str] = None


@dataclass
class DailyCycleResult:
    """
    Result of a complete daily cycle.

    Attributes:
        cycle_start_utc: ISO timestamp when cycle started.
        cycle_end_utc: ISO timestamp when cycle ended.
        total_tickers_scanned: Count returned by scanner.
        tickers_processed: Count of tickers successfully processed.
        tickers_failed: Count of tickers with errors.
        actions_taken: Mapping of action → count.
        total_execution_time_seconds: Cycle runtime.
        errors: List of error messages encountered.
    """

    cycle_start_utc: str
    cycle_end_utc: str
    total_tickers_scanned: int
    tickers_processed: int
    tickers_failed: int
    actions_taken: Dict[str, int]
    total_execution_time_seconds: float
    errors: List[str]


# ──────────────────────────────────────────────────────────
# MARKET SCANNER
# ──────────────────────────────────────────────────────────


def scan_market_movers(limit: int = 10, min_volume: int = 1_000_000) -> List[str]:
    """
    Scan for top market movers by volume and price change.

    Strategy:
        1. Use a predefined watchlist (extendable to S&P 500 / screens).
        2. Pull last 2 daily bars via yfinance.
        3. Filter by minimum volume.
        4. Compute activity score:
               score = 0.5 * volume_ratio + 0.5 * |price_change_pct|
        5. Return top N tickers by score.

    Args:
        limit: Maximum number of tickers to return.
        min_volume: Minimum latest-day volume to be considered.

    Returns:
        List of ticker symbols sorted by activity score.
    """
    logger.info("Scanning for top %d market movers...", limit)

    watchlist: List[str] = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "DIS",
        "NFLX",
        "INTC",
        "AMD",
        "PYPL",
        "BA",
        "GE",
        "F",
        "GM",
        "COIN",
        "RBLX",
        "PLTR",
        "SOFI",
        "RIVN",
    ]

    movers: List[Dict[str, float]] = []

    for ticker in watchlist:
        try:
            df = yf.download(
                tickers=ticker,
                period="2d",
                interval="1d",
                progress=False,
            )
            if df.empty or len(df) < 2:
                continue

            current = df.iloc[-1]
            previous = df.iloc[-2]

            volume = float(current["Volume"])
            if volume < min_volume:
                continue

            prev_close = float(previous["Close"])
            curr_close = float(current["Close"])
            if prev_close <= 0:
                continue

            price_change_pct = abs((curr_close - prev_close) / prev_close * 100.0)
            avg_volume = (float(previous["Volume"]) + volume) / 2.0
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

            activity_score = 0.5 * volume_ratio + 0.5 * price_change_pct

            movers.append(
                {
                    "ticker": ticker,
                    "score": activity_score,
                    "price_change_pct": price_change_pct,
                    "volume": volume,
                    "volume_ratio": volume_ratio,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error scanning %s: %s", ticker, exc)
            continue

    movers.sort(key=lambda m: m["score"], reverse=True)
    top = movers[:limit]
    top_tickers = [m["ticker"] for m in top]

    logger.info("Top movers: %s", ", ".join(top_tickers))
    for mover in top:
        logger.info(
            "  %s: score=%.2f, price_change=%.2f%%, volume_ratio=%.2fx",
            mover["ticker"],
            mover["score"],
            mover["price_change_pct"],
            mover["volume_ratio"],
        )

    return top_tickers


# ──────────────────────────────────────────────────────────
# TICKER PROCESSING PIPELINE
# ──────────────────────────────────────────────────────────


def process_ticker(
    ticker: str,
    trader: PaperTrader,
    execute_trades: bool = True,
) -> ProcessResult:
    """
    Process a single ticker through the full pipeline.

    Pipeline:
        1. Gather intelligence (SEC, news, sentiment).
        2. Fetch market data and compute technicals.
        3. Detect market regime (optional).
        4. Fetch upcoming events (optional).
        5. Consult Oracle for decision.
        6. Check circuit breakers.
        7. Execute trade if allowed and configured.
        8. Send Discord alert for executed signals.
        9. Log decision for drift monitoring (optional).

    Args:
        ticker: Stock symbol to process.
        trader: Shared PaperTrader instance.
        execute_trades: If False, pipeline runs in dry-run mode.

    Returns:
        ProcessResult describing outcome.
    """
    t0 = time.time()
    symbol = ticker.upper().strip()

    logger.info("=" * 70)
    logger.info("PROCESSING: %s", symbol)
    logger.info("=" * 70)

    try:
        # STEP 1: Intelligence
        logger.info("[1/7] Gathering intelligence for %s...", symbol)
        intel = get_god_tier_intel(symbol)

        # STEP 2: Technicals
        logger.info("[2/7] Computing technicals for %s...", symbol)
        df = get_market_data(symbol, period="3mo", interval="1d")
        technicals = calculate_hard_technicals(df, ticker=symbol, timeframe="1d")

        dq = technicals.get("data_quality", {})
        quality_score = float(dq.get("quality_score", 0.0))
        if quality_score < 70.0:
            logger.warning(
                "Low data quality for %s (score=%.1f, issues=%s).",
                symbol,
                quality_score,
                dq.get("issues"),
            )

        # STEP 3: Market regime (optional)
        logger.info("[3/7] Detecting market regime...")
        regime = "UNKNOWN"
        try:
            if getattr(settings, "enable_regime_filter", False):
                from strategy.regime_detector import get_current_regime

                regime = get_current_regime()
                logger.info("Current regime: %s", regime)
        except ImportError:
            logger.debug("Regime detector not available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Regime detection failed: %s", exc)

        # STEP 4: Events (optional)
        logger.info("[4/7] Checking upcoming events for %s...", symbol)
        events: List[Dict[str, object]] = []
        try:
            if getattr(settings, "enable_earnings_filter", False):
                from strategy.event_calendar import get_upcoming_events

                events = get_upcoming_events(symbol)
                if events:
                    logger.info("Found %d upcoming events.", len(events))
        except ImportError:
            logger.debug("Event calendar not available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Event retrieval failed: %s", exc)

        # STEP 5: Oracle
        logger.info("[5/7] Consulting Oracle for %s...", symbol)
        decision = consult_oracle(symbol, intel=intel, technicals=technicals, regime=regime, events=events)

        action = decision.action
        confidence = int(decision.confidence)
        logger.info("Oracle verdict for %s: %s (confidence=%d%%).", symbol, action, confidence)

        # STEP 6: Circuit breakers
        logger.info("[6/7] Checking circuit breakers for %s...", symbol)
        trade_allowed = True
        circuit_reason: Optional[str] = None

        try:
            if getattr(settings, "circuit_breaker_enabled", False) and action in ("BUY", "SELL"):
                from risk.circuit_breakers import check_trade_allowed

                trade_allowed, circuit_reason = check_trade_allowed(trader)
                if not trade_allowed:
                    logger.warning("Circuit breaker triggered: %s", circuit_reason)
        except ImportError:
            logger.debug("Circuit breaker module not available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Circuit breaker check failed: %s", exc)

        final_action = action
        if action in ("BUY", "SELL") and not trade_allowed:
            final_action = "SKIP"

        # STEP 7: Execution
        logger.info("[7/7] Execution phase for %s...", symbol)
        exec_msg = ""
        last_price = float(technicals["price_action"]["last_price"])

        if final_action == "WAIT":
            exec_msg = "No trade (Oracle WAIT)."
        elif final_action == "SKIP":
            exec_msg = f"Trade blocked by circuit breaker: {circuit_reason or 'N/A'}."
        elif final_action in ("BUY", "SELL") and not execute_trades:
            exec_msg = "Dry-run mode; trade not executed."
        elif final_action in ("BUY", "SELL") and execute_trades:
            # Position sizing
            portfolio_value = trader.get_portfolio_state().total_value
            quantity: int
            try:
                from risk.position_sizer import calculate_position_size

                quantity = calculate_position_size(
                    portfolio_value=portfolio_value,
                    risk_per_trade_pct=getattr(settings, "max_risk_per_trade_pct", 1.0),
                    entry_price=last_price,
                    stop_loss=float(decision.stop_loss),
                )
            except ImportError:
                quantity = int((portfolio_value * 0.1) / last_price) if last_price > 0 else 0
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position sizing failed: %s", exc)
                quantity = 0

            if quantity <= 0:
                exec_msg = "Trade skipped; computed quantity is 0."
            else:
                trade_result = trader.execute_trade(
                    action=final_action,
                    ticker=symbol,
                    price=last_price,
                    quantity=quantity,
                    notes=f"Oracle confidence={confidence}%",
                )
                if trade_result.success:
                    exec_msg = trade_result.message
                    logger.info("Trade executed: %s", exec_msg)
                    pf = trader.get_portfolio_state()
                    send_discord_alert(
                        decision=asdict(decision),
                        portfolio_summary={
                            "cash": pf.cash,
                            "positions": pf.position_count,
                            "total_value": pf.total_value,
                        },
                    )
                else:
                    exec_msg = f"Trade failed: {trade_result.message}"
                    logger.error("Trade failed: %s", trade_result.error or trade_result.message)
        else:
            exec_msg = "No recognized action; nothing executed."

        # Drift monitoring
        try:
            if getattr(settings, "enable_drift_monitoring", False):
                from validation.drift_monitor import log_prediction

                log_prediction(
                    ticker=symbol,
                    action=decision.action,
                    confidence=int(decision.confidence),
                    technicals=technicals,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
        except ImportError:
            logger.debug("Drift monitor not available.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Drift logging failed for %s: %s", symbol, exc)

        elapsed = time.time() - t0
        logger.info("Ticker %s processed in %.1fs; final action=%s.", symbol, elapsed, final_action)
        logger.info("=" * 70)

        return ProcessResult(
            ticker=symbol,
            success=True,
            action_taken=final_action,
            confidence=confidence,
            message=exec_msg or f"{final_action} signal processed.",
            execution_time_seconds=elapsed,
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        msg = f"Error processing {symbol}: {exc}"
        logger.error(msg)
        logger.debug(traceback.format_exc())

        if "OpenAI" in str(exc) or "API" in str(exc):
            send_error_alert(msg, "ERROR")

        return ProcessResult(
            ticker=symbol,
            success=False,
            action_taken="ERROR",
            confidence=None,
            message=msg,
            execution_time_seconds=elapsed,
            error=str(exc),
        )


# ──────────────────────────────────────────────────────────
# DAILY CYCLE ORCHESTRATION
# ──────────────────────────────────────────────────────────


def run_daily_cycle(limit: int = 10, execute_trades: bool = True) -> DailyCycleResult:
    """
    Execute the complete daily trading cycle on the top market movers.

    Args:
        limit: Number of tickers to scan and process.
        execute_trades: If False, run pipeline in dry-run mode.

    Returns:
        DailyCycleResult summary.
    """
    cycle_start = datetime.now(timezone.utc)
    logger.info("\n" + "=" * 70)
    logger.info("DAILY CYCLE STARTED @ %s", cycle_start.isoformat())
    logger.info("=" * 70 + "\n")

    trader = PaperTrader()
    errors: List[str] = []
    results: List[ProcessResult] = []

    try:
        tickers = scan_market_movers(limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.error("Market scan failed: %s", exc, exc_info=True)
        cycle_end = datetime.now(timezone.utc)
        return DailyCycleResult(
            cycle_start_utc=cycle_start.isoformat(),
            cycle_end_utc=cycle_end.isoformat(),
            total_tickers_scanned=0,
            tickers_processed=0,
            tickers_failed=0,
            actions_taken={},
            total_execution_time_seconds=(cycle_end - cycle_start).total_seconds(),
            errors=[str(exc)],
        )

    for idx, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Processing %s...", idx, len(tickers), ticker)
        result = process_ticker(ticker=ticker, trader=trader, execute_trades=execute_trades)
        results.append(result)

        if not result.success:
            errors.append(f"{ticker}: {result.error or result.message}")

        time.sleep(2)

    cycle_end = datetime.now(timezone.utc)
    total_time = (cycle_end - cycle_start).total_seconds()

    actions: Dict[str, int] = {}
    for r in results:
        actions[r.action_taken] = actions.get(r.action_taken, 0) + 1

    processed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    summary = DailyCycleResult(
        cycle_start_utc=cycle_start.isoformat(),
        cycle_end_utc=cycle_end.isoformat(),
        total_tickers_scanned=len(tickers),
        tickers_processed=processed,
        tickers_failed=failed,
        actions_taken=actions,
        total_execution_time_seconds=total_time,
        errors=errors,
    )

    logger.info("\n" + "=" * 70)
    logger.info("DAILY CYCLE COMPLETED @ %s", cycle_end.isoformat())
    logger.info("=" * 70)
    logger.info("Duration: %.1fs", total_time)
    logger.info("Tickers processed: %d/%d", processed, len(tickers))
    logger.info("Actions: %s", actions)
    if errors:
        logger.warning("Errors encountered: %d", len(errors))
    logger.info("=" * 70 + "\n")

    return summary


# ──────────────────────────────────────────────────────────
# SCHEDULER MODES
# ──────────────────────────────────────────────────────────


def start_scheduler(mode: str = "scheduled", **kwargs) -> None:
    """
    Start the scheduler in a given mode.

    Modes:
        - scheduled: Run daily at configured market open time.
        - once: Run a single cycle immediately.
        - ticker: Process a single ticker.
        - test: Run a small dry-run cycle.

    Args:
        mode: Execution mode.
        **kwargs: Mode-specific parameters.
    """
    logger.info("Starting ALPHA-PRIME Scheduler (mode=%s).", mode)

    if mode == "scheduled":
        market_open = getattr(settings, "market_open_time", "09:30")
        schedule.every().day.at(market_open).do(run_daily_cycle)

        logger.info(
            "Scheduled daily run at %s %s.",
            market_open,
            getattr(settings, "timezone", "IST"),
        )
        logger.info("Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
        return

    if mode == "once":
        logger.info("Running one-time daily cycle...")
        result = run_daily_cycle(
            limit=int(kwargs.get("limit", 10)),
            execute_trades=bool(kwargs.get("execute", True)),
        )
        logger.info(
            "One-time cycle completed: %d processed, actions=%s.",
            result.tickers_processed,
            result.actions_taken,
        )
        return

    if mode == "ticker":
        ticker = kwargs.get("ticker")
        if not ticker:
            logger.error("Ticker mode requires 'ticker' kwarg.")
            return

        trader = PaperTrader()
        logger.info("Processing single ticker: %s", ticker)
        result = process_ticker(
            ticker=ticker,
            trader=trader,
            execute_trades=bool(kwargs.get("execute", True)),
        )
        logger.info(
            "Single ticker result for %s: %s (msg=%s).",
            ticker,
            result.action_taken,
            result.message,
        )
        return

    if mode == "test":
        logger.info("Running TEST mode (dry-run, small sample)...")
        result = run_daily_cycle(limit=3, execute_trades=False)
        logger.info(
            "Test cycle completed: %d processed, actions=%s.",
            result.tickers_processed,
            result.actions_taken,
        )
        return

    logger.error("Unknown scheduler mode: %s", mode)


# ──────────────────────────────────────────────────────────
# CLI INTERFACE
# ──────────────────────────────────────────────────────────


def _print_usage() -> None:
    """Print CLI usage instructions."""
    print("\n" + "=" * 70)
    print("ALPHA-PRIME Automated Scheduler")
    print("=" * 70 + "\n")
    print("Usage:")
    print("  python scheduler.py scheduled          # Run daily at market open")
    print("  python scheduler.py once [--limit N]   # Run once immediately")
    print("  python scheduler.py ticker SYMBOL      # Process specific ticker")
    print("  python scheduler.py test               # Test mode (dry-run)")
    print("\nExamples:")
    print("  python scheduler.py scheduled")
    print("  python scheduler.py once --limit 5")
    print("  python scheduler.py ticker AAPL")
    print("  python scheduler.py test")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(0)

    mode = sys.argv[1].lower()
    kwargs: Dict[str, object] = {}

    if mode == "once":
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            try:
                kwargs["limit"] = int(sys.argv[idx + 1])
            except (IndexError, ValueError):
                print("Invalid or missing value for --limit.")
                sys.exit(1)

    elif mode == "ticker":
        if len(sys.argv) < 3:
            print("Error: Ticker symbol required.")
            print("Usage: python scheduler.py ticker SYMBOL")
            sys.exit(1)
        kwargs["ticker"] = sys.argv[2].upper()

    try:
        from config import validate_environment

        validate_environment()
    except Exception as exc:  # noqa: BLE001
        print(f"\nConfiguration error: {exc}\n")
        sys.exit(1)

    try:
        start_scheduler(mode, **kwargs)
    except KeyboardInterrupt:
        print("\nStopped by user.\n")
    except Exception as exc:  # noqa: BLE001
        logger.critical("Scheduler crashed: %s", exc, exc_info=True)
        send_error_alert(f"Scheduler crashed: {exc}", "CRITICAL")
        sys.exit(1)
