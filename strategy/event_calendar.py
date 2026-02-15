"""
============================================================
ALPHA-PRIME v2.0 - Event Calendar (Strategy Module)
============================================================

Tracks upcoming high-impact events to avoid volatile periods:

Why Event Awareness Matters:
- Earnings surprises can cause 10-20% moves in minutes.
- Fed announcements can swing entire indices several percent intraday. [web:225][web:233]
- Economic data such as CPI and NFP drives sharp repricing in rates and equities. [web:225][web:231]
- Trading through events without preparation is closer to gambling than systematic trading.

Event Types & Risk:
1. EARNINGS (HIGH RISK):
   - Blackout: 1 day before to 1 day after.
   - Avoid new positions entirely.
   - Consider reducing or hedging existing positions.

2. FOMC MEETINGS (HIGH RISK):
   - Federal Reserve rate decisions.
   - Blackout: Day of meeting.
   - Market-wide volatility in equities, rates, and FX.

3. ECONOMIC DATA (MEDIUM/HIGH RISK):
   - CPI (inflation), NFP (jobs), GDP, PCE.
   - Blackout: Day of release for high-importance prints.
   - Sector and market-wide impact.

4. DIVIDENDS (LOW RISK):
   - Ex-dividend date (price adjusts down).
   - Monitor for gap risk; usually no full blackout.

5. STOCK SPLITS (LOW RISK):
   - Price adjustment events.
   - Monitor for structural changes; no blackout.

Trading Rules:
1. HIGH RISK events → Block trades inside blackout window.
2. MEDIUM RISK → Reduce position size (e.g. 50%) or require higher confidence.
3. LOW RISK → Proceed with awareness.
4. No events → Normal operation.

Example:
    AAPL earnings on Jan 30 (after market close)

    Jan 29: ❌ No new AAPL positions (1 day before).
    Jan 30: ❌ No trading AAPL (earnings day).
    Jan 31: ❌ No trading AAPL (1 day after, digest reaction).
    Feb 01: ✅ Resume trading.

Usage:
    from strategy.event_calendar import check_event_risk

    result = check_event_risk("AAPL", "BUY")
    if not result.allowed:
        logger.warning("Event risk: %s", result.message)

Integration:
- Called by scheduler before processing tickers.
- Passed to brain.py for context.
- Displayed on dashboard and used for alerting.
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from diskcache import Cache

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()
cache = Cache(f"{settings.cache_dir}/events")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & ENUMS
# ──────────────────────────────────────────────────────────


class EventType(str, Enum):
    """Event type classification."""

    EARNINGS = "EARNINGS"
    ECONOMIC_DATA = "ECONOMIC_DATA"
    FOMC_MEETING = "FOMC_MEETING"
    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    FDA_APPROVAL = "FDA_APPROVAL"
    OTHER = "OTHER"


class RiskLevel(str, Enum):
    """Event risk level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Event:
    """
    Generic event representation.

    Attributes:
        event_type: Category of event (earnings, FOMC, etc.).
        title: Human-readable event title.
        date: Event timestamp in UTC.
        risk_level: RiskLevel classification.
        ticker: Optional symbol, if company-specific.
        description: Additional details.
        blackout_days_before: Blackout window start offset in days.
        blackout_days_after: Blackout window end offset in days.
    """

    event_type: EventType
    title: str
    date: datetime
    risk_level: RiskLevel
    ticker: Optional[str] = None
    description: str = ""
    blackout_days_before: int = 1
    blackout_days_after: int = 1


@dataclass
class EarningsEvent(Event):
    """
    Earnings-specific event.

    Attributes extend Event with:
        quarter: Fiscal quarter (Q1–Q4).
        fiscal_year: Fiscal year.
        estimated_eps: Analyst EPS estimate if available.
        estimated_revenue: Revenue estimate if available.
        time_of_day: "BMO" (before open), "AMC" (after close), "UNKNOWN".
    """

    quarter: str = "Q1"
    fiscal_year: int = 1970
    estimated_eps: Optional[float] = None
    estimated_revenue: Optional[float] = None
    time_of_day: str = "UNKNOWN"


@dataclass
class EconomicEvent(Event):
    """
    Economic data release event.

    Attributes extend Event with:
        indicator: Identifier (CPI, NFP, GDP, PCE, etc.).
        country: Country code (default "US").
        previous_value: Previous release value.
        forecast_value: Consensus forecast.
        actual_value: Actual release value when known.
    """

    indicator: str = ""
    country: str = "US"
    previous_value: Optional[float] = None
    forecast_value: Optional[float] = None
    actual_value: Optional[float] = None


@dataclass
class EventCheckResult:
    """
    Result of event risk check.

    Attributes:
        allowed: Whether trading is allowed given active events.
        events_found: List of events considered in the decision.
        highest_risk_level: Highest RiskLevel among relevant events.
        message: Human-readable explanation.
    """

    allowed: bool
    events_found: List[Event]
    highest_risk_level: RiskLevel
    message: str


# ──────────────────────────────────────────────────────────
# EARNINGS CALENDAR
# ──────────────────────────────────────────────────────────


def fetch_earnings_calendar(ticker: str) -> Optional[EarningsEvent]:
    """
    Fetch next earnings date for a ticker using yfinance. [web:222][web:228][web:232]

    Args:
        ticker: Stock symbol.

    Returns:
        EarningsEvent for next upcoming earnings, or None if not found.
    """
    cache_key = f"earnings_{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached:
        logger.debug("Earnings cache hit for %s.", ticker)
        return cached

    try:
        yf_ticker = yf.Ticker(ticker)
        earnings_df = yf_ticker.get_earnings_dates(limit=12)
        if earnings_df is None or earnings_df.empty:
            logger.debug("No earnings dates returned for %s.", ticker)
            return None

        now = datetime.now(timezone.utc)

        for idx, row in earnings_df.iterrows():
            earnings_date = pd.to_datetime(idx)
            if earnings_date.tzinfo is None:
                earnings_date = earnings_date.replace(tzinfo=timezone.utc)

            if earnings_date <= now:
                continue

            quarter = f"Q{(earnings_date.month - 1) // 3 + 1}"
            est_eps = row.get("EPS Estimate", None)
            est_rev = row.get("Revenue Estimate", None)

            event = EarningsEvent(
                event_type=EventType.EARNINGS,
                title=f"{ticker.upper()} Earnings Report",
                date=earnings_date,
                risk_level=RiskLevel.HIGH,
                ticker=ticker.upper(),
                description=f"{ticker.upper()} quarterly earnings release.",
                blackout_days_before=1,
                blackout_days_after=1,
                quarter=quarter,
                fiscal_year=earnings_date.year,
                estimated_eps=float(est_eps) if pd.notna(est_eps) else None,
                estimated_revenue=float(est_rev) if pd.notna(est_rev) else None,
                time_of_day="UNKNOWN",
            )

            logger.info(
                "Next earnings for %s on %s (quarter %s, FY %d).",
                ticker,
                earnings_date.date(),
                quarter,
                earnings_date.year,
            )
            cache.set(cache_key, event, expire=4 * 60 * 60)
            return event

        logger.debug("No future earnings dates found for %s.", ticker)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching earnings for %s: %s", ticker, exc, exc_info=True)
        return None


def get_earnings_countdown(event: EarningsEvent) -> int:
    """
    Compute days until an earnings event.

    Args:
        event: EarningsEvent instance.

    Returns:
        Integer day difference (negative if event is in the past).
    """
    now = datetime.now(timezone.utc)
    delta = event.date - now
    return delta.days


# ──────────────────────────────────────────────────────────
# ECONOMIC CALENDAR & FOMC
# ──────────────────────────────────────────────────────────


def _hardcoded_economic_releases() -> List[Dict[str, object]]:
    """
    Provide a minimal hardcoded list of major US economic releases.

    In production, this can be replaced or extended via an API such as
    FRED, TradingEconomics, or OpenBB's economic calendar. [web:225][web:231]

    Returns:
        List of dicts describing key events.
    """
    return [
        {
            "date": "2026-02-06",
            "indicator": "NFP",
            "title": "Non-Farm Payrolls",
            "description": "Monthly US jobs report.",
            "risk": RiskLevel.HIGH,
        },
        {
            "date": "2026-02-12",
            "indicator": "CPI",
            "title": "Consumer Price Index",
            "description": "Monthly US CPI inflation data.",
            "risk": RiskLevel.HIGH,
        },
        {
            "date": "2026-02-27",
            "indicator": "GDP",
            "title": "GDP Release",
            "description": "Quarterly US GDP release.",
            "risk": RiskLevel.MEDIUM,
        },
        {
            "date": "2026-02-28",
            "indicator": "PCE",
            "title": "PCE Inflation",
            "description": "Core PCE inflation (Fed's preferred gauge).",
            "risk": RiskLevel.HIGH,
        },
    ]


def fetch_economic_calendar(days_ahead: int = 7) -> List[EconomicEvent]:
    """
    Fetch upcoming economic events for the next N days.

    This implementation uses a hardcoded schedule for critical events
    (CPI, NFP, GDP, PCE) that can be replaced by an external API. [web:225][web:229]

    Args:
        days_ahead: Look-ahead window in days.

    Returns:
        List of EconomicEvent instances.
    """
    now_date = datetime.now(timezone.utc).date()
    cutoff = now_date + timedelta(days=days_ahead)
    events: List[EconomicEvent] = []

    for rel in _hardcoded_economic_releases():
        rel_date = datetime.strptime(rel["date"], "%Y-%m-%d").date()
        if now_date <= rel_date <= cutoff:
            dt = datetime.combine(rel_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            events.append(
                EconomicEvent(
                    event_type=EventType.ECONOMIC_DATA,
                    title=str(rel["title"]),
                    date=dt,
                    risk_level=rel["risk"],
                    ticker=None,
                    description=str(rel["description"]),
                    blackout_days_before=0,
                    blackout_days_after=0,
                    indicator=str(rel["indicator"]),
                    country="US",
                )
            )

    logger.info(
        "Found %d economic events in next %d days.", len(events), days_ahead
    )
    return events


def fetch_fomc_meetings(days_ahead: int = 30) -> List[Event]:
    """
    Fetch upcoming FOMC meeting dates for the next N days.

    FOMC schedule is maintained manually and should be updated yearly
    from the Federal Reserve website. [web:233]

    Args:
        days_ahead: Look-ahead window in days.

    Returns:
        List of Event instances (FOMC_MEETING).
    """
    fomc_dates = [
        "2026-01-28",
        "2026-03-17",
        "2026-04-29",
        "2026-06-16",
        "2026-07-28",
        "2026-09-15",
        "2026-10-27",
        "2026-12-15",
    ]

    now_date = datetime.now(timezone.utc).date()
    cutoff = now_date + timedelta(days=days_ahead)
    events: List[Event] = []

    for date_str in fomc_dates:
        meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if now_date <= meeting_date <= cutoff:
            dt = datetime.combine(meeting_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            )
            events.append(
                Event(
                    event_type=EventType.FOMC_MEETING,
                    title="FOMC Meeting",
                    date=dt,
                    risk_level=RiskLevel.HIGH,
                    ticker=None,
                    description="Federal Reserve interest rate decision.",
                    blackout_days_before=0,
                    blackout_days_after=0,
                )
            )

    logger.info(
        "Found %d FOMC meeting(s) in next %d days.", len(events), days_ahead
    )
    return events


# ──────────────────────────────────────────────────────────
# DIVIDENDS & SPLITS
# ──────────────────────────────────────────────────────────


def fetch_dividend_events(
    ticker: str,
    days_ahead: int = 30,
) -> List[Event]:
    """
    Estimate next ex-dividend date from historical dividend series.

    Uses yfinance's dividends series to infer approximate periodicity. [web:224][web:232]

    Args:
        ticker: Stock symbol.
        days_ahead: Look-ahead window in days.

    Returns:
        List containing at most one dividend Event.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        dividends = yf_ticker.dividends
        if dividends is None or dividends.empty:
            return []

        last_div_date = dividends.index[-1]
        next_div_date = last_div_date + pd.DateOffset(months=3)

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        next_div_dt = next_div_date.to_pydatetime().replace(tzinfo=timezone.utc)

        if now <= next_div_dt <= cutoff:
            event = Event(
                event_type=EventType.DIVIDEND,
                title=f"{ticker.upper()} Ex-Dividend Date",
                date=next_div_dt,
                risk_level=RiskLevel.LOW,
                ticker=ticker.upper(),
                description="Approximate next ex-dividend date based on dividend history.",
                blackout_days_before=0,
                blackout_days_after=0,
            )
            logger.info(
                "Estimated ex-dividend event for %s on %s.", ticker, next_div_dt.date()
            )
            return [event]
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("Error fetching dividends for %s: %s", ticker, exc)
        return []


def fetch_split_events(
    ticker: str,
    days_ahead: int = 30,
) -> List[Event]:
    """
    Fetch upcoming or recent stock split events from yfinance. [web:224][web:232][web:234]

    Args:
        ticker: Stock symbol.
        days_ahead: Look-ahead window in days.

    Returns:
        List of stock split Event instances.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        splits = yf_ticker.splits
        if splits is None or splits.empty:
            return []

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        events: List[Event] = []

        for idx, ratio in splits.items():
            split_dt = pd.to_datetime(idx).to_pydatetime().replace(tzinfo=timezone.utc)
            if now <= split_dt <= cutoff:
                event = Event(
                    event_type=EventType.STOCK_SPLIT,
                    title=f"{ticker.upper()} Stock Split ({ratio}:1)",
                    date=split_dt,
                    risk_level=RiskLevel.LOW,
                    ticker=ticker.upper(),
                    description=f"Stock split ratio: {ratio}:1.",
                    blackout_days_before=0,
                    blackout_days_after=0,
                )
                events.append(event)

        if events:
            logger.info("Found %d split event(s) for %s.", len(events), ticker)
        return events
    except Exception as exc:  # noqa: BLE001
        logger.debug("Error fetching splits for %s: %s", ticker, exc)
        return []


# ──────────────────────────────────────────────────────────
# EVENT AGGREGATION
# ──────────────────────────────────────────────────────────


def get_upcoming_events(
    ticker: str,
    days_ahead: int = 7,
) -> List[Event]:
    """
    Aggregate all upcoming events affecting a ticker and the broad market.

    Includes:
        - Earnings for the ticker.
        - Dividend and split events for the ticker.
        - Economic data and FOMC meetings (market-wide).

    Args:
        ticker: Stock symbol.
        days_ahead: Days ahead to search for events.

    Returns:
        Chronologically sorted list of Event/EarningsEvent/EconomicEvent.
    """
    logger.info(
        "Fetching upcoming events for %s within %d days ahead.", ticker, days_ahead
    )

    all_events: List[Event] = []

    earnings = fetch_earnings_calendar(ticker)
    if earnings is not None:
        days_until = get_earnings_countdown(earnings)
        if 0 <= days_until <= days_ahead:
            all_events.append(earnings)

    all_events.extend(fetch_dividend_events(ticker, days_ahead=days_ahead))
    all_events.extend(fetch_split_events(ticker, days_ahead=days_ahead))
    all_events.extend(fetch_economic_calendar(days_ahead=days_ahead))
    all_events.extend(fetch_fomc_meetings(days_ahead=days_ahead))

    all_events.sort(key=lambda e: e.date)

    logger.info(
        "Found %d event(s) for %s in the next %d days.",
        len(all_events),
        ticker,
        days_ahead,
    )
    return all_events


# ──────────────────────────────────────────────────────────
# BLACKOUT LOGIC & RISK CHECKING
# ──────────────────────────────────────────────────────────


def get_event_blackout_period(event: Event) -> Tuple[datetime, datetime]:
    """
    Compute the blackout window (start, end) for an event.

    High-risk events (earnings, FOMC) typically enforce blackout periods,
    while low-risk events (dividends, splits) often have zero blackout days.

    Args:
        event: Event instance.

    Returns:
        (blackout_start, blackout_end) datetimes in UTC.
    """
    start = event.date - timedelta(days=event.blackout_days_before)
    end = event.date + timedelta(days=event.blackout_days_after)
    return start, end


def _determine_highest_risk(events: List[Event]) -> RiskLevel:
    """Return the highest RiskLevel from a list of events."""
    if not events:
        return RiskLevel.LOW

    if any(e.risk_level == RiskLevel.HIGH for e in events):
        return RiskLevel.HIGH
    if any(e.risk_level == RiskLevel.MEDIUM for e in events):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def check_event_risk(
    ticker: str,
    action: str = "BUY",
    days_ahead: int = 3,
) -> EventCheckResult:
    """
    Evaluate whether trading a ticker is safe given upcoming events.

    Policy:
        - HIGH risk inside blackout → block trade.
        - MEDIUM risk events today → allowed but flagged.
        - LOW risk events → allow, but surface context.

    Args:
        ticker: Stock symbol.
        action: Planned action ("BUY"/"SELL").
        days_ahead: Look-ahead horizon for events.

    Returns:
        EventCheckResult with decision and explanation.
    """
    events = get_upcoming_events(ticker, days_ahead=days_ahead)
    if not events:
        return EventCheckResult(
            allowed=True,
            events_found=[],
            highest_risk_level=RiskLevel.LOW,
            message="No upcoming events detected.",
        )

    now = datetime.now(timezone.utc)
    active_events: List[Event] = []

    for event in events:
        start, end = get_event_blackout_period(event)
        if start <= now <= end:
            active_events.append(event)

    highest_risk = _determine_highest_risk(active_events or events)

    if active_events and highest_risk == RiskLevel.HIGH:
        titles = ", ".join(sorted({e.title for e in active_events}))
        msg = (
            f"❌ HIGH RISK EVENT(S): {titles}. "
            f"Trading {ticker.upper()} blocked during blackout window."
        )
        return EventCheckResult(
            allowed=False,
            events_found=active_events,
            highest_risk_level=RiskLevel.HIGH,
            message=msg,
        )

    upcoming_desc: List[str] = []
    for event in events[:5]:
        days_until = (event.date - now).days
        label = f"{event.title} (in {days_until} day{'s' if abs(days_until) != 1 else ''})"
        upcoming_desc.append(label)

    msg_prefix = "No blocking events. "
    if highest_risk == RiskLevel.MEDIUM:
        msg_prefix = "Caution: medium event risk. "

    message = msg_prefix + "Upcoming: " + "; ".join(upcoming_desc)
    return EventCheckResult(
        allowed=True,
        events_found=events,
        highest_risk_level=highest_risk,
        message=message,
    )


def should_avoid_trading(
    ticker: str,
    current_date: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Simple gate to decide if trading should be avoided today.

    Args:
        ticker: Stock symbol.
        current_date: Date to check (ignored in current implementation; uses now).

    Returns:
        (should_avoid, reason) tuple.
    """
    _ = current_date
    result = check_event_risk(ticker=ticker, action="BUY", days_ahead=3)
    if not result.allowed:
        return True, result.message
    return False, "No event-based restrictions detected."


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Event Calendar - Test Tool")
    print("=" * 70 + "\n")

    if len(sys.argv) < 2:
        print("Usage: python event_calendar.py TICKER [DAYS_AHEAD]")
        print("Example: python event_calendar.py AAPL 14\n")
        sys.exit(1)

    ticker_cli = sys.argv[1].upper()
    if len(sys.argv) > 2:
        try:
            days_ahead_cli = int(sys.argv[2])
        except ValueError:
            days_ahead_cli = 14
    else:
        days_ahead_cli = 14

    print(
        f"Checking events for {ticker_cli} in the next {days_ahead_cli} day(s)...\n"
    )

    try:
        events_cli = get_upcoming_events(ticker_cli, days_ahead_cli)

        if not events_cli:
            print(
                f"✅ No events found for {ticker_cli} in the next {days_ahead_cli} days."
            )
        else:
            print(f"Found {len(events_cli)} event(s):\n")
            print("-" * 70)
            now_cli = datetime.now(timezone.utc)

            for event in events_cli:
                days_until = (event.date - now_cli).days
                risk_icon = {
                    RiskLevel.HIGH: "🔴",
                    RiskLevel.MEDIUM: "🟡",
                    RiskLevel.LOW: "🟢",
                }.get(event.risk_level, "⚪")

                print(f"{risk_icon} {event.title}")
                print(f"   Date : {event.date.date()} ({days_until} days away)")
                print(f"   Risk : {event.risk_level.value}")
                print(f"   Type : {event.event_type.value}")
                if event.ticker:
                    print(f"   Ticker: {event.ticker}")
                if event.description:
                    print(f"   Desc : {event.description}")
                start, end = get_event_blackout_period(event)
                print(f"   Blackout: {start.date()} → {end.date()}")
                print()

        print("-" * 70)
        result_cli = check_event_risk(ticker_cli, "BUY", days_ahead=3)

        if result_cli.allowed:
            print("✅ TRADING ALLOWED")
            print(f"   {result_cli.message}")
        else:
            print("❌ TRADING BLOCKED")
            print(f"   {result_cli.message}")

        print("\n" + "=" * 70 + "\n")

    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error running event calendar: {exc}")
        traceback.print_exc()
