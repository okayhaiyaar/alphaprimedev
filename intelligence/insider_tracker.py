"""
============================================================
ALPHA-PRIME v2.0 - Insider Trading Tracker
============================================================

Tracks and analyzes corporate insider trading activity:

Why Track Insiders?
- Insiders have privileged information about company prospects. [web:285][web:288][web:291]
- Insider buying with personal capital is a strong bullish signal. [web:285][web:286]
- Insider selling is a weaker signal and often driven by liquidity needs or taxes.
- Clusters of insider activity are more informative than isolated trades. [web:289][web:292]
- Large purchases by key executives (CEO, CFO, Directors) carry disproportionate weight. [web:285][web:286]

Insider Transaction Types (Form 4 codes):
1. P (Purchase) - BULLISH:
   - Open market or private purchase of securities.
   - Strong confidence signal, especially with large notional value. [web:285][web:291]

2. S (Sale) - NEUTRAL/BEARISH:
   - Open market or private sale of securities.
   - Many legitimate reasons, but cluster selling or very large sales can be negative. [web:285][web:291]

3. A (Award) - NEUTRAL:
   - Grant/award of shares as compensation.
   - Generally not used as a directional sentiment signal. [web:285][web:286]

4. M (Option Exercise) - NEUTRAL:
   - Exercise or conversion of derivative securities.
   - Often followed by sales but not inherently bearish. [web:285][web:286]

Insider Roles (weighted by importance):
1. CEO (Chief Executive Officer) - Highest weight.
2. CFO (Chief Financial Officer) - High weight.
3. Director - High weight.
4. 10% Owner (Large shareholder) - Medium-high weight.
5. President/COO - Medium weight.
6. VP (Vice President) - Lower weight.
7. Other Officers - Lowest weight.

Scoring Rules:
1. Buying is more informative than selling (asymmetric weighting).
2. CEO/CFO/Director buys are up-weighted ~3x vs baseline.
3. Purchases with notional > 100k USD receive a size bonus.
4. Multiple insiders buying within ~30 days receive a cluster bonus.
5. Selling only becomes strongly bearish when a large fraction of holdings is sold or when multiple insiders sell.

Insider Sentiment Score (-100 to +100):
- +75 to +100: VERY_BULLISH (cluster insider buying).
- +50 to +75: BULLISH (significant net insider buying).
- +25 to +50: MODERATELY_BULLISH.
- -25 to +25: NEUTRAL.
- -50 to -25: MODERATELY_BEARISH.
- -75 to -50: BEARISH (unusual selling behavior).
- -100 to -75: VERY_BEARISH (panic-like selling).

Usage:
    from intelligence.insider_tracker import (
        get_insider_activity,
        calculate_insider_score,
        get_insider_summary,
        detect_unusual_activity,
    )

    tx = get_insider_activity("AAPL", months_back=6)
    score = calculate_insider_score("AAPL")
    summary = get_insider_summary("AAPL")
    alerts = detect_unusual_activity("AAPL")

Integration:
- research_engine.py consumes insider sentiment as an alpha feature.
- brain.py can gate or boost trades based on insider activity.
- Dashboard surfaces summaries and unusual-activity alerts.
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
from bs4 import BeautifulSoup  # noqa: F401  (reserved for future SEC parsing)
from diskcache import Cache

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()
cache = Cache(f"{settings.cache_dir}/insider")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & ENUMS
# ──────────────────────────────────────────────────────────


class TransactionType(str, Enum):
    """Insider transaction types (Form 4 codes P/S/A/M and other). [web:285][web:291]"""

    PURCHASE = "P"
    SALE = "S"
    AWARD = "A"
    EXERCISE = "M"
    OTHER = "O"


class InsiderRole(str, Enum):
    """Standardized insider role/title categories."""

    CEO = "CEO"
    CFO = "CFO"
    DIRECTOR = "DIRECTOR"
    OWNER_10PCT = "10% OWNER"
    PRESIDENT = "PRESIDENT"
    COO = "COO"
    VP = "VP"
    OFFICER = "OFFICER"
    OTHER = "OTHER"


@dataclass
class InsiderTransaction:
    """
    Individual insider transaction (Form 4 item).

    Attributes:
        ticker: Underlying symbol.
        insider_name: Name of reporting insider.
        insider_role: Normalized role classification.
        transaction_type: TransactionType (P, S, A, M, O).
        transaction_date: Execution date of the transaction (UTC).
        shares: Number of shares transacted (absolute).
        price_per_share: Transaction price per share.
        total_value: Monetary notional (shares × price_per_share).
        shares_owned_after: Shares beneficially owned after the transaction.
        percent_of_holdings_traded: Optional % of previous holdings traded.
        filing_date: Date of Form 4 filing if known.
        form_type: Filing form identifier (default "4").
    """

    ticker: str
    insider_name: str
    insider_role: InsiderRole
    transaction_type: TransactionType
    transaction_date: datetime
    shares: int
    price_per_share: float
    total_value: float
    shares_owned_after: int
    percent_of_holdings_traded: Optional[float] = None
    filing_date: Optional[datetime] = None
    form_type: str = "4"


@dataclass
class InsiderSentiment:
    """
    Aggregated insider sentiment for a ticker.

    Attributes:
        ticker: Symbol.
        sentiment_score: Score in [-100, 100].
        sentiment_label: Qualitative label.
        total_purchases: Count of purchase transactions.
        total_sales: Count of sale transactions.
        net_shares: Shares bought minus shares sold.
        net_value: Dollar value of net insider activity.
        buy_sell_ratio: Purchases / Sales count (inf if no sales).
        significant_buyers: Names of high-importance insiders buying.
        confidence: Data-quality confidence score (0–100).
        analysis_period_days: Span between earliest and latest tx.
        last_updated_utc: ISO timestamp of computation.
    """

    ticker: str
    sentiment_score: float
    sentiment_label: str
    total_purchases: int
    total_sales: int
    net_shares: int
    net_value: float
    buy_sell_ratio: float
    significant_buyers: List[str]
    confidence: float
    analysis_period_days: int
    last_updated_utc: str


@dataclass
class InsiderAlert:
    """
    Alert representing unusual insider activity.

    Attributes:
        ticker: Symbol.
        alert_type: e.g. "CLUSTER_BUYING", "EXECUTIVE_BUYING", "LARGE_SALE".
        severity: "HIGH", "MEDIUM", or "LOW".
        description: Human-readable description.
        transactions: Underlying transactions that triggered alert.
        detected_at_utc: ISO timestamp.
    """

    ticker: str
    alert_type: str
    severity: str
    description: str
    transactions: List[InsiderTransaction]
    detected_at_utc: str


@dataclass
class InsiderSummary:
    """
    Summary roll-up of insider activity.

    Attributes:
        ticker: Symbol.
        total_transactions: Total transaction count.
        total_buys: Count of purchase transactions.
        total_sells: Count of sale transactions.
        total_buy_value: Aggregate notional value of buys.
        total_sell_value: Aggregate notional value of sells.
        unique_insiders: Number of unique insiders.
        recent_activity_30d: Number of transactions in last 30 days.
        insider_score: Sentiment score [-100, 100].
        sentiment: Label matching InsiderSentiment.sentiment_label.
    """

    ticker: str
    total_transactions: int
    total_buys: int
    total_sells: int
    total_buy_value: float
    total_sell_value: float
    unique_insiders: int
    recent_activity_30d: int
    insider_score: float
    sentiment: str


@dataclass
class InsiderTrends:
    """
    Trend-oriented view of insider activity over time.

    Attributes:
        ticker: Symbol.
        period: Requested period string (e.g. "6M", "1Y").
        monthly_buy_counts: Mapping YYYY-MM → count of purchases.
        monthly_sell_counts: Mapping YYYY-MM → count of sales.
        rolling_net_value: Mapping YYYY-MM → net notional (buy - sell).
        rolling_score: Mapping YYYY-MM → approximate sentiment score.
        generated_at_utc: ISO timestamp of trend computation.
    """

    ticker: str
    period: str
    monthly_buy_counts: Dict[str, int]
    monthly_sell_counts: Dict[str, int]
    rolling_net_value: Dict[str, float]
    rolling_score: Dict[str, float]
    generated_at_utc: str


# ──────────────────────────────────────────────────────────
# INSIDER DATA FETCHING
# ──────────────────────────────────────────────────────────


def fetch_insider_data_sec(ticker: str, months_back: int = 6) -> List[InsiderTransaction]:
    """
    Fetch insider transactions from SEC EDGAR Form 4 filings (placeholder). [web:288][web:291][web:294]

    In a production deployment this function should:
        - Query EDGAR ownership API or RSS feeds.
        - Download recent Form 4 XML/HTML.
        - Parse transaction tables and map codes P/S/A/M/… into TransactionType.

    Args:
        ticker: Stock symbol.
        months_back: Number of months of history to retrieve.

    Returns:
        List of InsiderTransaction (currently empty; yfinance is primary source).
    """
    logger.info("Fetching SEC insider data for %s (not yet implemented).", ticker)
    logger.warning("SEC EDGAR parsing not implemented - returning empty list for %s.", ticker)
    return []


def _infer_transaction_type(code: str) -> TransactionType:
    """Map raw transaction code string to TransactionType."""
    code = (code or "").upper().strip()
    if "PURCHASE" in code or code == "P":
        return TransactionType.PURCHASE
    if "SALE" in code or code == "S":
        return TransactionType.SALE
    if "AWARD" in code or code == "A":
        return TransactionType.AWARD
    if "EXERCISE" in code or code == "M":
        return TransactionType.EXERCISE
    return TransactionType.OTHER


def _infer_role(title: str) -> InsiderRole:
    """Normalize insider title/position string to InsiderRole."""
    t = (title or "").upper()
    if "CEO" in t or "CHIEF EXECUTIVE" in t:
        return InsiderRole.CEO
    if "CFO" in t or "CHIEF FINANCIAL" in t:
        return InsiderRole.CFO
    if "DIRECTOR" in t or "BOARD MEMBER" in t:
        return InsiderRole.DIRECTOR
    if "10%" in t or "TEN PERCENT" in t or "OWNER" in t:
        return InsiderRole.OWNER_10PCT
    if "PRESIDENT" in t:
        return InsiderRole.PRESIDENT
    if "COO" in t or "CHIEF OPERATING" in t:
        return InsiderRole.COO
    if "VICE PRESIDENT" in t or "VP" in t:
        return InsiderRole.VP
    if "OFFICER" in t:
        return InsiderRole.OFFICER
    return InsiderRole.OTHER


def fetch_insider_data_yfinance(ticker: str) -> List[InsiderTransaction]:
    """
    Fetch insider transactions using yfinance's insider_transactions field. [web:284][web:287][web:293]

    yfinance exposes a DataFrame `insider_transactions` with columns such as:
        - 'Insider', 'Position', 'Transaction', 'Start Date', 'Shares',
          'Value', 'Ownership' (subject to Yahoo Finance schema).

    Args:
        ticker: Stock symbol.

    Returns:
        List of InsiderTransaction objects.
    """
    cache_key = f"insider_yf_{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Insider yfinance cache hit for %s.", ticker)
        return cached

    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.insider_transactions  # type: ignore[attr-defined]
        if df is None or df.empty:
            logger.warning("No insider data from yfinance for %s.", ticker)
            cache.set(cache_key, [], expire=24 * 60 * 60)
            return []

        transactions: List[InsiderTransaction] = []

        for _, row in df.iterrows():
            trans_type = _infer_transaction_type(str(row.get("Transaction", "")))
            role = _infer_role(str(row.get("Position", "")))
            shares_raw = row.get("Shares", 0)
            try:
                shares = abs(int(shares_raw))
            except Exception:  # noqa: BLE001
                shares = 0

            value_raw = row.get("Value", 0)
            try:
                value = float(value_raw)
            except Exception:  # noqa: BLE001
                value = 0.0

            price = value / shares if shares > 0 else 0.0
            try:
                date_val = row.get("Start Date", None)
                trans_dt = pd.to_datetime(date_val).to_pydatetime()
                if trans_dt.tzinfo is None:
                    trans_dt = trans_dt.replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                trans_dt = datetime.now(timezone.utc)

            try:
                owned_after = int(row.get("Ownership", 0))
            except Exception:  # noqa: BLE001
                owned_after = 0

            percent_traded: Optional[float]
            if owned_after > 0 and trans_type in (TransactionType.SALE, TransactionType.PURCHASE):
                prev_holdings = owned_after + shares if trans_type == TransactionType.SALE else max(
                    owned_after - shares, 1
                )
                percent_traded = float(shares) / float(prev_holdings) * 100.0
            else:
                percent_traded = None

            tx = InsiderTransaction(
                ticker=ticker.upper(),
                insider_name=str(row.get("Insider", "Unknown")),
                insider_role=role,
                transaction_type=trans_type,
                transaction_date=trans_dt,
                shares=shares,
                price_per_share=price,
                total_value=shares * price,
                shares_owned_after=owned_after,
                percent_of_holdings_traded=percent_traded,
                filing_date=None,
            )
            transactions.append(tx)

        logger.info(
            "Fetched %d insider transaction(s) for %s via yfinance.",
            len(transactions),
            ticker,
        )
        cache.set(cache_key, transactions, expire=24 * 60 * 60)
        return transactions
    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching insider data from yfinance for %s: %s", ticker, exc, exc_info=True)
        cache.set(cache_key, [], expire=6 * 60 * 60)
        return []


def get_insider_activity(ticker: str, months_back: int = 6) -> List[InsiderTransaction]:
    """
    Main entry point to retrieve recent insider transactions for a ticker.

    Combines primary (yfinance) and optional SEC/other sources in future.

    Args:
        ticker: Stock symbol.
        months_back: History window in months.

    Returns:
        Filtered list of InsiderTransaction within the lookback window.
    """
    logger.info("Getting insider activity for %s (%d months back).", ticker, months_back)

    transactions = fetch_insider_data_yfinance(ticker)
    # Additional sources (e.g., SEC or OpenInsider) could be merged here.

    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    filtered = [t for t in transactions if t.transaction_date >= cutoff]

    logger.info(
        "Found %d insider transaction(s) for %s within last %d months.",
        len(filtered),
        ticker,
        months_back,
    )
    return filtered


# ──────────────────────────────────────────────────────────
# INSIDER SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────


def get_role_weight(role: InsiderRole) -> float:
    """
    Return importance weight for a given insider role.

    CEO/CFO/Directors receive the highest weights, reflecting their informational edge. [web:286][web:289]

    Args:
        role: InsiderRole.

    Returns:
        Weight multiplier (around 1–3).
    """
    weights: Dict[InsiderRole, float] = {
        InsiderRole.CEO: 3.0,
        InsiderRole.CFO: 3.0,
        InsiderRole.DIRECTOR: 2.5,
        InsiderRole.OWNER_10PCT: 2.0,
        InsiderRole.PRESIDENT: 1.5,
        InsiderRole.COO: 1.5,
        InsiderRole.VP: 1.2,
        InsiderRole.OFFICER: 1.0,
        InsiderRole.OTHER: 1.0,
    }
    return weights.get(role, 1.0)


def analyze_insider_sentiment(transactions: List[InsiderTransaction]) -> InsiderSentiment:
    """
    Compute insider sentiment score and related metrics from transactions.

    Heuristics:
        - Purchases (P) contribute positively with role and size weighting.
        - Sales (S) contribute negatively but with lower magnitude.
        - Option exercises (M) and awards (A) are treated as neutral for score.
        - Cluster of ≥3 separate purchasers boosts bullish bias.

    Args:
        transactions: List of InsiderTransaction.

    Returns:
        InsiderSentiment object summarizing sentiment.
    """
    if not transactions:
        return InsiderSentiment(
            ticker="UNKNOWN",
            sentiment_score=0.0,
            sentiment_label="NEUTRAL",
            total_purchases=0,
            total_sales=0,
            net_shares=0,
            net_value=0.0,
            buy_sell_ratio=0.0,
            significant_buyers=[],
            confidence=0.0,
            analysis_period_days=0,
            last_updated_utc=datetime.now(timezone.utc).isoformat(),
        )

    ticker = transactions[0].ticker
    purchases = [t for t in transactions if t.transaction_type == TransactionType.PURCHASE]
    sales = [t for t in transactions if t.transaction_type == TransactionType.SALE]

    buy_score = 0.0
    sell_score = 0.0

    for p in purchases:
        role_weight = get_role_weight(p.insider_role)
        value_score = min(p.total_value / 100_000.0, 3.0)
        base = 10.0 + value_score * 5.0
        buy_score += role_weight * base

    for s in sales:
        role_weight = get_role_weight(s.insider_role)
        base = 5.0
        sell_score += role_weight * base * 0.3

    if buy_score + sell_score > 0:
        sentiment_score = (buy_score - sell_score) / (buy_score + sell_score) * 100.0
    else:
        sentiment_score = 0.0

    unique_buyers = {t.insider_name for t in purchases}
    if len(unique_buyers) >= 3:
        sentiment_score = min(100.0, sentiment_score + 15.0)

    if sentiment_score >= 75.0:
        label = "VERY_BULLISH"
    elif sentiment_score >= 50.0:
        label = "BULLISH"
    elif sentiment_score >= 25.0:
        label = "MODERATELY_BULLISH"
    elif sentiment_score >= -25.0:
        label = "NEUTRAL"
    elif sentiment_score >= -50.0:
        label = "MODERATELY_BEARISH"
    elif sentiment_score >= -75.0:
        label = "BEARISH"
    else:
        label = "VERY_BEARISH"

    total_buys = len(purchases)
    total_sells = len(sales)
    net_shares = sum(t.shares for t in purchases) - sum(t.shares for t in sales)
    net_value = sum(t.total_value for t in purchases) - sum(t.total_value for t in sales)
    buy_sell_ratio = total_buys / total_sells if total_sells > 0 else float("inf") if total_buys > 0 else 0.0

    significant_buyers = sorted(
        {t.insider_name for t in purchases if t.insider_role in (InsiderRole.CEO, InsiderRole.CFO, InsiderRole.DIRECTOR)}
    )

    confidence = min(100.0, len(transactions) * 10.0)

    dates = [t.transaction_date for t in transactions]
    period_days = (max(dates) - min(dates)).days if dates else 0

    sentiment = InsiderSentiment(
        ticker=ticker,
        sentiment_score=float(sentiment_score),
        sentiment_label=label,
        total_purchases=total_buys,
        total_sales=total_sells,
        net_shares=int(net_shares),
        net_value=float(net_value),
        buy_sell_ratio=float(buy_sell_ratio),
        significant_buyers=significant_buyers,
        confidence=float(confidence),
        analysis_period_days=int(period_days),
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Insider sentiment %s: %s (%.1f), buys=%d, sells=%d.",
        ticker,
        label,
        sentiment_score,
        total_buys,
        total_sells,
    )
    return sentiment


def calculate_insider_score(ticker: str, months_back: int = 6) -> float:
    """
    Convenience function to compute insider sentiment score for a ticker.

    Args:
        ticker: Stock symbol.
        months_back: History window in months.

    Returns:
        Sentiment score in range [-100, 100].
    """
    tx = get_insider_activity(ticker, months_back=months_back)
    sentiment = analyze_insider_sentiment(tx)
    return sentiment.sentiment_score


# ──────────────────────────────────────────────────────────
# UNUSUAL ACTIVITY DETECTION
# ──────────────────────────────────────────────────────────


def detect_unusual_activity(ticker: str, months_back: int = 3) -> List[InsiderAlert]:
    """
    Detect notable insider patterns such as cluster buying and large sales.

    Currently implemented patterns:
        - CLUSTER_BUYING: ≥3 purchase transactions in last 30 days.
        - EXECUTIVE_BUYING: CEO/CFO purchases with notional > 100k USD.
        - LARGE_SALE: Single sale > 50% of previous holdings.

    Args:
        ticker: Stock symbol.
        months_back: Lookback window for underlying transactions.

    Returns:
        List of InsiderAlert objects.
    """
    tx = get_insider_activity(ticker, months_back=months_back)
    alerts: List[InsiderAlert] = []

    if not tx:
        logger.info("No insider transactions available for unusual-activity scan on %s.", ticker)
        return alerts

    now = datetime.now(timezone.utc)

    recent_cutoff = now - timedelta(days=30)
    recent_purchases = [
        t for t in tx if t.transaction_type == TransactionType.PURCHASE and t.transaction_date >= recent_cutoff
    ]
    if len(recent_purchases) >= 3:
        alerts.append(
            InsiderAlert(
                ticker=ticker.upper(),
                alert_type="CLUSTER_BUYING",
                severity="HIGH",
                description=f"{len(recent_purchases)} insider purchases in the last 30 days.",
                transactions=recent_purchases,
                detected_at_utc=now.isoformat(),
            )
        )

    exec_purchases = [
        t
        for t in tx
        if t.transaction_type == TransactionType.PURCHASE
        and t.insider_role in (InsiderRole.CEO, InsiderRole.CFO)
        and t.total_value > 100_000.0
    ]
    for p in exec_purchases:
        alerts.append(
            InsiderAlert(
                ticker=ticker.upper(),
                alert_type="EXECUTIVE_BUYING",
                severity="HIGH",
                description=f"{p.insider_role.value} {p.insider_name} bought ${p.total_value:,.0f} of stock.",
                transactions=[p],
                detected_at_utc=now.isoformat(),
            )
        )

    for t_item in tx:
        if (
            t_item.transaction_type == TransactionType.SALE
            and t_item.percent_of_holdings_traded is not None
            and t_item.percent_of_holdings_traded > 50.0
        ):
            alerts.append(
                InsiderAlert(
                    ticker=ticker.upper(),
                    alert_type="LARGE_SALE",
                    severity="MEDIUM",
                    description=(
                        f"{t_item.insider_name} sold {t_item.percent_of_holdings_traded:.0f}% "
                        f"of their holdings."
                    ),
                    transactions=[t_item],
                    detected_at_utc=now.isoformat(),
                )
            )

    logger.info("Detected %d insider alert(s) for %s.", len(alerts), ticker)
    return alerts


# ──────────────────────────────────────────────────────────
# SUMMARY & TREND FUNCTIONS
# ──────────────────────────────────────────────────────────


def get_insider_summary(ticker: str, months_back: int = 6) -> InsiderSummary:
    """
    Generate a summary view of insider activity and sentiment.

    Args:
        ticker: Stock symbol.
        months_back: Lookback window in months.

    Returns:
        InsiderSummary instance consolidating key metrics.
    """
    tx = get_insider_activity(ticker, months_back=months_back)
    sentiment = analyze_insider_sentiment(tx)

    buys = [t for t in tx if t.transaction_type == TransactionType.PURCHASE]
    sells = [t for t in tx if t.transaction_type == TransactionType.SALE]

    buy_value = sum(t.total_value for t in buys)
    sell_value = sum(t.total_value for t in sells)

    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent_activity_30d = sum(1 for t in tx if t.transaction_date >= recent_cutoff)

    unique_insiders = len({t.insider_name for t in tx})

    summary = InsiderSummary(
        ticker=ticker.upper(),
        total_transactions=len(tx),
        total_buys=len(buys),
        total_sells=len(sells),
        total_buy_value=float(buy_value),
        total_sell_value=float(sell_value),
        unique_insiders=unique_insiders,
        recent_activity_30d=recent_activity_30d,
        insider_score=float(sentiment.sentiment_score),
        sentiment=sentiment.sentiment_label,
    )

    logger.info(
        "Insider summary %s: tx=%d, buys=%d, sells=%d, score=%.1f (%s).",
        ticker,
        summary.total_transactions,
        summary.total_buys,
        summary.total_sells,
        summary.insider_score,
        summary.sentiment,
    )
    return summary


def _months_to_days(period: str) -> int:
    """Convert simple period strings like '3M', '6M', '1Y' to approx. days."""
    p = period.strip().upper()
    if p.endswith("Y"):
        years = float(p[:-1] or 1)
        return int(years * 365)
    if p.endswith("M"):
        months = float(p[:-1] or 1)
        return int(months * 30)
    try:
        return int(p)
    except Exception:  # noqa: BLE001
        return 180


def track_insider_trends(ticker: str, period: str = "6M") -> InsiderTrends:
    """
    Build coarse monthly trends for insider activity.

    Args:
        ticker: Stock symbol.
        period: Window spec like "3M", "6M", "1Y".

    Returns:
        InsiderTrends object with per-month aggregates.
    """
    days = _months_to_days(period)
    months_back = max(1, days // 30)
    tx = get_insider_activity(ticker, months_back=months_back)

    monthly_buy_counts: Dict[str, int] = {}
    monthly_sell_counts: Dict[str, int] = {}
    monthly_net_value: Dict[str, float] = {}

    for t in tx:
        key = t.transaction_date.strftime("%Y-%m")
        if t.transaction_type == TransactionType.PURCHASE:
            monthly_buy_counts[key] = monthly_buy_counts.get(key, 0) + 1
            monthly_net_value[key] = monthly_net_value.get(key, 0.0) + t.total_value
        elif t.transaction_type == TransactionType.SALE:
            monthly_sell_counts[key] = monthly_sell_counts.get(key, 0) + 1
            monthly_net_value[key] = monthly_net_value.get(key, 0.0) - t.total_value

    rolling_score: Dict[str, float] = {}
    for key in monthly_net_value:
        net_val = monthly_net_value[key]
        num_b = monthly_buy_counts.get(key, 0)
        num_s = monthly_sell_counts.get(key, 0)
        if num_b + num_s == 0:
            rolling_score[key] = 0.0
        else:
            raw = net_val / max(abs(net_val), 1.0) * 50.0
            rolling_score[key] = float(raw)

    trends = InsiderTrends(
        ticker=ticker.upper(),
        period=period,
        monthly_buy_counts=monthly_buy_counts,
        monthly_sell_counts=monthly_sell_counts,
        rolling_net_value=monthly_net_value,
        rolling_score=rolling_score,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("Generated insider trends for %s over period %s.", ticker, period)
    return trends


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Insider Trading Tracker")
    print("=" * 70 + "\n")

    if len(sys.argv) < 2:
        print("Usage: python insider_tracker.py TICKER [MONTHS]")
        print("Example: python insider_tracker.py AAPL 6\n")
        sys.exit(1)

    try:
        ticker_cli = sys.argv[1].upper()
        months_cli = int(sys.argv[2]) if len(sys.argv) > 2 else 6

        print(f"Analyzing insider activity for {ticker_cli} (last {months_cli} months)...\n")

        transactions_cli = get_insider_activity(ticker_cli, months_cli)

        if not transactions_cli:
            print(f"❌ No insider data available for {ticker_cli}")
            sys.exit(0)

        summary_cli = get_insider_summary(ticker_cli, months_cli)

        print("INSIDER SUMMARY")
        print("-" * 70)
        print(f"Total Transactions : {summary_cli.total_transactions}")
        print(f"Buys              : {summary_cli.total_buys} (${summary_cli.total_buy_value:,.0f})")
        print(f"Sells             : {summary_cli.total_sells} (${summary_cli.total_sell_value:,.0f})")
        print(f"Unique Insiders   : {summary_cli.unique_insiders}")
        print(f"Recent Activity 30d: {summary_cli.recent_activity_30d}")
        print(f"\nInsider Score    : {summary_cli.insider_score:.0f}/100")
        print(f"Sentiment        : {summary_cli.sentiment}\n")

        print("RECENT TRANSACTIONS")
        print("-" * 70)
        recent_sorted = sorted(transactions_cli, key=lambda x: x.transaction_date, reverse=True)[:10]
        for t in recent_sorted:
            icon = "🟢" if t.transaction_type == TransactionType.PURCHASE else "🔴"
            print(f"{icon} {t.transaction_date.date()} | {t.insider_name[:30]:30}")
            print(f"   {t.insider_role.value:15} | {t.transaction_type.value}")
            print(
                f"   {t.shares:,} shares @ ${t.price_per_share:.2f} "
                f"= ${t.total_value:,.0f}"
            )
            if t.percent_of_holdings_traded is not None:
                print(f"   {t.percent_of_holdings_traded:.0f}% of holdings traded")
            print()

        alerts_cli = detect_unusual_activity(ticker_cli, months_cli)
        if alerts_cli:
            print("\n🚨 UNUSUAL ACTIVITY ALERTS")
            print("-" * 70)
            for a in alerts_cli:
                icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(a.severity, "⚪")
                print(f"{icon} {a.alert_type}: {a.description}\n")

        print("=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Error running insider tracker: {exc}")
        traceback.print_exc()
