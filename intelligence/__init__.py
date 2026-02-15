"""
============================================================
ALPHA-PRIME v2.0 - Intelligence Module
============================================================

Organizes intelligence-gathering components:

Submodules:
1. news_aggregator:
   - Multi-source news (Alpha Vantage, NewsAPI, Finnhub)
   - Article summarization
   - Impact scoring

2. sec_parser:
   - SEC EDGAR filing retrieval (10-K, 10-Q, 8-K)
   - Risk factors extraction
   - Management discussion parsing
   - Financial table extraction

3. sentiment_analyzer:
   - Social media sentiment (Twitter, Reddit, StockTwits)
   - News headline sentiment
   - Aggregate sentiment scores
   - Bull/bear ratio

4. fundamental_analyzer:
   - Financial ratios (P/E, P/B, ROE, etc.)
   - Growth metrics (revenue, earnings growth)
   - Valuation analysis
   - Peer comparison

5. alternative_data (optional):
   - App download trends
   - Web traffic data
   - Satellite imagery (for retail, factories)
   - Credit card transaction data

Usage:
    # Clean imports
    from intelligence import get_comprehensive_intel

    intel = get_comprehensive_intel("AAPL")

    # Or import specific components
    from intelligence.news_aggregator import fetch_latest_news
    from intelligence.sentiment_analyzer import analyze_sentiment

    news = fetch_latest_news("AAPL", sources=["newsapi", "finnhub"])
    sentiment = analyze_sentiment("AAPL")

Integration:
- Called by research_engine.py (legacy unified interface).
- brain.py can call specific intelligence components.
- Dashboard displays intelligence breakdown.
============================================================
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "ALPHA-PRIME Team"

import logging
from typing import Any, Dict, List

from config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────
# SUBMODULE IMPORTS (with graceful degradation)
# ──────────────────────────────────────────────────────────

__all__: List[str] = [
    "__version__",
    "get_comprehensive_intel",
    "get_module_status",
    "list_available_modules",
]

_NEWS_AVAILABLE = False
_SEC_AVAILABLE = False
_SENTIMENT_AVAILABLE = False
_FUNDAMENTALS_AVAILABLE = False
_ALT_DATA_AVAILABLE = False

# News Aggregator
try:
    from .news_aggregator import (  # type: ignore[import]
        NewsAggregator,
        NewsArticle,
        aggregate_news_sources,
        fetch_latest_news,
        score_news_impact,
    )

    __all__.extend(
        [
            "fetch_latest_news",
            "aggregate_news_sources",
            "score_news_impact",
            "NewsArticle",
            "NewsAggregator",
        ]
    )
    _NEWS_AVAILABLE = True
except ImportError as exc:
    logger.warning("News aggregator not available: %s", exc)

# SEC Parser
try:
    from .sec_parser import (  # type: ignore[import]
        SECFiling,
        SECParser,
        extract_risk_factors,
        fetch_latest_filing,
        parse_sec_filing,
    )

    __all__.extend(
        [
            "parse_sec_filing",
            "fetch_latest_filing",
            "extract_risk_factors",
            "SECFiling",
            "SECParser",
        ]
    )
    _SEC_AVAILABLE = True
except ImportError as exc:
    logger.warning("SEC parser not available: %s", exc)

# Sentiment Analyzer
try:
    from .sentiment_analyzer import (  # type: ignore[import]
        SentimentAnalyzer,
        SentimentResult,
        analyze_sentiment,
        get_news_sentiment,
        get_social_sentiment,
    )

    __all__.extend(
        [
            "analyze_sentiment",
            "get_social_sentiment",
            "get_news_sentiment",
            "SentimentResult",
            "SentimentAnalyzer",
        ]
    )
    _SENTIMENT_AVAILABLE = True
except ImportError as exc:
    logger.warning("Sentiment analyzer not available: %s", exc)

# Fundamental Analyzer
try:
    from .fundamental_analyzer import (  # type: ignore[import]
        FundamentalAnalyzer,
        FundamentalMetrics,
        analyze_financial_health,
        calculate_valuation_ratios,
        get_fundamental_metrics,
    )

    __all__.extend(
        [
            "get_fundamental_metrics",
            "calculate_valuation_ratios",
            "analyze_financial_health",
            "FundamentalMetrics",
            "FundamentalAnalyzer",
        ]
    )
    _FUNDAMENTALS_AVAILABLE = True
except ImportError as exc:
    logger.warning("Fundamental analyzer not available: %s", exc)

# Alternative Data (optional)
try:
    from .alternative_data import (  # type: ignore[import]
        AlternativeDataSource,
        fetch_app_downloads,
        fetch_web_traffic,
        get_alternative_data,
    )

    __all__.extend(
        [
            "get_alternative_data",
            "fetch_app_downloads",
            "fetch_web_traffic",
            "AlternativeDataSource",
        ]
    )
    _ALT_DATA_AVAILABLE = True
except ImportError as exc:
    logger.debug("Alternative data module not available: %s", exc)


# ──────────────────────────────────────────────────────────
# UNIFIED INTELLIGENCE INTERFACE
# ──────────────────────────────────────────────────────────


def get_comprehensive_intel(
    ticker: str,
    include_news: bool = True,
    include_sec: bool = True,
    include_sentiment: bool = True,
    include_fundamentals: bool = True,
    include_alt_data: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve a comprehensive intelligence snapshot for a ticker.

    This function is a convenience façade that orchestrates calls into
    the available submodules (news, SEC, sentiment, fundamentals, alt-data)
    and aggregates results into a single dictionary.

    Args:
        ticker: Stock symbol.
        include_news: Whether to include news items and impact scores.
        include_sec: Whether to include latest SEC filing(s).
        include_sentiment: Whether to include sentiment metrics.
        include_fundamentals: Whether to include fundamental ratios.
        include_alt_data: Whether to include alternative data (if available).

    Returns:
        Dictionary with per-component intelligence, plus module availability.

    Example:
        >>> intel = get_comprehensive_intel("AAPL")
        >>> len(intel["news"])
        >>> intel["sentiment"].get("aggregate_score")
    """
    logger.info("Gathering comprehensive intelligence for %s.", ticker)

    data: Dict[str, Any] = {
        "ticker": ticker,
        "news": [],
        "sec_filings": {},
        "sentiment": {},
        "fundamentals": {},
        "alternative_data": {},
        "modules_available": {
            "news": _NEWS_AVAILABLE,
            "sec": _SEC_AVAILABLE,
            "sentiment": _SENTIMENT_AVAILABLE,
            "fundamentals": _FUNDAMENTALS_AVAILABLE,
            "alt_data": _ALT_DATA_AVAILABLE,
        },
    }

    if include_news and _NEWS_AVAILABLE:
        try:
            # default source logic is handled inside news_aggregator
            articles = fetch_latest_news(ticker, sources=None)  # type: ignore[arg-type]
            data["news"] = articles
            logger.info("Fetched %d news articles for %s.", len(articles), ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error fetching news for %s: %s", ticker, exc)

    if include_sec and _SEC_AVAILABLE:
        try:
            filing = fetch_latest_filing(ticker, filing_type="10-Q")  # type: ignore[arg-type]
            data["sec_filings"] = filing
            logger.info("Fetched latest SEC filing for %s.", ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error fetching SEC filings for %s: %s", ticker, exc)

    if include_sentiment and _SENTIMENT_AVAILABLE:
        try:
            sent = analyze_sentiment(ticker)  # type: ignore[call-arg]
            data["sentiment"] = sent
            agg_score = getattr(sent, "aggregate_score", None) if not isinstance(
                sent, dict
            ) else sent.get("aggregate_score")
            logger.info("Sentiment for %s: %s.", ticker, agg_score)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error analyzing sentiment for %s: %s", ticker, exc)

    if include_fundamentals and _FUNDAMENTALS_AVAILABLE:
        try:
            fundamentals = get_fundamental_metrics(ticker)  # type: ignore[call-arg]
            data["fundamentals"] = fundamentals
            logger.info("Fetched fundamental metrics for %s.", ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error fetching fundamentals for %s: %s", ticker, exc)

    if include_alt_data and _ALT_DATA_AVAILABLE:
        try:
            alt = get_alternative_data(ticker)  # type: ignore[call-arg]
            data["alternative_data"] = alt
            logger.info("Fetched alternative data for %s.", ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error fetching alternative data for %s: %s", ticker, exc)

    logger.info("Intelligence gathering complete for %s.", ticker)
    return data


# ──────────────────────────────────────────────────────────
# MODULE STATUS & DIAGNOSTICS
# ──────────────────────────────────────────────────────────


def get_module_status() -> Dict[str, bool]:
    """
    Return availability status of all intelligence submodules.

    Returns:
        Mapping of module name → availability flag.
    """
    return {
        "news_aggregator": _NEWS_AVAILABLE,
        "sec_parser": _SEC_AVAILABLE,
        "sentiment_analyzer": _SENTIMENT_AVAILABLE,
        "fundamental_analyzer": _FUNDAMENTALS_AVAILABLE,
        "alternative_data": _ALT_DATA_AVAILABLE,
    }


def list_available_modules() -> List[str]:
    """
    List the names of all available intelligence modules.

    Returns:
        List of module names with True availability.
    """
    status = get_module_status()
    return [name for name, available in status.items() if available]


logger.info("Intelligence module v%s initialized.", __version__)
_available = list_available_modules()
logger.info(
    "Available intelligence submodules: %s",
    ", ".join(_available) if _available else "None",
)

# ──────────────────────────────────────────────────────────
# PACKAGE INITIALIZATION SUMMARY
# ──────────────────────────────────────────────────────────

_init_summary_shown = False

if not _init_summary_shown:
    total_modules = 5
    available_count = sum(
        [
            _NEWS_AVAILABLE,
            _SEC_AVAILABLE,
            _SENTIMENT_AVAILABLE,
            _FUNDAMENTALS_AVAILABLE,
            _ALT_DATA_AVAILABLE,
        ]
    )
    logger.info(
        "Intelligence package summary: %d/%d modules available.",
        available_count,
        total_modules,
    )
    _init_summary_shown = True
