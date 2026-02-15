"""
============================================================
ALPHA-PRIME v2.0 - Sentiment Aggregator
============================================================

Aggregates sentiment from multiple sources into unified score:

Why Sentiment Analysis?
- Market behaviour is heavily influenced by investor psychology (fear and greed). [web:300][web:306]
- Sentiment can lead price action, especially around news and inflection points.
- Extremes in sentiment often precede reversals as positioning becomes crowded.
- Widely shared consensus can become a contrarian indicator.
- Social platforms amplify and accelerate retail sentiment transmission.

Sentiment Sources & Weights (default):
1. NEWS SENTIMENT (25%):
   - Financial news headlines and summaries.
   - Simple NLP / keyword-based polarity.
   - Major outlets (indirectly via Yahoo Finance news). [web:88][web:324][web:329]

2. SOCIAL MEDIA (20%):
   - Twitter/X, Reddit, StockTwits (placeholder hooks).
   - Volume-weighted sentiment (more mentions = more influence).
   - Trend detection via changes in mention volume.

3. ANALYST RATINGS (25%):
   - Buy/Hold/Sell distributions and recent changes.
   - Consensus rating and price target vs current price. [web:230][web:330][web:332]
   - Institutional research signal.

4. INSIDER ACTIVITY (15%):
   - Insider buying vs selling patterns.
   - Uses internal insider_tracker module when available.

5. OPTIONS FLOW (15%):
   - Options sentiment, including put/call skew and flow.
   - Uses internal options_flow module when available.

Sentiment Scoring:
- Each component is scored on a -100 to +100 scale.
- Weighted average yields aggregate sentiment.
- Confidence depends on coverage and data quality.
- Recent information implicitly weighted via source recency.

Sentiment Bands:
- +75 to +100: VERY_BULLISH.
- +50 to +75: BULLISH.
- +25 to +50: MODERATELY_BULLISH.
- -25 to +25: NEUTRAL.
- -50 to -25: MODERATELY_BEARISH.
- -75 to -50: BEARISH.
- -100 to -75: VERY_BEARISH.

Contrarian Indicators:
- Extreme optimism can indicate overcrowded longs and exhaustion risk.
- Extreme pessimism can accompany forced selling and capitulation.
- “Be fearful when others are greedy, and greedy when others are fearful.”

Usage:
    from intelligence.sentiment_aggregator import get_aggregate_sentiment

    s = get_aggregate_sentiment("AAPL")
    print(s.aggregate_score, s.sentiment_label.value)

Integration:
- research_engine.py consumes aggregate and component scores.
- brain.py uses sentiment as a feature in ranking and risk overlays.
- Dashboard displays sentiment gauge, breakdown, and shift diagnostics.
============================================================
"""

from __future__ import annotations

import logging
import re
from collections import Counter
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
cache = Cache(f"{settings.cache_dir}/sentiment")

# Optional intelligence modules
try:
    from intelligence.insider_tracker import calculate_insider_score  # type: ignore[import]

    _INSIDER_AVAILABLE = True
except ImportError:
    _INSIDER_AVAILABLE = False
    logger.debug("Insider tracker not available; insider sentiment disabled.")

try:
    from intelligence.options_flow import analyze_options_sentiment  # type: ignore[import]

    _OPTIONS_AVAILABLE = True
except ImportError:
    _OPTIONS_AVAILABLE = False
    logger.debug("Options flow module not available; options sentiment disabled.")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS & SENTIMENT DICTIONARIES
# ──────────────────────────────────────────────────────────


class SentimentLabel(str, Enum):
    """Sentiment scale classification."""

    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    MODERATELY_BULLISH = "MODERATELY_BULLISH"
    NEUTRAL = "NEUTRAL"
    MODERATELY_BEARISH = "MODERATELY_BEARISH"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


POSITIVE_WORDS: set[str] = {
    "bullish",
    "buy",
    "strong",
    "growth",
    "beat",
    "surge",
    "rally",
    "gain",
    "positive",
    "upgrade",
    "outperform",
    "excellent",
    "profit",
    "win",
    "breakout",
    "momentum",
    "boom",
    "soar",
    "jump",
    "rocket",
    "moon",
    "optimistic",
    "confident",
    "success",
    "opportunity",
    "upside",
}

NEGATIVE_WORDS: set[str] = {
    "bearish",
    "sell",
    "weak",
    "decline",
    "miss",
    "plunge",
    "crash",
    "loss",
    "negative",
    "downgrade",
    "underperform",
    "poor",
    "risk",
    "fail",
    "breakdown",
    "drop",
    "dump",
    "tank",
    "fall",
    "collapse",
    "concern",
    "pessimistic",
    "worried",
    "trouble",
    "threat",
    "downside",
}


@dataclass
class NewsSentiment:
    """News-based sentiment metrics for a ticker."""

    ticker: str
    sentiment_score: float
    positive_count: int
    negative_count: int
    neutral_count: int
    total_articles: int
    recent_headlines: List[str]
    confidence: float


@dataclass
class SocialSentiment:
    """Aggregated social-media sentiment metrics."""

    ticker: str
    sentiment_score: float
    mention_count: int
    positive_mentions: int
    negative_mentions: int
    bullish_ratio: float
    trending: bool
    top_keywords: List[str]
    confidence: float


@dataclass
class AnalystSentiment:
    """Analyst recommendation and price-target sentiment."""

    ticker: str
    sentiment_score: float
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    total_analysts: int
    consensus: str
    price_target: Optional[float]
    upside_potential: Optional[float]
    confidence: float


@dataclass
class AggregateSentiment:
    """
    Unified sentiment view across all components.

    Attributes:
        ticker: Symbol.
        aggregate_score: Weighted score in [-100, 100].
        sentiment_label: Classified sentiment band.
        news_score, social_score, analyst_score, insider_score, options_score:
            Component scores normalized to [-100, 100].
        component_weights: Effective weights after availability normalization.
        confidence: Aggregate confidence (0–100).
        data_sources_available: Number of non-zero components.
        contrarian_signal: Whether sentiment is in extreme band.
        recommendation: FOLLOW / FADE / NEUTRAL.
        last_updated_utc: ISO timestamp.
    """

    ticker: str
    aggregate_score: float
    sentiment_label: SentimentLabel
    news_score: float
    social_score: float
    analyst_score: float
    insider_score: float
    options_score: float
    component_weights: Dict[str, float]
    confidence: float
    data_sources_available: int
    contrarian_signal: bool
    recommendation: str
    last_updated_utc: str


@dataclass
class SentimentShift:
    """Change in aggregate sentiment over a lookback window."""

    ticker: str
    shift_detected: bool
    previous_score: float
    current_score: float
    change_magnitude: float
    shift_direction: str
    shift_speed: str
    description: str


@dataclass
class SentimentBreakdown:
    """Detailed breakdown of sentiment components for a ticker."""

    ticker: str
    aggregate: AggregateSentiment
    news: NewsSentiment
    social: SocialSentiment
    analyst: AnalystSentiment
    insider_score: Optional[float]
    options_score: Optional[float]


# ──────────────────────────────────────────────────────────
# NEWS SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────


def score_text_sentiment(text: str) -> float:
    """
    Score sentiment of a text snippet via keyword dictionaries.

    Uses simple term presence rather than full tokenization.
    Multiple matches increase magnitude but score is normalized
    to [-100, 100].

    Args:
        text: Text to analyse.

    Returns:
        Sentiment score in range [-100, 100].
    """
    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    score = (pos_count - neg_count) / total * 100.0
    return float(score)


def analyze_news_sentiment(ticker: str, days_back: int = 7) -> NewsSentiment:
    """
    Analyse news sentiment for a ticker based on recent headlines. [web:88][web:324][web:327][web:329]

    Limitations:
        - Uses Yahoo Finance Ticker.news feed; coverage varies by symbol.
        - Headlines may include broader market news that still influences sentiment.

    Args:
        ticker: Stock symbol.
        days_back: Number of days of news to consider (advisory, via headline count).

    Returns:
        NewsSentiment object with aggregate score and headline stats.
    """
    cache_key = f"news_sentiment_{ticker.upper()}_{days_back}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("News sentiment cache hit for %s.", ticker)
        return cached

    try:
        yf_ticker = yf.Ticker(ticker)
        news_items = getattr(yf_ticker, "news", [])  # may be list of dicts
    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching news for %s: %s", ticker, exc, exc_info=True)
        news_items = []

    if not news_items:
        logger.warning("No news entries for %s.", ticker)
        sentiment = NewsSentiment(
            ticker=ticker.upper(),
            sentiment_score=0.0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            total_articles=0,
            recent_headlines=[],
            confidence=0.0,
        )
        cache.set(cache_key, sentiment, expire=60 * 60)
        return sentiment

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    scores: List[float] = []
    recent_headlines: List[str] = []
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts = now_ts - days_back * 24 * 60 * 60

    for article in news_items[:50]:
        title = str(article.get("title", "") or "")
        pub_ts = article.get("providerPublishTime") or article.get("published_at")
        if isinstance(pub_ts, (int, float)):
            if pub_ts < cutoff_ts:
                continue

        if not title:
            continue

        recent_headlines.append(title)
        score = score_text_sentiment(title)
        scores.append(score)

        if score > 20.0:
            positive_count += 1
        elif score < -20.0:
            negative_count += 1
        else:
            neutral_count += 1

    if scores:
        avg_score = float(np.mean(scores))
    else:
        avg_score = 0.0

    confidence = float(min(100.0, len(scores) / 20.0 * 100.0)) if scores else 0.0

    sentiment = NewsSentiment(
        ticker=ticker.upper(),
        sentiment_score=avg_score,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        total_articles=len(news_items),
        recent_headlines=recent_headlines[:5],
        confidence=confidence,
    )

    logger.info(
        "News sentiment %s: score=%.1f, pos=%d, neg=%d, total=%d.",
        ticker,
        avg_score,
        positive_count,
        negative_count,
        len(news_items),
    )
    cache.set(cache_key, sentiment, expire=60 * 60)
    return sentiment


# ──────────────────────────────────────────────────────────
# SOCIAL MEDIA SENTIMENT (PLACEHOLDER)
# ──────────────────────────────────────────────────────────


def analyze_social_sentiment(ticker: str) -> SocialSentiment:
    """
    Placeholder social sentiment module.

    In production:
        - Integrate Twitter/X API, Reddit (e.g. r/wallstreetbets), StockTwits.
        - Use volume-weighted sentiment and trend analysis.
        - Maintain per-source confidence and combined score.

    Args:
        ticker: Stock symbol.

    Returns:
        SocialSentiment with neutral score and low confidence.
    """
    logger.debug("Social sentiment analysis for %s (placeholder).", ticker)
    return SocialSentiment(
        ticker=ticker.upper(),
        sentiment_score=0.0,
        mention_count=0,
        positive_mentions=0,
        negative_mentions=0,
        bullish_ratio=0.5,
        trending=False,
        top_keywords=[],
        confidence=0.0,
    )


# ──────────────────────────────────────────────────────────
# ANALYST SENTIMENT
# ──────────────────────────────────────────────────────────


def get_analyst_sentiment(ticker: str) -> AnalystSentiment:
    """
    Derive analyst sentiment from recommendations and price targets. [web:230][web:330][web:332]

    Uses yfinance analysis endpoints, falling back to neutral if data is absent.

    Args:
        ticker: Stock symbol.

    Returns:
        AnalystSentiment object.
    """
    cache_key = f"analyst_sentiment_{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Analyst sentiment cache hit for %s.", ticker)
        return cached

    try:
        yf_ticker = yf.Ticker(ticker)

        try:
            rec_df = yf_ticker.get_recommendations(as_dict=False)  # type: ignore[attr-defined]
        except Exception:
            rec_df = getattr(yf_ticker, "recommendations", None)

        if rec_df is None or len(rec_df) == 0:
            logger.warning("No analyst recommendations for %s.", ticker)
            sentiment = AnalystSentiment(
                ticker=ticker.upper(),
                sentiment_score=0.0,
                strong_buy=0,
                buy=0,
                hold=0,
                sell=0,
                strong_sell=0,
                total_analysts=0,
                consensus="HOLD",
                price_target=None,
                upside_potential=None,
                confidence=0.0,
            )
            cache.set(cache_key, sentiment, expire=4 * 60 * 60)
            return sentiment

        if isinstance(rec_df.index, pd.DatetimeIndex):
            cutoff = datetime.now() - timedelta(days=90)
            recent = rec_df[rec_df.index >= cutoff]
            if recent.empty:
                recent = rec_df.tail(50)
        else:
            recent = rec_df.tail(50)

        to_grade_col = None
        for col_candidate in ("To Grade", "toGrade", "to_grade"):
            if col_candidate in recent.columns:
                to_grade_col = col_candidate
                break

        if to_grade_col is None:
            sentiment = AnalystSentiment(
                ticker=ticker.upper(),
                sentiment_score=0.0,
                strong_buy=0,
                buy=0,
                hold=0,
                sell=0,
                strong_sell=0,
                total_analysts=0,
                consensus="HOLD",
                price_target=None,
                upside_potential=None,
                confidence=0.0,
            )
            cache.set(cache_key, sentiment, expire=4 * 60 * 60)
            return sentiment

        grades = recent[to_grade_col].astype(str)
        counts = grades.value_counts()

        strong_buy = float(counts.get("Strong Buy", 0))
        buy = float(counts.get("Buy", 0) + counts.get("Outperform", 0))
        hold = float(counts.get("Hold", 0) + counts.get("Neutral", 0))
        sell = float(counts.get("Sell", 0) + counts.get("Underperform", 0))
        strong_sell = float(counts.get("Strong Sell", 0))

        total = strong_buy + buy + hold + sell + strong_sell

        if total > 0:
            weighted_score = (
                strong_buy * 2.0 + buy * 1.0 + hold * 0.0 + sell * -1.0 + strong_sell * -2.0
            ) / total
            sentiment_score = float(weighted_score * 50.0)
        else:
            sentiment_score = 0.0

        if sentiment_score > 30.0:
            consensus = "BUY"
        elif sentiment_score < -30.0:
            consensus = "SELL"
        else:
            consensus = "HOLD"

        info = yf_ticker.info
        target_price = info.get("targetMeanPrice")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        if target_price and current_price:
            try:
                upside = float(target_price) / float(current_price) - 1.0
                upside_pct = upside * 100.0
            except Exception:  # noqa: BLE001
                upside_pct = None
        else:
            upside_pct = None

        confidence = float(min(100.0, total / 10.0 * 100.0))

        sentiment = AnalystSentiment(
            ticker=ticker.upper(),
            sentiment_score=sentiment_score,
            strong_buy=int(strong_buy),
            buy=int(buy),
            hold=int(hold),
            sell=int(sell),
            strong_sell=int(strong_sell),
            total_analysts=int(total),
            consensus=consensus,
            price_target=float(target_price) if target_price is not None else None,
            upside_potential=upside_pct,
            confidence=confidence,
        )

        logger.info(
            "Analyst sentiment %s: score=%.1f, consensus=%s, analysts=%d.",
            ticker,
            sentiment_score,
            consensus,
            int(total),
        )
        cache.set(cache_key, sentiment, expire=4 * 60 * 60)
        return sentiment
    except Exception as exc:  # noqa: BLE001
        logger.error("Error deriving analyst sentiment for %s: %s", ticker, exc, exc_info=True)
        sentiment = AnalystSentiment(
            ticker=ticker.upper(),
            sentiment_score=0.0,
            strong_buy=0,
            buy=0,
            hold=0,
            sell=0,
            strong_sell=0,
            total_analysts=0,
            consensus="HOLD",
            price_target=None,
            upside_potential=None,
            confidence=0.0,
        )
        cache.set(cache_key, sentiment, expire=4 * 60 * 60)
        return sentiment


# ──────────────────────────────────────────────────────────
# AGGREGATE SENTIMENT CALCULATION
# ──────────────────────────────────────────────────────────


def _normalize_weights_by_availability(
    scores: Dict[str, float], base_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Re-normalize component weights given availability of non-zero scores.

    Args:
        scores: Mapping from component name to score.
        base_weights: Default weights.

    Returns:
        Normalized weights summing to 1.0 for available components.
    """
    available = {k: base_weights[k] for k, v in scores.items() if abs(v) > 1e-6}
    if not available:
        return base_weights
    total = sum(available.values())
    if total <= 0:
        return base_weights
    return {k: v / total for k, v in available.items()}


def calculate_weighted_sentiment(
    news_score: float,
    social_score: float,
    analyst_score: float,
    insider_score: float,
    options_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate weighted aggregate sentiment from component scores.

    Default weights:
        news:   0.25
        social: 0.20
        analyst:0.25
        insider:0.15
        options:0.15

    Components with effectively zero signal are down-weighted via
    re-normalisation.

    Args:
        news_score: News sentiment in [-100, 100].
        social_score: Social sentiment in [-100, 100].
        analyst_score: Analyst sentiment in [-100, 100].
        insider_score: Insider sentiment in [-100, 100].
        options_score: Options sentiment in [-100, 100].
        weights: Optional custom weights.

    Returns:
        (aggregate_score, effective_weights) tuple.
    """
    base_weights = weights or {
        "news": 0.25,
        "social": 0.20,
        "analyst": 0.25,
        "insider": 0.15,
        "options": 0.15,
    }

    scores = {
        "news": float(news_score),
        "social": float(social_score),
        "analyst": float(analyst_score),
        "insider": float(insider_score),
        "options": float(options_score),
    }

    eff_weights = _normalize_weights_by_availability(scores, base_weights)

    aggregate = (
        eff_weights.get("news", 0.0) * scores["news"]
        + eff_weights.get("social", 0.0) * scores["social"]
        + eff_weights.get("analyst", 0.0) * scores["analyst"]
        + eff_weights.get("insider", 0.0) * scores["insider"]
        + eff_weights.get("options", 0.0) * scores["options"]
    )

    return float(aggregate), eff_weights


def _label_from_score(score: float) -> SentimentLabel:
    """Map numeric aggregate score into SentimentLabel band."""
    if score >= 75.0:
        return SentimentLabel.VERY_BULLISH
    if score >= 50.0:
        return SentimentLabel.BULLISH
    if score >= 25.0:
        return SentimentLabel.MODERATELY_BULLISH
    if score >= -25.0:
        return SentimentLabel.NEUTRAL
    if score >= -50.0:
        return SentimentLabel.MODERATELY_BEARISH
    if score >= -75.0:
        return SentimentLabel.BEARISH
    return SentimentLabel.VERY_BEARISH


def get_aggregate_sentiment(ticker: str) -> AggregateSentiment:
    """
    Compute aggregate sentiment for a ticker across all available sources.

    Sources:
        - News (yfinance Ticker.news).
        - Social (placeholder).
        - Analyst recommendations and price targets.
        - Insider_tracker (if available).
        - Options_flow (if available).

    Args:
        ticker: Stock symbol.

    Returns:
        AggregateSentiment instance.
    """
    ticker_u = ticker.upper()
    logger.info("Aggregating sentiment for %s.", ticker_u)

    news = analyze_news_sentiment(ticker_u)
    social = analyze_social_sentiment(ticker_u)
    analyst = get_analyst_sentiment(ticker_u)

    if _INSIDER_AVAILABLE:
        try:
            insider_score = float(calculate_insider_score(ticker_u))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error computing insider sentiment for %s: %s", ticker_u, exc, exc_info=True)
            insider_score = 0.0
    else:
        insider_score = 0.0

    if _OPTIONS_AVAILABLE:
        try:
            opt_sent = analyze_options_sentiment(ticker_u)
            options_score = float(opt_sent.sentiment_score)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error computing options sentiment for %s: %s", ticker_u, exc, exc_info=True)
            options_score = 0.0
    else:
        options_score = 0.0

    agg_score, eff_weights = calculate_weighted_sentiment(
        news_score=news.sentiment_score,
        social_score=social.sentiment_score,
        analyst_score=analyst.sentiment_score,
        insider_score=insider_score,
        options_score=options_score,
    )

    label = _label_from_score(agg_score)

    confidences = [news.confidence, social.confidence, analyst.confidence]
    non_zero_conf = [c for c in confidences if c > 0]
    avg_conf = float(np.mean(non_zero_conf)) if non_zero_conf else 0.0

    sources_available = sum(
        1
        for v in (
            news.sentiment_score,
            social.sentiment_score,
            analyst.sentiment_score,
            insider_score,
            options_score,
        )
        if abs(v) > 1e-6
    )

    contrarian_signal = abs(agg_score) > 75.0

    if contrarian_signal:
        recommendation = "FADE"
    elif 40.0 < agg_score < 75.0 or -75.0 < agg_score < -40.0:
        recommendation = "FOLLOW"
    else:
        recommendation = "NEUTRAL"

    agg = AggregateSentiment(
        ticker=ticker_u,
        aggregate_score=float(agg_score),
        sentiment_label=label,
        news_score=float(news.sentiment_score),
        social_score=float(social.sentiment_score),
        analyst_score=float(analyst.sentiment_score),
        insider_score=float(insider_score),
        options_score=float(options_score),
        component_weights=eff_weights,
        confidence=avg_conf,
        data_sources_available=int(sources_available),
        contrarian_signal=bool(contrarian_signal),
        recommendation=recommendation,
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Aggregate sentiment %s: score=%.1f (%s), sources=%d, contrarian=%s.",
        ticker_u,
        agg_score,
        label.value,
        sources_available,
        contrarian_signal,
    )
    return agg


# ──────────────────────────────────────────────────────────
# SENTIMENT BREAKDOWN & SHIFT DETECTION
# ──────────────────────────────────────────────────────────


def get_sentiment_breakdown(ticker: str) -> SentimentBreakdown:
    """
    Retrieve a full sentiment breakdown for diagnostics and UI.

    Args:
        ticker: Stock symbol.

    Returns:
        SentimentBreakdown with aggregate and component structures.
    """
    ticker_u = ticker.upper()
    aggregate = get_aggregate_sentiment(ticker_u)
    news = analyze_news_sentiment(ticker_u)
    social = analyze_social_sentiment(ticker_u)
    analyst = get_analyst_sentiment(ticker_u)

    breakdown = SentimentBreakdown(
        ticker=ticker_u,
        aggregate=aggregate,
        news=news,
        social=social,
        analyst=analyst,
        insider_score=aggregate.insider_score,
        options_score=aggregate.options_score,
    )
    return breakdown


def detect_sentiment_shift(ticker: str, lookback_days: int = 30) -> SentimentShift:
    """
    Detect shifts in aggregate sentiment over a lookback window.

    Implementation:
        - Stores an in-cache history of aggregate scores for up to 90 days.
        - Compares current score with average of scores older than lookback_days.
        - Flags a shift if absolute change > 25 points.

    Args:
        ticker: Stock symbol.
        lookback_days: Lookback window in days.

    Returns:
        SentimentShift description.
    """
    ticker_u = ticker.upper()
    current = get_aggregate_sentiment(ticker_u)
    current_score = current.aggregate_score

    cache_key = f"sentiment_history_{ticker_u}"
    history: List[Dict[str, object]] = cache.get(cache_key) or []

    now = datetime.now(timezone.utc)
    history.append({"timestamp": now, "score": float(current_score)})

    cutoff_90 = now - timedelta(days=90)
    history = [h for h in history if isinstance(h.get("timestamp"), datetime) and h["timestamp"] >= cutoff_90]  # type: ignore[index]

    cache.set(cache_key, history, expire=90 * 24 * 60 * 60)

    cutoff_lb = now - timedelta(days=lookback_days)
    old_scores = [
        float(h["score"])
        for h in history
        if isinstance(h.get("timestamp"), datetime) and h["timestamp"] < cutoff_lb  # type: ignore[index]
    ]

    if old_scores:
        previous_score = float(np.mean(old_scores))
        change = current_score - previous_score
        shift_detected = abs(change) > 25.0

        if change > 0:
            direction = "POSITIVE"
        elif change < 0:
            direction = "NEGATIVE"
        else:
            direction = "STABLE"

        speed = "RAPID" if abs(change) > 40.0 else "GRADUAL"
        description = f"Sentiment shifted {direction.lower()} by {abs(change):.0f} points."
    else:
        previous_score = float(current_score)
        change = 0.0
        shift_detected = False
        direction = "STABLE"
        speed = "GRADUAL"
        description = "Insufficient historical sentiment data."

    shift = SentimentShift(
        ticker=ticker_u,
        shift_detected=shift_detected,
        previous_score=float(previous_score),
        current_score=float(current_score),
        change_magnitude=float(abs(change)),
        shift_direction=direction,
        shift_speed=speed,
        description=description,
    )

    logger.info(
        "Sentiment shift %s: prev=%.1f, curr=%.1f, change=%.1f, detected=%s.",
        ticker_u,
        previous_score,
        current_score,
        change,
        shift_detected,
    )
    return shift


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    import traceback

    print("\n" + "=" * 70)
    print("ALPHA-PRIME Sentiment Aggregator")
    print("=" * 70 + "\n")

    if len(sys.argv) < 2:
        print("Usage: python sentiment_aggregator.py TICKER")
        print("Example: python sentiment_aggregator.py AAPL\n")
        sys.exit(1)

    ticker_cli = sys.argv[1].upper()
    print(f"Analyzing sentiment for {ticker_cli}...\n")

    try:
        agg_cli = get_aggregate_sentiment(ticker_cli)

        print("AGGREGATE SENTIMENT")
        print("-" * 70)
        print(f"Score         : {agg_cli.aggregate_score:.0f}/100")
        print(f"Label         : {agg_cli.sentiment_label.value}")
        print(f"Confidence    : {agg_cli.confidence:.0f}%")
        print(f"Data Sources  : {agg_cli.data_sources_available}/5")
        print(f"Recommendation: {agg_cli.recommendation}")
        if agg_cli.contrarian_signal:
            print("\n⚠️  CONTRARIAN SIGNAL: Extreme sentiment band detected.")

        print("\nCOMPONENT BREAKDOWN")
        print("-" * 70)
        w = agg_cli.component_weights
        print(f"News     : {agg_cli.news_score:+6.0f} (w={w.get('news', 0)*100:.0f}%)")
        print(f"Social   : {agg_cli.social_score:+6.0f} (w={w.get('social', 0)*100:.0f}%)")
        print(f"Analyst  : {agg_cli.analyst_score:+6.0f} (w={w.get('analyst', 0)*100:.0f}%)")
        print(f"Insider  : {agg_cli.insider_score:+6.0f} (w={w.get('insider', 0)*100:.0f}%)")
        print(f"Options  : {agg_cli.options_score:+6.0f} (w={w.get('options', 0)*100:.0f}%)")

        breakdown_cli = get_sentiment_breakdown(ticker_cli)

        print("\nNEWS DETAILS")
        print("-" * 70)
        ns = breakdown_cli.news
        print(f"Articles : {ns.total_articles}")
        print(f"Positive : {ns.positive_count}, Negative: {ns.negative_count}, Neutral: {ns.neutral_count}")
        if ns.recent_headlines:
            print("Recent headlines:")
            for h in ns.recent_headlines[:3]:
                print(f"  - {h[:80]}")

        print("\nANALYST DETAILS")
        print("-" * 70)
        an = breakdown_cli.analyst
        print(f"Consensus      : {an.consensus}")
        print(f"Total Analysts : {an.total_analysts}")
        print(f"Strong Buy/Buy : {an.strong_buy}/{an.buy}")
        print(f"Hold/Sell/StrS : {an.hold}/{an.sell}/{an.strong_sell}")
        if an.price_target is not None:
            print(f"Price Target   : ${an.price_target:.2f}")
            if an.upside_potential is not None:
                print(f"Upside Potential: {an.upside_potential:+.1f}%")

        print("\nSENTIMENT SHIFT (30-day)")
        print("-" * 70)
        shift_cli = detect_sentiment_shift(ticker_cli, lookback_days=30)
        print(f"Description: {shift_cli.description}")
        if shift_cli.shift_detected:
            print(f"Direction  : {shift_cli.shift_direction}")
            print(f"Speed      : {shift_cli.shift_speed}")
            print(f"Change     : {shift_cli.change_magnitude:.0f} points")

        print("\n" + "=" * 70 + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error analysing sentiment for {ticker_cli}: {exc}")
        traceback.print_exc()
