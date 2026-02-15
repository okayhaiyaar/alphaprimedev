"""
============================================================
ALPHA-PRIME v2.0 - Research Engine (The Hunter)
============================================================
Module 1: Gathers comprehensive market intelligence.

Data Sources:
1. SEC EDGAR: 10-K filings, Form 4 (insider transactions)
2. News: RSS feeds + optional paid APIs
3. Social: Reddit (PRAW), X/Twitter (official API)

Usage:
    from research_engine import get_god_tier_intel

    intel = get_god_tier_intel("AAPL")
    print(intel["fundamentals"]["going_concern_flags"])
    print(intel["sentiment_score"]["hype_score"])

Output Schema:
    GodTierIntel {
        ticker: str
        fundamentals: FundamentalsIntel
        news_catalysts: NewsIntel
        sentiment_score: SentimentIntel
        generated_at_utc: str
    }

Notes:
- All timestamps are UTC ISO 8601 strings.
- External APIs are called with basic retry logic and caching.
- If optional sources (Reddit, etc.) are unavailable, module degrades gracefully.
============================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_logger, get_settings

# Initialize settings, logger, and cache
settings = get_settings()
logger = get_logger(__name__)
cache = Cache(str(Path(settings.cache_dir) / "research"))

# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class QuoteEvidence:
    """
    Evidence snippet from a filing or document.

    Attributes:
        text: Short snippet of the relevant text.
        url: URL to the full filing or document.
        filing_type: Type of filing (e.g., "10-K", "10-Q", "8-K").
        accession: SEC accession number, if applicable.
        date: Filing date in ISO 8601 or YYYY-MM-DD format.
    """

    text: str
    url: str
    filing_type: str
    accession: Optional[str] = None
    date: Optional[str] = None


@dataclass
class SourceRef:
    """
    Reference to a data source used in intel.

    Attributes:
        name: Human-readable source name (e.g., "SEC EDGAR").
        url: URL of the API or page used.
        retrieved_at_utc: Time the data was fetched in UTC ISO format.
    """

    name: str
    url: str
    retrieved_at_utc: str


@dataclass
class FundamentalsIntel:
    """
    Fundamental analysis summary from SEC filings.

    Attributes:
        going_concern_flags: Snippets related to going concern risks.
        debt_maturity_flags: Snippets related to near-term debt or refinance risk.
        insider_selling_summary: High-level summary of insider activity.
        key_quotes: List of evidence snippets from filings.
        sources: List of source references used.
    """

    going_concern_flags: List[str]
    debt_maturity_flags: List[str]
    insider_selling_summary: str
    key_quotes: List[QuoteEvidence]
    sources: List[SourceRef]


@dataclass
class NewsItem:
    """
    Single news article.

    Attributes:
        title: Headline text.
        url: Canonical article URL.
        published_at_utc: Publication time as UTC ISO string if known.
        source: Publisher/source label (e.g., "Yahoo Finance").
    """

    title: str
    url: str
    published_at_utc: str
    source: str


@dataclass
class NewsIntel:
    """
    News and catalyst analysis.

    Attributes:
        headlines: List of recent news items.
        catalysts: Short descriptions of potential catalysts.
        sources: List of source references used.
    """

    headlines: List[NewsItem]
    catalysts: List[str]
    sources: List[SourceRef]


@dataclass
class SocialPost:
    """
    Single social media post.

    Attributes:
        platform: Platform name (e.g., "reddit").
        url: URL to the post.
        author: Username or author handle.
        created_at_utc: Creation time in UTC ISO string.
        text: Main text content (trimmed).
    """

    platform: str
    url: str
    author: Optional[str]
    created_at_utc: str
    text: str


@dataclass
class SentimentIntel:
    """
    Social sentiment analysis.

    Attributes:
        hype_score: Aggregate hype score on a 0–100 scale.
        polarity: Average sentiment polarity (-1.0 to 1.0).
        volume: Number of posts considered.
        top_posts: Representative posts.
        sources: List of source references used.
        availability: Flags for which platforms were checked.
    """

    hype_score: int
    polarity: float
    volume: int
    top_posts: List[SocialPost]
    sources: List[SourceRef]
    availability: Dict[str, bool]


@dataclass
class GodTierIntel:
    """
    Complete intelligence package for a ticker.

    Attributes:
        ticker: Uppercase stock symbol.
        fundamentals: Fundamental intel from filings.
        news_catalysts: News and catalyst intel.
        sentiment_score: Social sentiment intel.
        generated_at_utc: Time intel was generated in UTC ISO.
    """

    ticker: str
    fundamentals: FundamentalsIntel
    news_catalysts: NewsIntel
    sentiment_score: SentimentIntel
    generated_at_utc: str


# ──────────────────────────────────────────────────────────
# SEC EDGAR (10-K Filings + Form 4 Insider Transactions)
# ──────────────────────────────────────────────────────────

SEC_BASE_URL = "https://data.sec.gov"
SEC_HEADERS = {
    "User-Agent": "ALPHA-PRIME/2.0 (contact@alpha-prime.dev)",
    "Accept-Encoding": "gzip, deflate",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """
    Resolve ticker to CIK using SEC's company_tickers.json.

    Args:
        ticker: Stock symbol (e.g., "AAPL").

    Returns:
        Optional[str]: CIK string with leading zeros (10 digits), or None if not found.
    """
    cache_key = f"cik_{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        url = f"{SEC_BASE_URL}/files/company_tickers.json"
        response = requests.get(url, headers=SEC_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        resolved_cik: Optional[str] = None
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                resolved_cik = str(entry["cik_str"]).zfill(10)
                break

        if not resolved_cik:
            logger.warning("CIK not found for ticker %s", ticker)
            return None

        cache.set(cache_key, resolved_cik, expire=86400 * 30)
        return resolved_cik
    except Exception as exc:  # noqa: BLE001
        logger.error("Error resolving CIK for %s: %s", ticker, exc)
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_recent_10k_filings(cik: str, years: int = 10) -> List[Dict[str, str]]:
    """
    Get recent 10-K filings for a CIK.

    Args:
        cik: Company CIK (10 digits).
        years: How many years back to search.

    Returns:
        List[Dict[str, str]]: Filings with accession, filing_date, primary_document.
    """
    try:
        url = f"{SEC_BASE_URL}/submissions/CIK{cik}.json"
        response = requests.get(url, headers=SEC_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_documents = filings.get("primaryDocument", [])

        results: List[Dict[str, str]] = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=years * 365)

        for index, form in enumerate(forms):
            if form != "10-K":
                continue

            filing_date_str = filing_dates[index]
            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )

            if filing_date < cutoff_date:
                continue

            accession = accession_numbers[index].replace("-", "")
            primary_document = primary_documents[index]

            results.append(
                {
                    "accession": accession,
                    "filing_date": filing_date_str,
                    "primary_document": primary_document,
                }
            )

        logger.info("Found %d 10-K filings for CIK %s", len(results), cik)
        return results
    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching 10-K filings for CIK %s: %s", cik, exc)
        return []


def download_and_parse_10k(cik: str, accession: str, primary_doc: str) -> str:
    """
    Download and extract text from a 10-K filing, with disk cache.

    Args:
        cik: Company CIK.
        accession: Filing accession number (no hyphens).
        primary_doc: Primary document filename.

    Returns:
        str: Extracted plain text content, or empty string on failure.
    """
    cache_dir = Path(settings.cache_dir) / "sec" / cik
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{accession}.txt"

    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read cached 10-K %s: %s", cache_file, exc)

    try:
        url = (
            f"{SEC_BASE_URL}/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
        )
        response = requests.get(url, headers=SEC_HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        try:
            cache_file.write_text(text, encoding="utf-8")
            logger.debug("Cached 10-K to %s", cache_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cache 10-K %s: %s", cache_file, exc)

        return text
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error downloading 10-K for CIK %s accession %s: %s", cik, accession, exc
        )
        return ""


def analyze_fundamentals(ticker: str) -> FundamentalsIntel:
    """
    Analyze SEC filings for fundamental red flags.

    Args:
        ticker: Stock symbol.

    Returns:
        FundamentalsIntel: Fundamental intel including going concern and debt flags.
    """
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return FundamentalsIntel(
            going_concern_flags=[],
            debt_maturity_flags=[],
            insider_selling_summary="CIK not found for ticker.",
            key_quotes=[],
            sources=[],
        )

    filings = get_recent_10k_filings(cik, years=10)

    going_concern_flags: List[str] = []
    debt_maturity_flags: List[str] = []
    key_quotes: List[QuoteEvidence] = []

    going_concern_keywords = [
        "going concern",
        "substantial doubt",
        "ability to continue as a going concern",
    ]
    debt_keywords = [
        "maturity",
        "debt due",
        "refinancing",
        "credit facility maturity",
    ]

    for filing in filings[:5]:
        text = download_and_parse_10k(
            cik, filing["accession"], filing["primary_document"]
        )
        if not text:
            continue

        text_lower = text.lower()

        for keyword in going_concern_keywords:
            if keyword in text_lower:
                index = text_lower.index(keyword)
                snippet = text[max(0, index - 100) : index + 200]
                going_concern_flags.append(
                    f"{filing['filing_date']}: {snippet[:150]}..."
                )
                key_quotes.append(
                    QuoteEvidence(
                        text=snippet[:200],
                        url=(
                            "https://www.sec.gov/cgi-bin/viewer"
                            f"?action=view&cik={cik}&accession_number={filing['accession']}"
                        ),
                        filing_type="10-K",
                        accession=filing["accession"],
                        date=filing["filing_date"],
                    )
                )
                break

        for keyword in debt_keywords:
            if keyword in text_lower:
                index = text_lower.index(keyword)
                snippet = text[max(0, index - 100) : index + 200]
                debt_maturity_flags.append(
                    f"{filing['filing_date']}: {snippet[:150]}..."
                )
                break

    insider_selling_summary = "Form 4 parsing not yet implemented (placeholder)."

    sources: List[SourceRef] = [
        SourceRef(
            name="SEC EDGAR",
            url=(
                "https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik}"
            ),
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    ]

    return FundamentalsIntel(
        going_concern_flags=going_concern_flags[:3],
        debt_maturity_flags=debt_maturity_flags[:3],
        insider_selling_summary=insider_selling_summary,
        key_quotes=key_quotes[:3],
        sources=sources,
    )


# ──────────────────────────────────────────────────────────
# NEWS & CATALYSTS
# ──────────────────────────────────────────────────────────


def _parse_rss_pubdate(raw: Optional[str]) -> str:
    """
    Parse RSS pubDate into UTC ISO string.

    Args:
        raw: Raw date string from RSS.

    Returns:
        str: UTC ISO formatted string, or empty string if parsing fails.
    """
    if not raw:
        return ""
    try:
        # Many RSS feeds are RFC 2822; fall back to naive parse if required
        parsed = datetime.strptime(raw[:25], "%a, %d %b %Y %H:%M:%S")
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        return ""


def fetch_news(ticker: str) -> NewsIntel:
    """
    Fetch recent news for a ticker.

    Uses Yahoo Finance RSS (public) as a baseline.
    Caches results for a short period to limit network calls.

    Args:
        ticker: Stock symbol.

    Returns:
        NewsIntel: News headlines and heuristic catalysts.
    """
    cache_key = f"news_{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached:
        try:
            return NewsIntel(
                headlines=[
                    NewsItem(**item) for item in cached.get("headlines", [])
                ],
                catalysts=cached.get("catalysts", []),
                sources=[SourceRef(**src) for src in cached.get("sources", [])],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to deserialize cached news for %s: %s", ticker, exc)

    headlines: List[NewsItem] = []
    sources: List[SourceRef] = []

    # Yahoo Finance RSS
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")

        for item in items[:10]:
            title_tag = item.find("title")
            link_tag = item.find("link")
            date_tag = item.find("pubDate")

            title = title_tag.text.strip() if title_tag else ""
            link = link_tag.text.strip() if link_tag else ""
            published_at_utc = _parse_rss_pubdate(
                date_tag.text.strip() if date_tag else None
            )

            headlines.append(
                NewsItem(
                    title=title,
                    url=link,
                    published_at_utc=published_at_utc,
                    source="Yahoo Finance RSS",
                )
            )

        sources.append(
            SourceRef(
                name="Yahoo Finance RSS",
                url=url,
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error fetching Yahoo Finance RSS for %s: %s", ticker, exc)

    catalysts: List[str] = []
    catalyst_keywords = [
        "earnings",
        "guidance",
        "fda",
        "approval",
        "merger",
        "acquisition",
        "lawsuit",
        "downgrade",
        "upgrade",
    ]
    for item in headlines:
        lowered = item.title.lower()
        for keyword in catalyst_keywords:
            if keyword in lowered:
                catalysts.append(f"{keyword.title()}: {item.title}")
                break

    result = NewsIntel(
        headlines=headlines,
        catalysts=catalysts[:5],
        sources=sources,
    )

    try:
        cache.set(
            cache_key,
            {
                "headlines": [asdict(h) for h in result.headlines],
                "catalysts": result.catalysts,
                "sources": [asdict(s) for s in result.sources],
            },
            expire=settings.cache_ttl_minutes * 60,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cache news for %s: %s", ticker, exc)

    return result


# ──────────────────────────────────────────────────────────
# SOCIAL SENTIMENT (Reddit, X/Twitter)
# ──────────────────────────────────────────────────────────


def fetch_social_sentiment(ticker: str) -> SentimentIntel:
    """
    Fetch social sentiment from Reddit and X/Twitter.

    Currently:
    - Reddit via PRAW if credentials are configured.
    - Twitter/X integration is left as a placeholder.

    If APIs are not configured or calls fail, returns neutral sentiment.

    Args:
        ticker: Stock symbol.

    Returns:
        SentimentIntel: Social sentiment intel.
    """
    cache_key = f"sentiment_{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached:
        try:
            return SentimentIntel(
                hype_score=cached["hype_score"],
                polarity=cached["polarity"],
                volume=cached["volume"],
                top_posts=[
                    SocialPost(**post) for post in cached.get("top_posts", [])
                ],
                sources=[SourceRef(**src) for src in cached.get("sources", [])],
                availability=cached.get("availability", {}),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to deserialize cached sentiment for %s: %s", ticker, exc
            )

    top_posts: List[SocialPost] = []
    sources: List[SourceRef] = []
    availability: Dict[str, bool] = {"reddit": False, "twitter": False}

    # Reddit via PRAW
    if settings.reddit_client_id and settings.reddit_client_secret:
        try:
            import praw

            reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )

            subreddits = reddit.subreddit("wallstreetbets+stocks+investing")
            for submission in subreddits.search(
                ticker, time_filter="day", limit=10
            ):
                created_at = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ).isoformat()
                top_posts.append(
                    SocialPost(
                        platform="reddit",
                        url=f"https://reddit.com{submission.permalink}",
                        author=str(submission.author) if submission.author else None,
                        created_at_utc=created_at,
                        text=submission.title[:280],
                    )
                )

            availability["reddit"] = True
            sources.append(
                SourceRef(
                    name="Reddit",
                    url="https://reddit.com/r/wallstreetbets",
                    retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reddit fetch failed for %s: %s", ticker, exc)

    # Twitter/X placeholder (requires official API integration)
    # availability["twitter"] remains False unless implemented.

    volume = len(top_posts)
    polarity = 0.0

    if volume > 0:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores: List[float] = []
            for post in top_posts:
                score = analyzer.polarity_scores(post.text)["compound"]
                scores.append(score)
            if scores:
                polarity = sum(scores) / len(scores)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sentiment analysis failed for %s, defaulting polarity to 0.0: %s",
                ticker,
                exc,
            )
            polarity = 0.0

    hype_score = 0
    if volume > 0:
        hype_score = min(100, int((volume / 2) * 10 + (polarity + 1.0) * 25))

    result = SentimentIntel(
        hype_score=hype_score,
        polarity=polarity,
        volume=volume,
        top_posts=top_posts[:20],
        sources=sources,
        availability=availability,
    )

    try:
        cache.set(
            cache_key,
            {
                "hype_score": result.hype_score,
                "polarity": result.polarity,
                "volume": result.volume,
                "top_posts": [asdict(p) for p in result.top_posts],
                "sources": [asdict(s) for s in result.sources],
                "availability": result.availability,
            },
            expire=settings.cache_ttl_minutes * 60,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cache sentiment for %s: %s", ticker, exc)

    return result


# ──────────────────────────────────────────────────────────
# MAIN INTELLIGENCE GATHERING FUNCTION
# ──────────────────────────────────────────────────────────


def get_god_tier_intel(ticker: str) -> Dict[str, object]:
    """
    Gather comprehensive intelligence for a ticker.

    This is the main entry point for Module 1 (The Hunter).
    It orchestrates calls to fundamentals, news, and sentiment engines.

    Args:
        ticker: Stock symbol (e.g., "AAPL").

    Returns:
        Dict[str, object]: GodTierIntel as a JSON-serializable dictionary.

    Raises:
        RuntimeError: If a critical error prevents intel from being generated.
    """
    symbol = ticker.upper().strip()
    logger.info("Gathering intelligence for %s", symbol)

    try:
        fundamentals = analyze_fundamentals(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Fundamentals analysis failed for %s: %s", symbol, exc, exc_info=True
        )
        fundamentals = FundamentalsIntel(
            going_concern_flags=[],
            debt_maturity_flags=[],
            insider_selling_summary="Fundamentals analysis failed.",
            key_quotes=[],
            sources=[],
        )

    try:
        news = fetch_news(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.error("News fetch failed for %s: %s", symbol, exc, exc_info=True)
        news = NewsIntel(headlines=[], catalysts=[], sources=[])

    try:
        sentiment = fetch_social_sentiment(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Sentiment fetch failed for %s: %s", symbol, exc, exc_info=True
        )
        sentiment = SentimentIntel(
            hype_score=0,
            polarity=0.0,
            volume=0,
            top_posts=[],
            sources=[],
            availability={"reddit": False, "twitter": False},
        )

    intel = GodTierIntel(
        ticker=symbol,
        fundamentals=fundamentals,
        news_catalysts=news,
        sentiment_score=sentiment,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Intel for %s: %d key quotes, %d headlines, %d social posts (hype_score=%d)",
        symbol,
        len(fundamentals.key_quotes),
        len(news.headlines),
        sentiment.volume,
        sentiment.hype_score,
    )

    return asdict(intel)


# ──────────────────────────────────────────────────────────
# CLI TOOL
# ──────────────────────────────────────────────────────────


def _cli() -> None:
    """
    Simple CLI entry point for manual testing.

    Usage:
        python research_engine.py TICKER
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python research_engine.py TICKER")
        print("Example: python research_engine.py AAPL")
        raise SystemExit(1)

    symbol = sys.argv[1]
    print(f"\nGathering intelligence for {symbol}...\n")
    intel = get_god_tier_intel(symbol)
    print(json.dumps(intel, indent=2, default=str))


if __name__ == "__main__":
    _cli()
