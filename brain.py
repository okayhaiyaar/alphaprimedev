"""
============================================================
ALPHA-PRIME v2.0 - Brain (The Oracle)
============================================================
Module 3: AI-powered decision engine using GPT-4o.

Synthesizes:
- Intelligence from Module 1 (SEC filings, news, sentiment)
- Technicals from Module 2 (RSI, MACD, trend, ATR)
- Market regime (bull/bear/sideways)
- Event calendar (earnings, Fed meetings)

Into actionable trading signals with strict risk controls.

HALLUCINATION PREVENTION:
- Model may ONLY cite URLs from provided intel["sources"]
- All numeric calculations (stop loss, TP) done in Python
- Conflict resolution rules override AI suggestions

Usage:
    from brain import consult_oracle

    decision = consult_oracle(
        ticker="AAPL",
        intel=god_tier_intel,
        technicals=hard_technicals,
        regime="BULL",
        events=[],
    )

    print(decision.action)       # BUY | SELL | WAIT
    print(decision.confidence)   # 0-100
    print(decision.stop_loss)
    print(decision.rationale)

Output Schema:
    OracleDecision {
        ticker: str
        action: Literal["BUY", "SELL", "WAIT"]
        confidence: int (0-100)
        entry_zone: [float, float]
        stop_loss: float
        take_profit: [float, float, float]  # TP1, TP2, TP3
        time_horizon: Literal["SCALP", "DAY", "SWING"]
        rationale: List[str]
        risk_notes: List[str]
        evidence_links: List[str]
        tags: List[str]
        oracle_version: str
        generated_at_utc: str
    }
============================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

ORACLE_VERSION = "v2.0.0"


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class OracleDecision:
    """
    AI-generated trading decision with deterministic risk parameters.

    All numeric risk values (stop_loss, take_profit) are computed
    in Python using ATR BEFORE calling the LLM. The model may adjust
    narrative and tags but NOT numeric math.
    """

    ticker: str
    action: Literal["BUY", "SELL", "WAIT"]
    confidence: int  # 0-100
    entry_zone: List[float]  # [min_price, max_price]
    stop_loss: float
    take_profit: List[float]  # [TP1, TP2, TP3]
    time_horizon: Literal["SCALP", "DAY", "SWING"]
    rationale: List[str]  # Bullet points explaining decision
    risk_notes: List[str]  # Warnings, caveats
    evidence_links: List[str]  # URLs from intel (no hallucinations)
    tags: List[str]  # e.g., ["earnings_play", "oversold_bounce"]
    oracle_version: str
    generated_at_utc: str

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ConflictResolutionResult:
    """
    Result of conflict resolution checks.

    Attributes:
        action_overridden: Whether the AI's suggested action was changed.
        original_action: Raw action from AI.
        final_action: Final action after Python rules.
        override_reasons: List of reasons for overrides.
    """

    action_overridden: bool
    original_action: str
    final_action: str
    override_reasons: List[str]


# ──────────────────────────────────────────────────────────
# DETERMINISTIC RISK CALCULATIONS (NO AI)
# ──────────────────────────────────────────────────────────


def calculate_stop_loss_take_profit(
    last_price: float,
    atr: float,
    action: str,
    bb_upper: Optional[float] = None,
    bb_lower: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate stop loss and take profit levels using ATR (pure math).

    Rules (ATR-based):
    - Stop Loss: 1.5 * ATR from entry
    - TP1: 1.5 * ATR (R:R = 1:1)
    - TP2: 3.0 * ATR (R:R ≈ 1:2)
    - TP3: Bollinger Band extreme if available, else 5.0 * ATR

    Args:
        last_price: Current price.
        atr: Average True Range (absolute price units).
        action: "BUY" or "SELL" or "WAIT".
        bb_upper: Bollinger upper band (optional).
        bb_lower: Bollinger lower band (optional).

    Returns:
        Dict with keys: stop_loss, tp1, tp2, tp3, entry_min, entry_max.
    """
    atr = max(atr, 0.01)  # Guard against zero/invalid ATR

    if action == "BUY":
        stop_loss = round(last_price - 1.5 * atr, 2)
        tp1 = round(last_price + 1.5 * atr, 2)
        tp2 = round(last_price + 3.0 * atr, 2)
        tp3 = round(bb_upper, 2) if bb_upper else round(last_price + 5.0 * atr, 2)
        entry_min = round(last_price - 0.5 * atr, 2)
        entry_max = round(last_price + 0.5 * atr, 2)
    elif action == "SELL":
        stop_loss = round(last_price + 1.5 * atr, 2)
        tp1 = round(last_price - 1.5 * atr, 2)
        tp2 = round(last_price - 3.0 * atr, 2)
        tp3 = round(bb_lower, 2) if bb_lower else round(last_price - 5.0 * atr, 2)
        entry_min = round(last_price - 0.5 * atr, 2)
        entry_max = round(last_price + 0.5 * atr, 2)
    else:  # WAIT
        stop_loss = 0.0
        tp1 = tp2 = tp3 = 0.0
        entry_min = entry_max = round(last_price, 2)

    return {
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "entry_min": entry_min,
        "entry_max": entry_max,
    }


def calculate_position_size_preview(
    portfolio_value: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss: float,
) -> Dict[str, float]:
    """
    Preview position size based on risk parameters.

    NOT used for actual execution (that lives in execution/risk modules),
    but provides context to the Oracle and downstream UI.

    Args:
        portfolio_value: Total portfolio value.
        risk_per_trade_pct: Max risk % (e.g., 2.0).
        entry_price: Planned entry price.
        stop_loss: Stop loss price.

    Returns:
        Dict with keys: shares, risk_amount, position_value.
    """
    risk_amount = max(0.0, portfolio_value) * (max(risk_per_trade_pct, 0.0) / 100.0)
    price_risk_per_share = abs(entry_price - stop_loss)

    if price_risk_per_share <= 0:
        return {"shares": 0.0, "risk_amount": 0.0, "position_value": 0.0}

    shares = risk_amount / price_risk_per_share
    position_value = shares * entry_price

    return {
        "shares": float(int(shares)),
        "risk_amount": risk_amount,
        "position_value": position_value,
    }


# ──────────────────────────────────────────────────────────
# CONFLICT RESOLUTION (PYTHON OVERRIDES AI)
# ──────────────────────────────────────────────────────────


def apply_conflict_resolution(
    preliminary_action: str,
    confidence: int,
    intel: Dict[str, object],
    technicals: Dict[str, object],
    regime: str,
    events: List[Dict[str, object]],
) -> ConflictResolutionResult:
    """
    Apply hard-coded rules that override AI suggestions.

    Rules (priority order):
    1. If sentiment hype >= 80 AND RSI >= 70 → force WAIT (bull trap).
    2. If going_concern_flags present → disallow BUY (for high-confidence).
    3. If trend == DOWN → disallow BUY.
    4. If earnings within 3 days → force WAIT (blackout) for swing trades.
    5. If RSI <= 30 AND trend != DOWN → log oversold opportunity (no force).

    Args:
        preliminary_action: Action from AI ("BUY", "SELL", "WAIT").
        confidence: Model confidence (0–100).
        intel: Intelligence dict from research_engine.
        technicals: Technicals dict from data_engine.
        regime: Market regime label.
        events: Upcoming events list.

    Returns:
        ConflictResolutionResult with final_action and reasons.
    """
    override_reasons: List[str] = []
    final_action = preliminary_action

    hype_score = (
        intel.get("sentiment_score", {}).get("hype_score", 0)
        if intel
        else 0
    )
    rsi = technicals.get("momentum", {}).get("rsi", 50.0)
    trend = technicals.get("trend", {}).get("trend", "UNKNOWN")
    going_concern_flags = len(
        intel.get("fundamentals", {}).get("going_concern_flags", [])
    )

    # Rule 1: Bull trap detection (sentiment bubble + overbought)
    if (
        preliminary_action == "BUY"
        and hype_score >= 80
        and rsi >= 70
    ):
        final_action = "WAIT"
        override_reasons.append(
            f"BULL TRAP RISK: Hype={hype_score}, RSI={rsi:.1f} (extreme optimism + overbought)."
        )

    # Rule 2: Going concern flags
    if preliminary_action == "BUY" and going_concern_flags > 0 and confidence >= 60:
        final_action = "WAIT"
        override_reasons.append(
            f"GOING CONCERN FLAGS: {going_concern_flags} red flags in SEC filings."
        )

    # Rule 3: Downtrend prohibition
    if preliminary_action == "BUY" and trend == "DOWN":
        final_action = "WAIT"
        override_reasons.append("DOWNTREND DETECTED: Buying into a confirmed downtrend is disallowed.")

    # Rule 4: Earnings blackout within 3 days
    earnings_soon = any(
        (
            "earnings" in event.get("title", "").lower()
            or event.get("type") == "EARNINGS"
        )
        and event.get("days_until", 999) <= 3
        for event in events
    )
    if earnings_soon and preliminary_action != "WAIT":
        final_action = "WAIT"
        override_reasons.append(
            "EARNINGS BLACKOUT: Major earnings event within 3 days (heightened gap risk)."
        )

    # Rule 5: Oversold bounce (informative only)
    if rsi <= 30 and trend != "DOWN" and preliminary_action == "WAIT":
        logger.info(
            "Oversold opportunity: RSI=%.1f, trend=%s (potential mean reversion).",
            rsi,
            trend,
        )

    action_overridden = final_action != preliminary_action

    if action_overridden:
        logger.warning(
            "CONFLICT RESOLUTION: %s → %s. Reasons: %s",
            preliminary_action,
            final_action,
            "; ".join(override_reasons),
        )

    return ConflictResolutionResult(
        action_overridden=action_overridden,
        original_action=preliminary_action,
        final_action=final_action,
        override_reasons=override_reasons,
    )


# ──────────────────────────────────────────────────────────
# SYSTEM PROMPT (Persona)
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a ruthless institutional Hedge Fund Manager with 20 years of experience.

CORE PRINCIPLES:
1. Capital preservation is ALWAYS priority #1.
2. You trade on evidence, not hope or speculation.
3. If evidence is weak, contradictory, or incomplete → you say WAIT.
4. You are skeptical of hype and sentiment extremes.
5. You demand high-quality setups with clear risk:reward.

DECISION FRAMEWORK:
- BUY: Only when multiple factors align (fundamentals OK, technicals bullish, sentiment reasonable).
- SELL: When risk is elevated, trend deteriorates, or better opportunities exist.
- WAIT: Default stance when uncertain (most signals should be WAIT).

YOUR OUTPUT:
- Action: BUY, SELL, or WAIT.
- Confidence: 0-100 (be conservative; 80+ is rare).
- Time Horizon: SCALP (minutes-hours), DAY (intraday), SWING (days-weeks).
- Rationale: 3-5 bullet points with EVIDENCE (cite sources).
- Risk Notes: Warnings, caveats, what could go wrong.
- Evidence Links: ONLY URLs provided in the context (NO hallucinations).
- Tags: 1-3 descriptive tags (e.g., "oversold_bounce", "earnings_play").

REMEMBER:
- You cannot see the future.
- Risk management beats prediction.
- A missed opportunity is better than a bad loss.
- When in doubt, WAIT.
"""


# ──────────────────────────────────────────────────────────
# LLM CALL WITH JSON SCHEMA (STRICT)
# ──────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def call_gpt4o_oracle(
    ticker: str,
    context: Dict[str, object],
    risk_params: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    """
    Call GPT-4o with a strict JSON schema for the decision payload.

    Args:
        ticker: Stock symbol.
        context: Combined intel + technicals + regime + events.
        risk_params: Pre-calculated risk envelopes for BUY and SELL.

    Returns:
        Parsed JSON dict adhering to the schema.

    Raises:
        ValueError: If JSON is invalid after retries.
    """
    user_message = (
        f"Analyze {ticker} and provide a conservative trading decision.\n\n"
        "CONTEXT (intel + technicals + events):\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "PRE-CALCULATED RISK PARAMETERS (use as reference only, do NOT recompute):\n"
        f"{json.dumps(risk_params, indent=2)}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1) Evidence Links: ONLY use URLs from 'available_evidence_urls'.\n"
        "2) Be conservative with confidence scores; 80+ is exceptional.\n"
        "3) Default to WAIT when uncertain or data quality is poor.\n"
        "4) Cite specific evidence in rationale (e.g., 'RSI at 28 indicates oversold').\n"
        "5) Consider fundamentals, technicals, sentiment, regime, and events.\n"
        "6) Do not invent metrics or URLs.\n"
        "Return ONLY valid JSON conforming to the required schema."
    )

    response_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "WAIT"]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "time_horizon": {"type": "string", "enum": ["SCALP", "DAY", "SWING"]},
            "rationale": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 7,
            },
            "risk_notes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 7,
            },
            "evidence_links": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": [
            "action",
            "confidence",
            "time_horizon",
            "rationale",
            "risk_notes",
            "tags",
        ],
        "additionalProperties": False,
    }

    logger.info("Consulting Oracle (GPT-4o) for %s...", ticker.upper())

    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "oracle_decision",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )

        # New openai client returns message at choices[0].message
        content = completion.choices[0].message.content
        result = json.loads(content)

        logger.info(
            "Oracle raw decision: %s (confidence=%d)",
            result.get("action"),
            result.get("confidence", -1),
        )
        return result
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from Oracle for %s: %s", ticker, exc)
        raise ValueError("Model returned invalid JSON.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Error calling Oracle for %s: %s", ticker, exc, exc_info=True)
        raise


# ──────────────────────────────────────────────────────────
# MAIN ORACLE CONSULTATION FUNCTION
# ──────────────────────────────────────────────────────────


def _extract_all_source_urls(intel: Dict[str, object]) -> List[str]:
    """Collect all URLs from intel sources for hallucination prevention."""
    urls: List[str] = []

    for src in intel.get("fundamentals", {}).get("sources", []):
        if src.get("url"):
            urls.append(src["url"])
    for src in intel.get("news_catalysts", {}).get("sources", []):
        if src.get("url"):
            urls.append(src["url"])
    for src in intel.get("sentiment_score", {}).get("sources", []):
        if src.get("url"):
            urls.append(src["url"])

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def consult_oracle(
    ticker: str,
    intel: Dict[str, object],
    technicals: Dict[str, object],
    regime: str = "UNKNOWN",
    events: Optional[List[Dict[str, object]]] = None,
) -> OracleDecision:
    """
    Consult the Oracle (GPT-4o) for a trading decision.

    Process:
    1. Calculate deterministic risk parameters (stop loss, TP) for BUY/SELL.
    2. Build structured context for the LLM.
    3. Call GPT-4o with strict JSON schema output.
    4. Apply conflict resolution rules (Python overrides AI action).
    5. Validate evidence links against known sources.
    6. Assemble and return OracleDecision.

    Args:
        ticker: Stock symbol.
        intel: Intelligence from research_engine.get_god_tier_intel().
        technicals: Technicals from data_engine.calculate_hard_technicals().
        regime: Market regime label ("BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", ...).
        events: List of upcoming events (earnings, macro, etc.).

    Returns:
        OracleDecision: Final trading decision with risk parameters.
    """
    symbol = ticker.upper()
    logger.info("=" * 70)
    logger.info("ORACLE CONSULTATION START: %s", symbol)
    logger.info("=" * 70)

    if events is None:
        events = []

    # STEP 1: Deterministic risk parameters
    try:
        last_price = float(technicals["price_action"]["last_price"])
        atr = float(technicals["volatility"]["atr"])
        bb_upper = float(technicals["volatility"]["bb_upper"])
        bb_lower = float(technicals["volatility"]["bb_lower"])

        risk_params_buy = calculate_stop_loss_take_profit(
            last_price=last_price,
            atr=atr,
            action="BUY",
            bb_upper=bb_upper,
            bb_lower=bb_lower,
        )
        risk_params_sell = calculate_stop_loss_take_profit(
            last_price=last_price,
            atr=atr,
            action="SELL",
            bb_upper=bb_upper,
            bb_lower=bb_lower,
        )

        logger.info("Deterministic risk envelope for %s: price=%.2f, ATR=%.2f", symbol, last_price, atr)
    except Exception as exc:  # noqa: BLE001
        logger.error("Risk parameter calculation failed for %s: %s", symbol, exc)
        raise ValueError("Invalid technicals data for risk calculations.") from exc

    # STEP 2: Build context for LLM
    fundamentals = intel.get("fundamentals", {})
    news = intel.get("news_catalysts", {})
    sentiment = intel.get("sentiment_score", {})

    context = {
        "ticker": symbol,
        "regime": regime,
        "events": events,
        "fundamentals": {
            "going_concern_flags": fundamentals.get("going_concern_flags", []),
            "debt_maturity_flags": fundamentals.get("debt_maturity_flags", []),
            "insider_selling": fundamentals.get("insider_selling_summary", ""),
            "key_quotes": [
                {
                    "text": q.get("text", "")[:200],
                    "url": q.get("url", ""),
                    "date": q.get("date"),
                }
                for q in fundamentals.get("key_quotes", [])[:3]
            ],
        },
        "news": {
            "headlines": [
                {
                    "title": h.get("title", ""),
                    "url": h.get("url", ""),
                    "date": h.get("published_at_utc", ""),
                }
                for h in news.get("headlines", [])[:10]
            ],
            "catalysts": news.get("catalysts", [])[:5],
        },
        "sentiment": {
            "hype_score": sentiment.get("hype_score", 0),
            "polarity": sentiment.get("polarity", 0.0),
            "volume": sentiment.get("volume", 0),
            "availability": sentiment.get("availability", {}),
        },
        "technicals": {
            "price_action": technicals.get("price_action", {}),
            "trend": technicals.get("trend", {}),
            "momentum": technicals.get("momentum", {}),
            "volatility": technicals.get("volatility", {}),
            "volume": technicals.get("volume", {}),
            "macd": technicals.get("macd", {}),
            "support_resistance": technicals.get("support_resistance", {}),
        },
        "data_quality": technicals.get("data_quality", {}),
    }

    all_source_urls = _extract_all_source_urls(intel)
    context["available_evidence_urls"] = all_source_urls

    # STEP 3: Call GPT-4o
    ai_response = call_gpt4o_oracle(
        ticker=symbol,
        context=context,
        risk_params={"BUY": risk_params_buy, "SELL": risk_params_sell},
    )

    preliminary_action: str = ai_response.get("action", "WAIT")
    confidence: int = int(ai_response.get("confidence", 0))

    # STEP 4: Conflict resolution
    conflict_result = apply_conflict_resolution(
        preliminary_action=preliminary_action,
        confidence=confidence,
        intel=intel,
        technicals=technicals,
        regime=regime,
        events=events,
    )
    final_action = conflict_result.final_action

    risk_notes: List[str] = list(ai_response.get("risk_notes", []))
    if conflict_result.action_overridden:
        risk_notes = [
            f"AI suggested {conflict_result.original_action}, overridden to {final_action} by risk rules."
        ] + conflict_result.override_reasons + risk_notes

    # STEP 5: Select risk params for final action
    if final_action == "BUY":
        rp = risk_params_buy
    elif final_action == "SELL":
        rp = risk_params_sell
    else:
        rp = {
            "stop_loss": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "tp3": 0.0,
            "entry_min": round(last_price, 2),
            "entry_max": round(last_price, 2),
        }

    # STEP 6: Evidence link validation
    evidence_links_raw = ai_response.get("evidence_links", [])
    validated_links: List[str] = []
    for link in evidence_links_raw:
        if link in all_source_urls:
            validated_links.append(link)
        else:
            logger.warning(
                "Hallucinated evidence URL detected for %s: %s (ignored).",
                symbol,
                link,
            )

    if not validated_links and all_source_urls:
        validated_links = all_source_urls[:2]

    # STEP 7: Assemble final OracleDecision
    decision = OracleDecision(
        ticker=symbol,
        action=final_action,  # type: ignore[arg-type]
        confidence=confidence,
        entry_zone=[rp["entry_min"], rp["entry_max"]],
        stop_loss=rp["stop_loss"],
        take_profit=[rp["tp1"], rp["tp2"], rp["tp3"]],
        time_horizon=ai_response.get("time_horizon", "DAY"),  # type: ignore[arg-type]
        rationale=list(ai_response.get("rationale", [])),
        risk_notes=risk_notes,
        evidence_links=validated_links,
        tags=list(ai_response.get("tags", [])),
        oracle_version=ORACLE_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("=" * 70)
    logger.info(
        "FINAL ORACLE DECISION for %s: %s (confidence=%d)",
        symbol,
        decision.action,
        decision.confidence,
    )
    logger.info(
        "Entry zone: %.2f - %.2f | SL: %.2f | TP: %.2f / %.2f / %.2f",
        decision.entry_zone[0],
        decision.entry_zone[1],
        decision.stop_loss,
        decision.take_profit[0],
        decision.take_profit[1],
        decision.take_profit[2],
    )
    logger.info("Time horizon: %s | Tags: %s", decision.time_horizon, ", ".join(decision.tags))
    logger.info("=" * 70)

    return decision


# ──────────────────────────────────────────────────────────
# CLI TOOL (Manual Smoke Test)
# ──────────────────────────────────────────────────────────


def _cli() -> None:
    """
    Simple CLI test harness for the Oracle.

    Usage:
        python brain.py TICKER
    """
    import sys

    from data_engine import calculate_hard_technicals, get_market_data
    from research_engine import get_god_tier_intel

    if len(sys.argv) < 2:
        print("Usage: python brain.py TICKER")
        print("Example: python brain.py AAPL")
        raise SystemExit(1)

    ticker = sys.argv[1].upper()

    logger.info("CLI Oracle run for %s", ticker)

    intel = get_god_tier_intel(ticker)
    df = get_market_data(ticker, period="3mo", interval="1d")
    technicals = calculate_hard_technicals(df, ticker=ticker, timeframe="1d")

    decision = consult_oracle(
        ticker=ticker,
        intel=intel,
        technicals=technicals,
        regime="UNKNOWN",
        events=[],
    )

    # Use logger instead of print as per constraints
    logger.info("CLI Result: %s", decision.to_dict())


if __name__ == "__main__":
    _cli()
