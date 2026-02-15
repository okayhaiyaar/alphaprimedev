"""
============================================================
ALPHA-PRIME v2.0 - Alerts & Notifications
============================================================
Module 5: Multi-channel notification system.

Primary Channel: Discord webhooks (rich embeds)
Optional: Email (SendGrid), SMS (Twilio)

Sends alerts for:
- High-confidence trading signals (BUY/SELL with confidence >= threshold)
- Circuit breaker triggers
- System errors (critical only)

Usage:
    from alerts import send_discord_alert

    result = send_discord_alert(
        decision=oracle_decision.to_dict(),
        portfolio_summary={"cash": 10000, "positions": 3, "total_value": 12500},
    )

    if not result.success:
        logger.error("Alert failed: %s", result.error or result.message)

Features:
- Rich Discord embeds with color coding
- Rate limiting (1 alert per ticker+action per 30 min)
- Retry logic with exponential backoff
- Confidence threshold filtering
- Graceful degradation (no crashes on failure)
============================================================
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional

import requests
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

# Rate limiting cache
alert_cache = Cache(f"{settings.cache_dir}/alerts")


# ──────────────────────────────────────────────────────────
# DATA SCHEMAS
# ──────────────────────────────────────────────────────────


@dataclass
class AlertResult:
    """
    Result of an alert send operation.

    Attributes:
        success: Whether the alert was successfully sent.
        channel: Channel identifier (e.g., "discord", "email", "sms").
        message: Human-readable summary of what happened.
        timestamp_utc: ISO timestamp when the result was recorded.
        error: Optional error message on failure.
    """

    success: bool
    channel: str
    message: str
    timestamp_utc: str
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────
# RATE LIMITING
# ──────────────────────────────────────────────────────────


def should_send_alert(
    ticker: str,
    action: str,
    cooldown_minutes: int = 30,
) -> bool:
    """
    Determine whether an alert should be sent based on per-ticker cooldown.

    Prevents alert spam by enforcing a cooldown per (ticker, action) pair.

    Args:
        ticker: Stock symbol.
        action: "BUY" | "SELL" | "WAIT".
        cooldown_minutes: Minimum time between alerts in minutes.

    Returns:
        True if an alert should be sent now; False otherwise.
    """
    key = f"alert_sent::{ticker.upper()}::{action}"
    last_sent = alert_cache.get(key)
    if last_sent is None:
        return True

    try:
        last_time = datetime.fromisoformat(last_sent)
    except Exception:  # noqa: BLE001
        return True

    elapsed = datetime.now(timezone.utc) - last_time
    minutes = elapsed.total_seconds() / 60.0

    if minutes < cooldown_minutes:
        logger.info(
            "Alert rate limit for %s %s: last sent %.1f minutes ago (cooldown=%d).",
            ticker,
            action,
            minutes,
            cooldown_minutes,
        )
        return False

    return True


def mark_alert_sent(
    ticker: str,
    action: str,
    ttl_minutes: int = 30,
) -> None:
    """
    Mark that an alert was sent, for rate limiting.

    Args:
        ticker: Stock symbol.
        action: "BUY" | "SELL" | "WAIT".
        ttl_minutes: TTL for rate-limit entry.
    """
    key = f"alert_sent::{ticker.upper()}::{action}"
    alert_cache.set(
        key,
        datetime.now(timezone.utc).isoformat(),
        expire=int(ttl_minutes * 60),
    )


# ──────────────────────────────────────────────────────────
# DISCORD EMBED FORMATTING
# ──────────────────────────────────────────────────────────


def _format_entry_zone(entry_zone: List[float]) -> str:
    """Format entry zone as a range string."""
    if not entry_zone or len(entry_zone) != 2:
        return "N/A"
    return f"${entry_zone[0]:.2f} - ${entry_zone[1]:.2f}"


def _format_take_profit(tp: List[float]) -> str:
    """Format take profit list."""
    if not tp or len(tp) != 3:
        return "N/A"
    return f"TP1: ${tp[0]:.2f}\nTP2: ${tp[1]:.2f}\nTP3: ${tp[2]:.2f}"


def build_discord_embed(
    decision: Dict[str, object],
    portfolio_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a rich Discord embed for a trading signal.

    Color coding:
    - BUY  → Green (0x00FF00)
    - SELL → Red   (0xFF0000)
    - WAIT → Gray  (0x808080)

    Args:
        decision: OracleDecision as a dictionary.
        portfolio_summary: Optional dictionary with portfolio stats.

    Returns:
        A dict representing a Discord embed payload.
    """
    action = str(decision.get("action", "WAIT"))
    confidence = int(decision.get("confidence", 0))
    ticker = str(decision.get("ticker", "")).upper()

    color_map = {
        "BUY": 0x00FF00,
        "SELL": 0xFF0000,
        "WAIT": 0x808080,
    }
    emoji_map = {
        "BUY": "🟢",
        "SELL": "🔴",
        "WAIT": "⚪",
    }

    color = color_map.get(action, 0x808080)
    emoji = emoji_map.get(action, "⚪")

    title = f"{emoji} {action} {ticker} — {confidence}% Confidence"

    description_lines: List[str] = []
    description_lines.append(f"**Time Horizon:** {decision.get('time_horizon', 'DAY')}")
    description_lines.append(f"**Oracle Version:** {decision.get('oracle_version', 'unknown')}")
    description_lines.append("")

    rationale = decision.get("rationale", []) or []
    description_lines.append("**Rationale:**")
    for point in rationale[:3]:
        text = str(point)
        if len(text) > 200:
            text = text[:197] + "..."
        description_lines.append(f"- {text}")
    description = "\n".join(description_lines)

    fields: List[Dict[str, object]] = []

    if action != "WAIT":
        entry_zone = decision.get("entry_zone", [0.0, 0.0])
        fields.append(
            {
                "name": "📍 Entry Zone",
                "value": _format_entry_zone(entry_zone),
                "inline": True,
            }
        )

    if action != "WAIT" and float(decision.get("stop_loss", 0.0)) > 0:
        fields.append(
            {
                "name": "🛑 Stop Loss",
                "value": f"${float(decision['stop_loss']):.2f}",
                "inline": True,
            }
        )

    if action != "WAIT":
        tp = decision.get("take_profit", [0.0, 0.0, 0.0])
        fields.append(
            {
                "name": "🎯 Take Profit",
                "value": _format_take_profit(tp),
                "inline": True,
            }
        )

    risk_notes = decision.get("risk_notes", []) or []
    if risk_notes:
        preview = risk_notes[:2]
        risk_text = "\n".join([f"⚠️ {str(note)[:200]}" for note in preview])
        fields.append(
            {
                "name": "⚠️ Risk Warnings",
                "value": risk_text or "N/A",
                "inline": False,
            }
        )

    tags = decision.get("tags", []) or []
    if tags:
        tag_text = " ".join(f"`{str(tag)}`" for tag in tags)
        fields.append(
            {
                "name": "🏷️ Tags",
                "value": tag_text,
                "inline": False,
            }
        )

    if portfolio_summary:
        cash = float(portfolio_summary.get("cash", 0.0))
        positions = int(portfolio_summary.get("positions", 0))
        total_value = float(portfolio_summary.get("total_value", 0.0))
        portfolio_text = (
            f"Cash: ${cash:,.2f}\n"
            f"Positions: {positions}\n"
            f"Total Value: ${total_value:,.2f}"
        )
        fields.append(
            {
                "name": "💼 Portfolio",
                "value": portfolio_text,
                "inline": True,
            }
        )

    ts = str(decision.get("generated_at_utc")) or datetime.now(timezone.utc).isoformat()

    footer_text = "ALPHA-PRIME v2.0 | AI-Powered Trading"
    evidence_links = decision.get("evidence_links", []) or []
    if evidence_links:
        footer_text += f" | {len(evidence_links)} sources"

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": footer_text},
        "timestamp": ts,
    }
    return embed


# ──────────────────────────────────────────────────────────
# DISCORD WEBHOOK
# ──────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def send_discord_webhook(embed: Dict[str, object]) -> bool:
    """
    Send an embed to the configured Discord webhook with retry logic.

    Args:
        embed: Discord embed object.

    Returns:
        True if the request succeeded (2xx); False if disabled or invalid URL.

    Raises:
        requests.exceptions.RequestException: On non-recoverable HTTP errors.
    """
    webhook_url = settings.discord_webhook_url
    if not webhook_url:
        logger.warning("Discord webhook URL not configured; skipping alert.")
        return False

    payload = {
        "embeds": [embed],
        "username": "ALPHA-PRIME",
        "avatar_url": "https://avatars.githubusercontent.com/u/135406289",  # Optional
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(
            "Discord alert sent successfully (status=%d).",
            response.status_code,
        )
        return True
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 429:
            logger.error("Discord returned HTTP 429 (rate limit).")
            raise
        if status == 404:
            logger.error("Discord webhook returned 404 (invalid/removed URL).")
            return False
        logger.error("Discord HTTP error (%s): %s", status, exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Error sending Discord webhook: %s", exc)
        raise


def send_discord_alert(
    decision: Dict[str, object],
    portfolio_summary: Optional[Dict[str, object]] = None,
    force_send: bool = False,
) -> AlertResult:
    """
    Send a trading signal alert to Discord.

    This is the main entry point for Discord notifications.

    Filtering rules (when force_send=False):
      - Only send if confidence >= settings.alert_min_confidence.
      - For WAIT actions, only alert if confidence >= 90.
      - Enforce rate limit per (ticker, action) with cooldown.

    Args:
        decision: OracleDecision as dict (from brain.py).
        portfolio_summary: Optional portfolio statistics.
        force_send: If True, bypass filtering and rate limiting.

    Returns:
        AlertResult describing the outcome.
    """
    ticker = str(decision.get("ticker", "")).upper()
    action = str(decision.get("action", "WAIT"))
    confidence = int(decision.get("confidence", 0))
    now = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Processing Discord alert: %s %s (confidence=%d).",
        ticker,
        action,
        confidence,
    )

    if not force_send:
        threshold = getattr(settings, "alert_min_confidence", 80)
        if confidence < threshold:
            logger.info(
                "Discord alert skipped for %s: confidence %d < threshold %d.",
                ticker,
                confidence,
                threshold,
            )
            return AlertResult(
                success=False,
                channel="discord",
                message=f"Below confidence threshold ({confidence} < {threshold})",
                timestamp_utc=now,
            )

        if action == "WAIT" and confidence < 90:
            logger.info("Discord alert skipped for %s: WAIT with low confidence.", ticker)
            return AlertResult(
                success=False,
                channel="discord",
                message="WAIT action; alert suppressed.",
                timestamp_utc=now,
            )

        if not should_send_alert(ticker, action):
            return AlertResult(
                success=False,
                channel="discord",
                message="Rate limit active; alert suppressed.",
                timestamp_utc=now,
            )

    try:
        embed = build_discord_embed(decision, portfolio_summary)
        success = send_discord_webhook(embed)
        if success:
            mark_alert_sent(ticker, action)
            return AlertResult(
                success=True,
                channel="discord",
                message=f"Alert sent for {ticker} {action}",
                timestamp_utc=now,
            )

        return AlertResult(
            success=False,
            channel="discord",
            message="Webhook disabled or invalid.",
            timestamp_utc=now,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send Discord alert for %s: %s", ticker, exc, exc_info=True)
        return AlertResult(
            success=False,
            channel="discord",
            message="Exception while sending alert.",
            timestamp_utc=now,
            error=str(exc),
        )


# ──────────────────────────────────────────────────────────
# OPTIONAL: EMAIL ALERTS (SendGrid)
# ──────────────────────────────────────────────────────────


def send_email_alert(decision: Dict[str, object]) -> AlertResult:
    """
    Send a trading signal via email using SendGrid.

    Requires SENDGRID_API_KEY in environment and at least one recipient.

    Args:
        decision: OracleDecision dict.

    Returns:
        AlertResult describing outcome.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        api_key = os.getenv("SENDGRID_API_KEY")
        to_email = os.getenv("ALPHAPRIME_ALERT_EMAIL")

        if not api_key or not to_email:
            logger.warning("SendGrid not fully configured; email alert skipped.")
            return AlertResult(
                success=False,
                channel="email",
                message="SendGrid not configured.",
                timestamp_utc=now,
            )

        subject = f"ALPHA-PRIME: {decision.get('action', 'WAIT')} {decision.get('ticker', '')}"
        entry_zone = decision.get("entry_zone", [0.0, 0.0])
        rationale_html = "".join(
            f"<li>{str(r)}</li>" for r in (decision.get("rationale") or [])[:5]
        )

        html_content = f"""
        <h2>{decision.get('action', 'WAIT')} {decision.get('ticker', '')}</h2>
        <p><strong>Confidence:</strong> {int(decision.get('confidence', 0))}%</p>
        <p><strong>Entry Zone:</strong> {_format_entry_zone(entry_zone)}</p>
        <p><strong>Stop Loss:</strong> ${float(decision.get('stop_loss', 0.0)):.2f}</p>
        <h3>Rationale</h3>
        <ul>{rationale_html}</ul>
        """

        message = Mail(
            from_email="alerts@alpha-prime.dev",
            to_emails=to_email,
            subject=subject,
            html_content=html_content,
        )

        sg = SendGridAPIClient(api_key)
        response = sg.send(message)

        logger.info("Email alert sent (status=%s).", response.status_code)
        return AlertResult(
            success=True,
            channel="email",
            message=f"Email sent (status={response.status_code})",
            timestamp_utc=now,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Email alert failed: %s", exc, exc_info=True)
        return AlertResult(
            success=False,
            channel="email",
            message="Email send failed.",
            timestamp_utc=now,
            error=str(exc),
        )


# ──────────────────────────────────────────────────────────
# OPTIONAL: SMS ALERTS (Twilio)
# ──────────────────────────────────────────────────────────


def send_sms_alert(decision: Dict[str, object]) -> AlertResult:
    """
    Send a trading signal via SMS using Twilio.

    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER,
    and ALPHAPRIME_ALERT_PHONE in environment.

    Args:
        decision: OracleDecision dict.

    Returns:
        AlertResult describing outcome.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        from twilio.rest import Client as TwilioClient

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_phone = os.getenv("TWILIO_PHONE_NUMBER")
        to_phone = os.getenv("ALPHAPRIME_ALERT_PHONE")

        if not all([account_sid, auth_token, from_phone, to_phone]):
            logger.warning("Twilio not fully configured; SMS alert skipped.")
            return AlertResult(
                success=False,
                channel="sms",
                message="Twilio not configured.",
                timestamp_utc=now,
            )

        entry_zone = decision.get("entry_zone", [0.0, 0.0])
        message_body = (
            f"ALPHA-PRIME: {decision.get('action', 'WAIT')} {decision.get('ticker', '')}\n"
            f"Conf: {int(decision.get('confidence', 0))}%\n"
            f"Entry: {_format_entry_zone(entry_zone)}\n"
            f"SL: ${float(decision.get('stop_loss', 0.0)):.2f}"
        )

        client = TwilioClient(account_sid, auth_token)
        msg = client.messages.create(body=message_body, from_=from_phone, to=to_phone)

        logger.info("SMS alert sent (SID=%s).", msg.sid)
        return AlertResult(
            success=True,
            channel="sms",
            message=f"SMS sent (SID={msg.sid})",
            timestamp_utc=now,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("SMS alert failed: %s", exc, exc_info=True)
        return AlertResult(
            success=False,
            channel="sms",
            message="SMS send failed.",
            timestamp_utc=now,
            error=str(exc),
        )


# ──────────────────────────────────────────────────────────
# MULTI-CHANNEL DISPATCHER
# ──────────────────────────────────────────────────────────


def send_multi_channel_alert(
    decision: Dict[str, object],
    channels: List[str] = ("discord",),
    portfolio_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, AlertResult]:
    """
    Send an alert to multiple channels.

    Args:
        decision: OracleDecision dict.
        channels: List of channels to send to ("discord", "email", "sms").
        portfolio_summary: Optional portfolio summary for Discord.

    Returns:
        Mapping of channel name → AlertResult.
    """
    results: Dict[str, AlertResult] = {}

    for channel in channels:
        if channel == "discord":
            results["discord"] = send_discord_alert(
                decision=decision,
                portfolio_summary=portfolio_summary,
            )
        elif channel == "email":
            results["email"] = send_email_alert(decision)
        elif channel == "sms":
            results["sms"] = send_sms_alert(decision)
        else:
            logger.warning("Unknown alert channel requested: %s", channel)

    success_count = sum(1 for r in results.values() if r.success)
    logger.info(
        "Multi-channel alert summary: %d/%d channels succeeded.",
        success_count,
        len(channels),
    )
    return results


# ──────────────────────────────────────────────────────────
# SYSTEM ERROR ALERTS
# ──────────────────────────────────────────────────────────


def send_error_alert(
    error_message: str,
    severity: Literal["WARNING", "ERROR", "CRITICAL"],
) -> bool:
    """
    Send a system error notification to Discord.

    Intended for critical system failures or circuit breaker events.

    Args:
        error_message: Description of the error or event.
        severity: "WARNING" | "ERROR" | "CRITICAL".

    Returns:
        True if sent successfully; False otherwise.
    """
    if not settings.discord_webhook_url:
        return False

    color_map = {
        "WARNING": 0xFFA500,
        "ERROR": 0xFF0000,
        "CRITICAL": 0x8B0000,
    }

    embed = {
        "title": f"🚨 SYSTEM {severity}",
        "description": error_message[:2000],
        "color": color_map.get(severity, 0xFF0000),
        "footer": {"text": "ALPHA-PRIME System Alert"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        return send_discord_webhook(embed)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to send system error alert: %s", exc, exc_info=True)
        return False


# ──────────────────────────────────────────────────────────
# CLI TOOL & TESTING
# ──────────────────────────────────────────────────────────


def _cli() -> None:
    """
    Simple CLI test tool for Discord alerts.

    Usage:
        python alerts.py
    """
    if not settings.discord_webhook_url:
        logger.error("DISCORD_WEBHOOK_URL not configured; cannot test alerts.")
        raise SystemExit(1)

    mock_decision: Dict[str, object] = {
        "ticker": "AAPL",
        "action": "BUY",
        "confidence": 88,
        "entry_zone": [150.0, 152.0],
        "stop_loss": 145.0,
        "take_profit": [155.0, 160.0, 165.0],
        "time_horizon": "SWING",
        "rationale": [
            "RSI has recovered from oversold region, indicating potential mean reversion.",
            "Trend starting to turn up with price above 20 EMA.",
            "No major negative fundamental red flags in recent filings.",
        ],
        "risk_notes": [
            "Earnings in ~2 weeks; expect elevated volatility.",
            "General macro uncertainty due to upcoming Fed commentary.",
        ],
        "evidence_links": [
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193",
            "https://finance.yahoo.com/quote/AAPL",
        ],
        "tags": ["oversold_bounce", "tech_sector"],
        "oracle_version": "v2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    mock_portfolio = {
        "cash": 8500.0,
        "positions": 2,
        "total_value": 10250.0,
    }

    logger.info("Sending test Discord alert for AAPL...")
    result = send_discord_alert(
        decision=mock_decision,
        portfolio_summary=mock_portfolio,
        force_send=True,
    )
    if result.success:
        logger.info("Test alert sent successfully: %s", result.message)
    else:
        logger.error("Test alert failed: %s (%s)", result.message, result.error)


if __name__ == "__main__":
    _cli()
