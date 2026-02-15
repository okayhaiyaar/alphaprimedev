"""
============================================================
ALPHA-PRIME v2.0 - Notifier (Alerting & Notification System)
============================================================

Multi-channel, severity-aware notification layer for trading systems.

Key features:
- 8+ delivery channels (Telegram, Discord, Slack, SMS, Email, Webhook, Console, File log).
- Async, non-blocking interface built on asyncio. [web:438][web:439][web:440][web:448][web:451]
- Severity-based escalation and adaptive retry.
- Token-bucket rate limiting and deduplication. [web:443][web:445][web:450][web:395]
- Offline queue and dead-letter handling (best-effort).
- Structured logs and metrics hooks for observability.

This module provides a robust baseline that can be wired to
real credentials and endpoints in production deployments.

============================================================
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import aiohttp
import pandas as pd

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

Severity = Literal["INFO", "WARN", "ALERT", "CRITICAL", "EMERGENCY"]

CHANNELS_PRIMARY = ["telegram", "discord", "slack", "sms"]
CHANNELS_SECONDARY = ["email", "webhook", "console", "file"]
CHANNELS_FUTURE = ["pagerduty", "msteams", "pushover"]


# ──────────────────────────────────────────────────────────
# CONFIG & DATA STRUCTURES
# ──────────────────────────────────────────────────────────


@dataclass
class NotifierConfig:
    """
    Configuration for Notifier.

    Attributes:
        channels: Enabled channels (subset of known channels).
        telegram_token: Telegram bot token.
        telegram_chat_id: Telegram chat/channel ID.
        discord_webhook: Discord webhook URL.
        slack_webhook: Slack webhook URL.
        twilio_sid: Twilio account SID.
        twilio_token: Twilio auth token.
        twilio_phone: Twilio from-phone.
        email_smtp: SMTP server.
        email_from: Sender email.
        email_to: Recipient list.
        rate_limit_messages_per_hour: Max messages/hour per channel.
        dedupe_window_minutes: Deduplication window.
        rich_text: Enable emoji/formatting.
        log_dir: Directory for file logs and offline queue.
    """

    channels: List[str] = field(default_factory=lambda: ["console", "file"])
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    discord_webhook: str | None = None
    slack_webhook: str | None = None
    twilio_sid: str | None = None
    twilio_token: str | None = None
    twilio_phone: str | None = None
    email_smtp: str | None = None
    email_from: str | None = None
    email_to: List[str] = field(default_factory=list)
    rate_limit_messages_per_hour: int = 60
    dedupe_window_minutes: int = 5
    rich_text: bool = True
    log_dir: Path = Path(os.path.expanduser("~/.alpha_prime/notifier"))


@dataclass
class RiskEvent:
    name: str
    severity: Severity
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    cpu_usage: float
    mem_usage: float
    api_rate_limited: bool
    cache_utilization: float
    model_drift: bool
    max_drawdown: float


@dataclass
class OfflineMessage:
    timestamp: datetime
    severity: Severity
    title: str
    message: str
    channels: List[str]


# ──────────────────────────────────────────────────────────
# TOKEN BUCKET FOR RATE LIMITING
# ──────────────────────────────────────────────────────────


class TokenBucket:
    """
    Simple async token bucket limiter per channel. [web:443][web:445][web:450][web:395]

    capacity: max messages per hour.
    refill_rate: tokens per second.
    """

    def __init__(self, capacity: int, window_seconds: int = 3600) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.window_seconds = window_seconds
        self.refill_rate = capacity / window_seconds
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            if elapsed > 0:
                self.tokens = min(
                    self.capacity, self.tokens + elapsed * self.refill_rate
                )
                self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# Deduplication cache (in-memory; extend to Redis/disk if needed)
class DedupeCache:
    def __init__(self, window_minutes: int) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.entries: Dict[str, datetime] = {}

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def seen(self, key: str) -> bool:
        now = self._now()
        cutoff = now - self.window
        self.entries = {k: t for k, t in self.entries.items() if t >= cutoff}
        if key in self.entries:
            return True
        self.entries[key] = now
        return False


# ──────────────────────────────────────────────────────────
# CHANNEL CLIENTS (SKELETON IMPLEMENTATIONS)
# ──────────────────────────────────────────────────────────


class BaseChannel:
    name: str = "base"

    def __init__(self, config: NotifierConfig, session: aiohttp.ClientSession) -> None:
        self.config = config
        self.session = session

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:  # noqa: D401
        """Send message (override in subclasses)."""
        raise NotImplementedError


class TelegramChannel(BaseChannel):
    name = "telegram"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        text = f"{title}\n{message}"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown" if self.config.rich_text else None,
        }
        async with self.session.post(url, json=payload) as resp:
            if resp.status >= 400:
                logger.warning("Telegram send failed: %s", await resp.text())


class DiscordChannel(BaseChannel):
    name = "discord"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        if not self.config.discord_webhook:
            return
        payload = {"content": f"**{title}**\n{message}"}
        async with self.session.post(self.config.discord_webhook, json=payload) as resp:
            if resp.status >= 400:
                logger.warning("Discord send failed: %s", await resp.text())


class SlackChannel(BaseChannel):
    name = "slack"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        if not self.config.slack_webhook:
            return
        payload = {"text": f"*{title}*\n{message}"}
        async with self.session.post(self.config.slack_webhook, json=payload) as resp:
            if resp.status >= 400:
                logger.warning("Slack send failed: %s", await resp.text())


class SMSChannel(BaseChannel):
    name = "sms"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        # Stub: integrate with Twilio REST API in production.
        if not (self.config.twilio_sid and self.config.twilio_token and self.config.twilio_phone):
            return
        logger.info("SMSChannel stub sending: %s - %s", title, message[:80])


class EmailChannel(BaseChannel):
    name = "email"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        # Stub: use aiosmtplib / SMTP in production.
        if not (self.config.email_smtp and self.config.email_from and self.config.email_to):
            return
        logger.info("EmailChannel stub sending email '%s' to %s", title, self.config.email_to)


class WebhookChannel(BaseChannel):
    name = "webhook"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        url = getattr(self.config, "generic_webhook", None)
        if not url:
            return
        payload = {"severity": severity, "title": title, "message": message}
        async with self.session.post(url, json=payload) as resp:
            if resp.status >= 400:
                logger.warning("Webhook send failed: %s", await resp.text())


class ConsoleChannel(BaseChannel):
    name = "console"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        print(f"[{severity}] {title} - {message}")


class FileChannel(BaseChannel):
    name = "file"

    def __init__(self, config: NotifierConfig, session: aiohttp.ClientSession) -> None:
        super().__init__(config, session)
        self.log_path = config.log_dir / "notifier.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "title": title,
            "message": message,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# Future stub channels
class PagerDutyChannel(BaseChannel):
    name = "pagerduty"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        logger.info("PagerDuty stub: %s - %s", title, message[:80])


class MSTeamsChannel(BaseChannel):
    name = "msteams"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        logger.info("MS Teams stub: %s - %s", title, message[:80])


class PushoverChannel(BaseChannel):
    name = "pushover"

    async def send(self, severity: Severity, title: str, message: str, attachments: List[Any] | None = None) -> None:
        logger.info("Pushover stub: %s - %s", title, message[:80])


CHANNEL_REGISTRY = {
    "telegram": TelegramChannel,
    "discord": DiscordChannel,
    "slack": SlackChannel,
    "sms": SMSChannel,
    "email": EmailChannel,
    "webhook": WebhookChannel,
    "console": ConsoleChannel,
    "file": FileChannel,
    "pagerduty": PagerDutyChannel,
    "msteams": MSTeamsChannel,
    "pushover": PushoverChannel,
}


# ──────────────────────────────────────────────────────────
# NOTIFIER CORE
# ──────────────────────────────────────────────────────────


class Notifier:
    """
    Multi-channel notifier with severity-based escalation, rate limiting,
    deduplication, and offline queue.
    """

    def __init__(self, config: NotifierConfig) -> None:
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
        self.channels: Dict[str, BaseChannel] = {}
        self.buckets: Dict[str, TokenBucket] = {}
        self.dedupe = DedupeCache(window_minutes=config.dedupe_window_minutes)
        self.offline_queue_path = self.config.log_dir / "offline_queue.jsonl"
        self.dead_letter_path = self.config.log_dir / "dead_letter.jsonl"
        self._warn_count = 0
        self._alert_streak = 0

    async def __aenter__(self) -> "Notifier":
        self.session = aiohttp.ClientSession()
        for ch_name in self.config.channels:
            cls = CHANNEL_REGISTRY.get(ch_name)
            if cls is None:
                continue
            self.channels[ch_name] = cls(self.config, self.session)
            self.buckets[ch_name] = TokenBucket(
                capacity=self.config.rate_limit_messages_per_hour
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            await self.session.close()

    # ---- public API --------------------------------------------------------

    async def send(
        self,
        severity: Severity,
        title: str,
        message: str,
        attachments: Optional[List[Any]] = None,
    ) -> None:
        """
        Send a message with severity and title across the configured channels.

        Non-blocking from caller perspective; internal fan-out is awaited
        but designed to be lightweight.
        """
        severity = await self._apply_escalation(severity)
        key = self._dedupe_key(severity, title, message)
        if self.dedupe.seen(key):
            logger.debug("Notifier deduped message: %s", key)
            return

        targets = self._channels_for_severity(severity)
        tasks = []
        for ch_name in targets:
            ch = self.channels.get(ch_name)
            if ch is None:
                continue
            bucket = self.buckets.get(ch_name)
            tasks.append(self._send_via_channel(ch_name, ch, bucket, severity, title, message, attachments))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def alert_trading_signal(self, symbol: str, signal: Any, confidence: float) -> None:
        emoji = "🚀" if confidence >= 0.8 else "📈"
        title = f"{emoji} {symbol} {str(signal).upper()} SIGNAL ({confidence*100:.0f}% confidence)"
        msg = f"Symbol: {symbol}\nSignal: {signal}\nConfidence: {confidence:.2f}"
        await self.send("ALERT", title, msg)

    async def alert_risk_event(self, event: RiskEvent) -> None:
        title = f"Risk Event: {event.name}"
        details = json.dumps(event.details, default=str)
        await self.send(event.severity, title, details)

    async def alert_system_health(self, health: SystemHealth) -> None:
        severity = self._severity_from_health(health)
        title = f"System Health ({severity})"
        msg = (
            f"CPU: {health.cpu_usage:.1f}% | MEM: {health.mem_usage:.1f}% | "
            f"Cache: {health.cache_utilization:.1f}% | Max DD: {health.max_drawdown*100:.1f}%\n"
            f"API rate limited: {health.api_rate_limited} | Model drift: {health.model_drift}"
        )
        await self.send(severity, title, msg)

    # ---- internal helpers --------------------------------------------------

    def _dedupe_key(self, severity: Severity, title: str, message: str) -> str:
        h = hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
        return f"{severity}:{title}:{h}"

    def _channels_for_severity(self, severity: Severity) -> List[str]:
        base = [c for c in self.config.channels if c in CHANNELS_SECONDARY]
        prim = [c for c in self.config.channels if c in CHANNELS_PRIMARY]
        fut = [c for c in self.config.channels if c in CHANNELS_FUTURE]

        if severity == "INFO":
            return ["console"] + ([c for c in base if c != "console"])
        if severity == "WARN":
            return ["console"] + [c for c in prim if c in ("telegram", "discord")] + base
        if severity == "ALERT":
            return prim + base
        if severity == "CRITICAL":
            return prim + base + fut
        if severity == "EMERGENCY":
            return prim + base + fut
        return base

    async def _send_via_channel(
        self,
        ch_name: str,
        channel: BaseChannel,
        bucket: Optional[TokenBucket],
        severity: Severity,
        title: str,
        message: str,
        attachments: Optional[List[Any]],
    ) -> None:
        if bucket is not None:
            allowed = await bucket.acquire()
            if not allowed and severity in ("INFO", "WARN"):
                logger.debug("Rate limit drop on %s (%s)", ch_name, severity)
                return

        offline_entry = OfflineMessage(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            title=title,
            message=message,
            channels=[ch_name],
        )

        for attempt in range(3):
            try:
                await channel.send(severity, title, message, attachments)
                logger.info(
                    '{"event": "notifier_send", "channel": "%s", "severity": "%s"}',
                    ch_name,
                    severity,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Channel %s send failed (attempt %d): %s", ch_name, attempt + 1, exc)
                await self._write_offline(offline_entry)
                await asyncio.sleep(min(5 * (attempt + 1), 60))
        await self._write_dead_letter(offline_entry)

    async def _apply_escalation(self, severity: Severity) -> Severity:
        if severity == "WARN":
            self._warn_count += 1
            if self._warn_count >= 3:
                self._warn_count = 0
                severity = "ALERT"
        elif severity == "ALERT":
            self._alert_streak += 1
            if self._alert_streak >= 2:
                severity = "CRITICAL"
        else:
            self._alert_streak = 0
        return severity

    def _severity_from_health(self, health: SystemHealth) -> Severity:
        if health.max_drawdown <= -0.10 and health.model_drift:
            return "EMERGENCY"
        if health.cpu_usage > 90 or health.mem_usage > 85 or health.cache_utilization > 95:
            return "ALERT"
        if health.api_rate_limited:
            return "WARN"
        return "INFO"

    async def _write_offline(self, msg: OfflineMessage) -> None:
        entry = dataclasses.asdict(msg)
        with self.offline_queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    async def _write_dead_letter(self, msg: OfflineMessage) -> None:
        entry = dataclasses.asdict(msg)
        with self.dead_letter_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# ──────────────────────────────────────────────────────────
# RICH FORMATTING HELPERS
# ──────────────────────────────────────────────────────────


def format_dataframe_table(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into a markdown-like table for Slack/Discord.
    """
    if df.empty:
        return "(no data)"
    headers = "| " + " | ".join(df.columns.astype(str)) + " |"
    sep = "|" + "|".join(["--------"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([headers, sep] + rows)


def format_trading_signal_message(
    symbol: str,
    signal: Any,
    confidence: float,
    price: Optional[float] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    rr: Optional[float] = None,
    position_size_pct: Optional[float] = None,
    sharpe: Optional[float] = None,
) -> str:
    emoji = "🚀" if confidence >= 0.8 else "📈"
    lines = [f"{emoji} {symbol} {str(signal).upper()} SIGNAL ({confidence*100:.0f}% confidence)"]
    if price is not None:
        lines.append(f"Price: ${price:.2f}")
    if target is not None and stop is not None:
        lines[-1] += f" | Target: ${target:.2f} | Stop: ${stop:.2f}"
    if rr is not None and position_size_pct is not None:
        lines.append(f"R:R = 1:{rr:.2f} | Position Size: {position_size_pct:.1f}%")
    if sharpe is not None:
        lines.append(f"[📈 Performance: Sharpe {sharpe:.2f}]")
    return "\n".join(lines)


def format_risk_event_message(
    name: str,
    drawdown_pct: Optional[float] = None,
    drawdown_value: Optional[float] = None,
    portfolio: Optional[str] = None,
    threshold_pct: Optional[float] = None,
) -> str:
    lines = [f"⚠️ {name}"]
    if drawdown_pct is not None and drawdown_value is not None:
        lines.append(f"MAX DRAWDOWN: {drawdown_pct:.1f}% (${drawdown_value:,.0f})")
    if portfolio is not None and threshold_pct is not None:
        lines.append(f"Portfolio: {portfolio} | Trigger: {threshold_pct:.1f}% threshold")
    return "\n".join(lines)


def format_drift_message(
    model_name: str,
    data_psi: float,
    error_ratio: float,
    retrain_score: float,
    eta_minutes: Optional[float] = None,
) -> str:
    lines = [f"🔄 MODEL DRIFT DETECTED ({model_name})"]
    lines.append(f"Data PSI: {data_psi:.2f} | Error Ratio: {error_ratio:.2f}")
    lines.append(f"Retraining triggered | Retrain score: {retrain_score:.2f}")
    if eta_minutes is not None:
        lines[-1] += f" | ETA: {eta_minutes:.0f}min"
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def _default_config() -> NotifierConfig:
    return NotifierConfig(
        channels=["console", "file", "telegram", "discord", "sms", "email"],
        telegram_token=os.getenv("TELEGRAM_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        discord_webhook=os.getenv("DISCORD_WEBHOOK"),
        slack_webhook=os.getenv("SLACK_WEBHOOK"),
        twilio_sid=os.getenv("TWILIO_SID"),
        twilio_token=os.getenv("TWILIO_TOKEN"),
        twilio_phone=os.getenv("TWILIO_PHONE"),
        email_smtp=os.getenv("EMAIL_SMTP"),
        email_from=os.getenv("EMAIL_FROM"),
        email_to=os.getenv("EMAIL_TO", "").split(",") if os.getenv("EMAIL_TO") else [],
    )


async def _cli_test() -> None:
    cfg = _default_config()
    async with Notifier(cfg) as notifier:
        print("🟢 INFO: Test message → Console")
        await notifier.send("INFO", "Test INFO", "This is an info test.")
        print("🟡 WARN: Test → Discord/Telegram")
        await notifier.send("WARN", "Test WARN", "This is a warn test.")
        print("🟠 ALERT: Test → All channels")
        await notifier.send("ALERT", "Test ALERT", "This is an alert test.")
        print("🔴 CRITICAL: Test → SMS + All")
        await notifier.send("CRITICAL", "Test CRITICAL", "This is a critical test.")


def _cli_config() -> None:
    cfg = _default_config()
    print("Channels:", ",".join(cfg.channels))
    print(
        f"Rate limit: {cfg.rate_limit_messages_per_hour}/hr | "
        f"Dedupe: {cfg.dedupe_window_minutes}min"
    )
    print(f"Rich text: {cfg.rich_text} | Log dir: {cfg.log_dir}")


def _cli_replay(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"No log file at {p}")
        return
    with p.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Replaying {len(lines)} alerts from {p}...")
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("ts") or obj.get("timestamp")
        sev = obj.get("severity")
        title = obj.get("title")
        msg = obj.get("message")
        print(f"{ts} [{sev}] {title} - {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Notifier CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Send test notifications across channels.")
    sub.add_parser("config", help="Show notifier configuration.")
    replay_p = sub.add_parser("replay", help="Replay notifications from a log file.")
    replay_p.add_argument("path", type=str)

    args = parser.parse_args()
    if args.command == "test":
        asyncio.run(_cli_test())
    elif args.command == "config":
        _cli_config()
    elif args.command == "replay":
        _cli_replay(args.path)


if __name__ == "__main__":
    main()
