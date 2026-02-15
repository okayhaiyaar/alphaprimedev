"""
ALPHA-PRIME v2.0 - Production Watchdog Monitor
==============================================

Always-on production watchdog for:

- Critical trading invariants (position limits, model health, risk)
- Automated recovery (restart services, reduce positions, pause signals)
- Escalated alerting (log → chat → email → SMS → PagerDuty)
- Zero-downtime async monitoring loop
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Severity & escalation
# ---------------------------------------------------------------------------

class Severity(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class EscalationLevel:
    name: str
    min_severity: Severity
    cooldown: timedelta
    channels: List[str]  # e.g. ["log", "discord", "email"]


@dataclass
class CheckResult:
    healthy: bool
    severity: Severity
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    remediation_actions: List[str] = field(default_factory=list)


class Check(ABC):
    """Base interface for watchdog checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def severity(self) -> Severity:
        ...

    @property
    def auto_remediate(self) -> bool:
        return True

    @abstractmethod
    async def run(self) -> CheckResult:
        ...


@dataclass
class WatchdogConfig:
    check_interval: float = 30.0
    alert_cooldown: float = 300.0
    max_consecutive_failures: int = 3
    auto_remediate: bool = True
    escalation_levels: List[EscalationLevel] = field(default_factory=list)
    redis_heartbeat: bool = True
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Notifier & metrics stubs (replace with real integrations)
# ---------------------------------------------------------------------------

class Notifier:
    """Simple, pluggable notifier for escalated alerts."""

    def __init__(self) -> None:
        self._last_sent: Dict[str, datetime] = {}

    async def send(self, level: EscalationLevel, severity: Severity, title: str, message: str) -> None:
        now = datetime.utcnow()
        key = f"{level.name}:{severity.value}"
        last = self._last_sent.get(key)
        if last and now - last < level.cooldown:
            return
        self._last_sent[key] = now
        payload = {
            "event": "watchdog_alert",
            "level": level.name,
            "severity": severity.value,
            "title": title,
            "message": message,
            "timestamp": now.isoformat() + "Z",
        }
        print(json.dumps(payload))


class MetricsSink:
    """Stub for Prometheus-style metrics."""

    def inc(self, name: str, labels: Dict[str, str]) -> None:
        payload = {"event": "metric_inc", "name": name, "labels": labels}
        print(json.dumps(payload))

    def observe(self, name: str, value: float, labels: Dict[str, str]) -> None:
        payload = {
            "event": "metric_observe",
            "name": name,
            "value": value,
            "labels": labels,
        }
        print(json.dumps(payload))


# ---------------------------------------------------------------------------
# Built-in checks (12+ invariants)
# NOTE: All remediation/actions are placeholders; wire to real systems.
# ---------------------------------------------------------------------------

async def _fake_io(delay: float = 0.01) -> None:
    await asyncio.sleep(delay)


class PositionLimitCheck(Check):
    @property
    def name(self) -> str:
        return "position_limits"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    async def run(self) -> CheckResult:
        await _fake_io()
        oversized = []  # real impl: query positions where pct > 0.20
        actions: List[str] = []
        if oversized:
            # real impl: auto_reduce_position(...)
            actions.append(f"Reduced {len(oversized)} oversized positions to target limits")
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="Position limits exceeded",
                remediation_actions=actions,
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="All positions within limits",
        )


class MaxDrawdownCheck(Check):
    @property
    def name(self) -> str:
        return "max_drawdown"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    async def run(self) -> CheckResult:
        await _fake_io()
        current_dd = 0.05  # 5%
        threshold = 0.1    # 10%
        if current_dd > threshold:
            # real: trigger emergency stop / reduce exposure
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"Max drawdown {current_dd:.2%} exceeds threshold {threshold:.2%}",
                metrics={"current_drawdown": current_dd},
                remediation_actions=["Triggered emergency stop-loss"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Drawdown within limits",
            metrics={"current_drawdown": current_dd},
        )


class ModelDriftCheck(Check):
    @property
    def name(self) -> str:
        return "model_drift"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    async def run(self) -> CheckResult:
        await _fake_io()
        drift_score = 0.12  # 12%
        if drift_score > 0.2:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"Model drift score {drift_score:.2%} above threshold",
                metrics={"drift_score": drift_score},
                remediation_actions=[
                    "Paused strategy signals",
                    "Scheduled model retrain",
                ],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Model drift within acceptable bounds",
            metrics={"drift_score": drift_score},
        )


class ApiRateLimitCheck(Check):
    @property
    def name(self) -> str:
        return "api_rate_limits"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    async def run(self) -> CheckResult:
        await _fake_io()
        nearing_limits = False
        if nearing_limits:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="API rate limits nearing exhaustion",
                remediation_actions=["Switched to backup data provider"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="API rate limits OK",
        )


class BrokerConnectivityCheck(Check):
    @property
    def name(self) -> str:
        return "broker_connectivity"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    async def run(self) -> CheckResult:
        await _fake_io()
        connected = True
        if not connected:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="Primary broker unreachable",
                remediation_actions=["Failover to backup broker"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Broker connectivity OK",
        )


class DatabaseDeadlockCheck(Check):
    @property
    def name(self) -> str:
        return "db_deadlock"

    @property
    def severity(self) -> Severity:
        return Severity.CRITICAL

    async def run(self) -> CheckResult:
        await _fake_io()
        deadlocked = False
        if deadlocked:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="Database deadlock detected",
                remediation_actions=["Restarted DB connection pool"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="No DB deadlocks detected",
        )


class HighLatencyCheck(Check):
    @property
    def name(self) -> str:
        return "high_latency"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    async def run(self) -> CheckResult:
        await _fake_io()
        latency_ms = 50.0
        if latency_ms > 250.0:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"High latency detected ({latency_ms:.1f}ms)",
                metrics={"latency_ms": latency_ms},
                remediation_actions=["Scaled worker pool"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Latency within target range",
            metrics={"latency_ms": latency_ms},
        )


class CacheMissRateCheck(Check):
    @property
    def name(self) -> str:
        return "cache_miss_rate"

    @property
    def severity(self) -> Severity:
        return Severity.MEDIUM

    async def run(self) -> CheckResult:
        await _fake_io()
        miss_rate = 0.05
        if miss_rate > 0.2:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"Cache miss rate {miss_rate:.2%} too high",
                metrics={"miss_rate": miss_rate},
                remediation_actions=["Warmed critical cache keys"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Cache hit rate acceptable",
            metrics={"miss_rate": miss_rate},
        )


class SignalConfidenceCheck(Check):
    @property
    def name(self) -> str:
        return "signal_confidence_drop"

    @property
    def severity(self) -> Severity:
        return Severity.MEDIUM

    async def run(self) -> CheckResult:
        await _fake_io()
        avg_conf = 0.8
        if avg_conf < 0.5:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"Signal confidence dropped to {avg_conf:.2f}",
                metrics={"avg_confidence": avg_conf},
                remediation_actions=["Reduced position sizes for affected strategies"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Signal confidence stable",
            metrics={"avg_confidence": avg_conf},
        )


class CorrelationSpikeCheck(Check):
    @property
    def name(self) -> str:
        return "correlation_spike"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    async def run(self) -> CheckResult:
        await _fake_io()
        correlation = 0.6
        if correlation > 0.8:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"Correlation spike detected ({correlation:.2f})",
                metrics={"avg_correlation": correlation},
                remediation_actions=["Increased diversification across sectors"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Correlation within acceptable range",
            metrics={"avg_correlation": correlation},
        )


class NewsSentimentCheck(Check):
    @property
    def name(self) -> str:
        return "news_sentiment_shock"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    async def run(self) -> CheckResult:
        await _fake_io()
        sentiment = -0.1
        if sentiment < -0.5:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message=f"Negative news sentiment shock ({sentiment:.2f})",
                metrics={"sentiment": sentiment},
                remediation_actions=["Reduced risk exposure"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="News sentiment within normal range",
            metrics={"sentiment": sentiment},
        )


class ResourceExhaustionCheck(Check):
    @property
    def name(self) -> str:
        return "resource_exhaustion"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    async def run(self) -> CheckResult:
        await _fake_io()
        cpu = 0.4
        mem = 0.5
        if cpu > 0.9 or mem > 0.95:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="System resources near exhaustion",
                metrics={"cpu": cpu, "mem": mem},
                remediation_actions=["Alerted ops for scaling"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="System resources within safe bounds",
            metrics={"cpu": cpu, "mem": mem},
        )


class RollingSharpeCheck(Check):
    @property
    def name(self) -> str:
        return "rolling_sharpe_degradation"

    @property
    def severity(self) -> Severity:
        return Severity.MEDIUM

    async def run(self) -> CheckResult:
        await _fake_io()
        sharpe_30d = 1.2
        sharpe_365d = 1.5
        if sharpe_30d < sharpe_365d * 0.5:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="Rolling Sharpe degraded significantly",
                metrics={"sharpe_30d": sharpe_30d, "sharpe_365d": sharpe_365d},
                remediation_actions=["Flagged strategy for review"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Rolling Sharpe stable",
            metrics={"sharpe_30d": sharpe_30d, "sharpe_365d": sharpe_365d},
        )


class WinRateDeclineCheck(Check):
    @property
    def name(self) -> str:
        return "win_rate_decline"

    @property
    def severity(self) -> Severity:
        return Severity.MEDIUM

    async def run(self) -> CheckResult:
        await _fake_io()
        win_rate_recent = 0.52
        win_rate_long = 0.58
        if win_rate_recent < win_rate_long - 0.1:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="Win rate declined materially",
                metrics={"win_rate_recent": win_rate_recent, "win_rate_long": win_rate_long},
                remediation_actions=["Reduced trading frequency"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="Win rate within expected band",
            metrics={"win_rate_recent": win_rate_recent, "win_rate_long": win_rate_long},
        )


class NewDrawdownHighCheck(Check):
    @property
    def name(self) -> str:
        return "new_drawdown_high"

    @property
    def severity(self) -> Severity:
        return Severity.HIGH

    async def run(self) -> CheckResult:
        await _fake_io()
        new_high = False
        if new_high:
            return CheckResult(
                healthy=False,
                severity=self.severity,
                message="New all-time drawdown high",
                remediation_actions=["Escalated to risk management"],
            )
        return CheckResult(
            healthy=True,
            severity=self.severity,
            message="No new drawdown highs",
        )


# ---------------------------------------------------------------------------
# Watchdog core
# ---------------------------------------------------------------------------

class Watchdog:
    def __init__(self, config: WatchdogConfig):
        self.config = config
        self.checks: List[Check] = []
        self.running: bool = False
        self._notifier = Notifier()
        self._metrics = MetricsSink()
        self._last_alert: Dict[str, datetime] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._heartbeat_key = "watchdog_heartbeat"

        if not self.config.escalation_levels:
            self.config.escalation_levels = [
                EscalationLevel(
                    name="log",
                    min_severity=Severity.LOW,
                    cooldown=timedelta(seconds=0),
                    channels=["log"],
                ),
                EscalationLevel(
                    name="chat",
                    min_severity=Severity.MEDIUM,
                    cooldown=timedelta(minutes=3),
                    channels=["discord", "telegram"],
                ),
                EscalationLevel(
                    name="email",
                    min_severity=Severity.HIGH,
                    cooldown=timedelta(hours=1),
                    channels=["email"],
                ),
                EscalationLevel(
                    name="sms",
                    min_severity=Severity.CRITICAL,
                    cooldown=timedelta(hours=6),
                    channels=["sms"],
                ),
                EscalationLevel(
                    name="pagerduty",
                    min_severity=Severity.CRITICAL,
                    cooldown=timedelta(minutes=0),
                    channels=["pagerduty"],
                ),
            ]

    def add_check(self, check: Check) -> None:
        self.checks.append(check)

    async def _heartbeat(self) -> None:
        if not self.config.redis_heartbeat:
            return
        payload = {
            "event": "watchdog_heartbeat",
            "key": self._heartbeat_key,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        print(json.dumps(payload))

    async def _run_check_with_timeout(self, check: Check, timeout: float = 1.0) -> CheckResult:
        try:
            return await asyncio.wait_for(check.run(), timeout=timeout)
        except asyncio.TimeoutError:
            return CheckResult(
                healthy=False,
                severity=Severity.CRITICAL,
                message=f"Check '{check.name}' timed out",
                remediation_actions=[],
            )
        except Exception as exc:  # pragma: no cover - defensive
            return CheckResult(
                healthy=False,
                severity=Severity.CRITICAL,
                message=f"Check '{check.name}' crashed: {exc}",
                remediation_actions=[],
            )

    async def process_results(self, results: List[CheckResult]) -> None:
        for check, result in zip(self.checks, results):
            labels = {"check": check.name, "status": "healthy" if result.healthy else "unhealthy"}
            self._metrics.inc("watchdog_checks_total", labels=labels)

            if not result.healthy:
                self._metrics.inc(
                    "watchdog_alerts_total",
                    labels={"check": check.name, "severity": result.severity.value},
                )
                self._consecutive_failures[check.name] = self._consecutive_failures.get(check.name, 0) + 1
                await self._handle_failure(check, result)
            else:
                self._consecutive_failures[check.name] = 0

    async def _handle_failure(self, check: Check, result: CheckResult) -> None:
        for action in result.remediation_actions:
            self._metrics.inc(
                "watchdog_remediations_total",
                labels={"check": check.name, "action": action},
            )

        if self.config.auto_remediate and check.auto_remediate:
            # real remediation would occur inside check.run(); here we just record.
            pass

        for level in self.config.escalation_levels:
            if result.severity.value >= level.min_severity.value:
                await self._notifier.send(
                    level=level,
                    severity=result.severity,
                    title=f"Watchdog: {check.name}",
                    message=result.message,
                )

    async def monitoring_loop(self) -> None:
        self.running = True
        while self.running:
            started = time.perf_counter()
            if not self.checks:
                await asyncio.sleep(self.config.check_interval)
                continue

            tasks: List[Awaitable[CheckResult]] = [
                self._run_check_with_timeout(check) for check in self.checks
            ]
            results = await asyncio.gather(*tasks)
            await self.process_results(results)
            await self._heartbeat()

            elapsed = time.perf_counter() - started
            delay = max(0.0, self.config.check_interval - elapsed)
            await asyncio.sleep(delay)

    async def start(self) -> None:
        await self.monitoring_loop()

    async def stop(self) -> None:
        self.running = False


# ---------------------------------------------------------------------------
# Watchdog factory & global instance
# ---------------------------------------------------------------------------

def create_default_watchdog(interval: float = 30.0) -> Watchdog:
    cfg = WatchdogConfig(check_interval=interval)
    wd = Watchdog(cfg)
    # Register built-in checks
    wd.add_check(PositionLimitCheck())
    wd.add_check(MaxDrawdownCheck())
    wd.add_check(ModelDriftCheck())
    wd.add_check(ApiRateLimitCheck())
    wd.add_check(BrokerConnectivityCheck())
    wd.add_check(DatabaseDeadlockCheck())
    wd.add_check(HighLatencyCheck())
    wd.add_check(CacheMissRateCheck())
    wd.add_check(SignalConfidenceCheck())
    wd.add_check(CorrelationSpikeCheck())
    wd.add_check(NewsSentimentCheck())
    wd.add_check(ResourceExhaustionCheck())
    wd.add_check(RollingSharpeCheck())
    wd.add_check(WinRateDeclineCheck())
    wd.add_check(NewDrawdownHighCheck())
    return wd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _cmd_start(args: argparse.Namespace) -> int:
    wd = create_default_watchdog(interval=args.interval)
    await wd.start()
    return 0


async def _cmd_status(args: argparse.Namespace) -> int:
    # In a real implementation, this would query shared state (e.g. Redis).
    status = {
        "running": True,
        "checks": [
            "position_limits",
            "max_drawdown",
            "model_drift",
            "api_rate_limits",
            "broker_connectivity",
        ],
    }
    print(json.dumps(status, indent=2))
    return 0


async def _cmd_test(args: argparse.Namespace) -> int:
    wd = create_default_watchdog()
    check = next((c for c in wd.checks if c.name == args.check), None)
    if not check:
        print(f"Unknown check: {args.check}", file=sys.stderr)
        return 1
    result = await check.run()
    print(json.dumps({
        "check": check.name,
        "healthy": result.healthy,
        "severity": result.severity.value,
        "message": result.message,
        "metrics": result.metrics,
        "remediation_actions": result.remediation_actions,
    }, indent=2))
    return 0


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ops.watchdog",
        description="ALPHA-PRIME Watchdog Daemon",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    start_p = subparsers.add_parser("start", help="Start watchdog daemon")
    start_p.add_argument("--interval", type=float, default=30.0, help="Check interval in seconds")

    subparsers.add_parser("status", help="Show watchdog status")

    test_p = subparsers.add_parser("test", help="Run a single check once")
    test_p.add_argument("check", help="Check name (e.g. position_limits)")

    args = parser.parse_args(argv)

    async def runner() -> int:
        if args.cmd == "start":
            return await _cmd_start(args)
        if args.cmd == "status":
            return await _cmd_status(args)
        if args.cmd == "test":
            return await _cmd_test(args)
        parser.print_help()
        return 1

    return asyncio.run(runner())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(_cli())


__all__ = [
    "Severity",
    "EscalationLevel",
    "CheckResult",
    "Check",
    "WatchdogConfig",
    "Watchdog",
    "create_default_watchdog",
]
