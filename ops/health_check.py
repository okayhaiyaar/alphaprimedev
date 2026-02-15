"""
ALPHA-PRIME v2.0 - System Health Monitoring
===========================================

Production-grade health checking system for:

- Docker HEALTHCHECK
- Kubernetes liveness/readiness probes
- Operational monitoring and dashboards

Design goals:
- < 2s execution (fast parallel async checks)
- Structured JSON output (for monitoring systems)
- Granular service health, critical vs non-critical
- Zero external dependencies (stdlib + project utils only)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Types & data classes
# ---------------------------------------------------------------------------

SystemStatus = Literal["healthy", "degraded", "critical", "unknown"]
ServiceStatus = Literal["healthy", "degraded", "unhealthy", "unknown"]


class HealthCheckTimeout(TimeoutError):
    """Raised when the overall health check exceeds the allowed timeout."""


@dataclass(frozen=True)
class ServiceHealth:
    name: str
    status: ServiceStatus
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    critical: bool = False


@dataclass(frozen=True)
class SystemHealth:
    status: SystemStatus
    overall_score: float
    timestamp: datetime
    services: Dict[str, ServiceHealth]
    critical_issues: List[str]
    recommendations: List[str]

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


# ---------------------------------------------------------------------------
# Internal timing helpers
# ---------------------------------------------------------------------------

def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


async def _timed(check_coro, name: str, critical: bool) -> ServiceHealth:
    start = time.perf_counter()
    try:
        details, status = await check_coro
    except Exception as exc:  # pragma: no cover - defensive
        end = time.perf_counter()
        return ServiceHealth(
            name=name,
            status="unhealthy" if critical else "degraded",
            response_time_ms=_ms(start, end),
            details={"error": str(exc)},
            critical=critical,
        )
    end = time.perf_counter()
    return ServiceHealth(
        name=name,
        status=status,
        response_time_ms=_ms(start, end),
        details=details,
        critical=critical,
    )


# ---------------------------------------------------------------------------
# Service check implementations (stubs with realistic shapes)
# NOTE: Replace contents with real project integrations.
# ---------------------------------------------------------------------------

# 1. PostgreSQL (critical)
async def _check_postgres() -> ServiceHealth:
    name = "postgres"
    critical = True
    async def _impl():
        # Replace with real async DB call: SELECT 1
        await asyncio.sleep(0.01)
        return {"query": "SELECT 1", "ok": True}, "healthy"
    return await _timed(_impl(), name, critical)


# 2. Redis (critical)
async def _check_redis() -> ServiceHealth:
    name = "redis"
    critical = True
    async def _impl():
        await asyncio.sleep(0.01)
        return {"ping": True, "dbsize": 0}, "healthy"
    return await _timed(_impl(), name, critical)


# 3. Strategy Registry (critical)
async def _check_strategies() -> ServiceHealth:
    name = "strategy_registry"
    critical = True
    async def _impl():
        await asyncio.sleep(0.01)
        active = 5
        healthy = 5
        status: ServiceStatus = "healthy" if healthy > 0 else "unhealthy"
        return {"active": active, "healthy": healthy}, status
    return await _timed(_impl(), name, critical)


# 4. Broker API (Zerodha) (critical)
async def _check_broker() -> ServiceHealth:
    name = "zerodha_broker"
    critical = True
    async def _impl():
        await asyncio.sleep(0.02)
        quote_ok = True
        status: ServiceStatus = "healthy" if quote_ok else "unhealthy"
        return {"symbol": "NSE:AAPL", "quote_ok": quote_ok}, status
    return await _timed(_impl(), name, critical)


# 5. Core API endpoints (critical)
async def _check_core_api() -> ServiceHealth:
    name = "core_api"
    critical = True
    async def _impl():
        await asyncio.sleep(0.01)
        return {"endpoints": ["/health", "/strategies"], "ok": True}, "healthy"
    return await _timed(_impl(), name, critical)


# 6. Live Trading Engine (critical)
async def _check_trading_engine() -> ServiceHealth:
    name = "trading_engine"
    critical = True
    async def _impl():
        await asyncio.sleep(0.015)
        heartbeat = True
        return {"heartbeat": heartbeat}, "healthy" if heartbeat else "unhealthy"
    return await _timed(_impl(), name, critical)


# 7. Risk Engine (critical)
async def _check_risk_engine() -> ServiceHealth:
    name = "risk_engine"
    critical = True
    async def _impl():
        await asyncio.sleep(0.02)
        var_ok = True
        return {"var_calc": var_ok}, "healthy" if var_ok else "unhealthy"
    return await _timed(_impl(), name, critical)


# 8. Polygon API (non-critical)
async def _check_polygon() -> ServiceHealth:
    name = "polygon_api"
    critical = False
    async def _impl():
        await asyncio.sleep(0.02)
        return {"symbol": "AAPL", "data_ok": True}, "healthy"
    return await _timed(_impl(), name, critical)


# 9. Yahoo Finance (non-critical)
async def _check_yahoo() -> ServiceHealth:
    name = "yahoo_finance"
    critical = False
    async def _impl():
        await asyncio.sleep(0.02)
        return {"symbol": "SPY", "data_ok": True}, "healthy"
    return await _timed(_impl(), name, critical)


# 10. Drift Monitor (non-critical)
async def _check_drift_monitor() -> ServiceHealth:
    name = "drift_monitor"
    critical = False
    async def _impl():
        await asyncio.sleep(0.01)
        last_check = datetime.utcnow() - timedelta(minutes=2)
        fresh = (datetime.utcnow() - last_check) < timedelta(minutes=5)
        status: ServiceStatus = "healthy" if fresh else "degraded"
        return {"last_check": last_check.isoformat()}, status
    return await _timed(_impl(), name, critical)


# 11. Backtest Engine (non-critical)
async def _check_backtest_engine() -> ServiceHealth:
    name = "backtest_engine"
    critical = False
    async def _impl():
        await asyncio.sleep(0.03)
        return {"sample_backtest": "ok"}, "healthy"
    return await _timed(_impl(), name, critical)


# 12. Notification System (non-critical)
async def _check_notifications() -> ServiceHealth:
    name = "notification_system"
    critical = False
    async def _impl():
        await asyncio.sleep(0.01)
        return {"test_alert": "queued"}, "healthy"
    return await _timed(_impl(), name, critical)


# 13. File Storage (non-critical)
async def _check_file_storage() -> ServiceHealth:
    name = "file_storage"
    critical = False
    async def _impl():
        await asyncio.sleep(0.01)
        return {"rw_test": True}, "healthy"
    return await _timed(_impl(), name, critical)


# 14. ML Models (non-critical)
async def _check_ml_models() -> ServiceHealth:
    name = "ml_models"
    critical = False
    async def _impl():
        await asyncio.sleep(0.02)
        return {"xgboost_inference": "ok"}, "healthy"
    return await _timed(_impl(), name, critical)


# 15. External News / Sentiment (non-critical)
async def _check_news() -> ServiceHealth:
    name = "external_news"
    critical = False
    async def _impl():
        await asyncio.sleep(0.02)
        return {"sentiment_ok": True}, "healthy"
    return await _timed(_impl(), name, critical)


# ---------------------------------------------------------------------------
# Aggregate async runner
# ---------------------------------------------------------------------------

async def _check_all_services(critical_only: bool) -> Dict[str, ServiceHealth]:
    tasks: List[asyncio.Task[ServiceHealth]] = []

    # Critical services (always checked)
    tasks.append(asyncio.create_task(_check_postgres()))
    tasks.append(asyncio.create_task(_check_redis()))
    tasks.append(asyncio.create_task(_check_strategies()))
    tasks.append(asyncio.create_task(_check_broker()))
    tasks.append(asyncio.create_task(_check_core_api()))
    tasks.append(asyncio.create_task(_check_trading_engine()))
    tasks.append(asyncio.create_task(_check_risk_engine()))

    # Non-critical services (optional)
    if not critical_only:
        tasks.extend(
            [
                asyncio.create_task(_check_polygon()),
                asyncio.create_task(_check_yahoo()),
                asyncio.create_task(_check_drift_monitor()),
                asyncio.create_task(_check_backtest_engine()),
                asyncio.create_task(_check_notifications()),
                asyncio.create_task(_check_file_storage()),
                asyncio.create_task(_check_ml_models()),
                asyncio.create_task(_check_news()),
            ]
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    services: Dict[str, ServiceHealth] = {}

    for result in results:
        if isinstance(result, ServiceHealth):
            services[result.name] = result
        else:  # Exception caught by gather
            # Generic fallback for an unnamed failure (shouldn't happen with our wrappers)
            services.setdefault(
                "unknown",
                ServiceHealth(
                    name="unknown",
                    status="unhealthy",
                    response_time_ms=0.0,
                    details={"error": str(result)},
                    critical=False,
                ),
            )
    return services


async def health_check_async(timeout: float = 2.0, critical_only: bool = False) -> SystemHealth:
    """
    Async health check with overall timeout protection.

    Raises:
        HealthCheckTimeout: If overall execution exceeds `timeout`.
    """
    started = datetime.utcnow()
    try:
        services = await asyncio.wait_for(
            _check_all_services(critical_only=critical_only),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise HealthCheckTimeout("Health check exceeded timeout") from exc

    # Scoring engine
    total_services = len(services)
    critical_services = [s for s in services.values() if s.critical]
    total_critical = len(critical_services)
    critical_failures = sum(1 for s in critical_services if s.status != "healthy")
    degraded_count = sum(1 for s in services.values() if s.status == "degraded")

    avg_response = (
        sum(s.response_time_ms for s in services.values()) / total_services
        if total_services
        else 0.0
    )

    # Base score
    score = 100.0
    if total_critical:
        score -= 50.0 * (critical_failures / float(total_critical))
    if total_services:
        score -= 25.0 * (degraded_count / float(total_services))
    if avg_response > 500.0:
        score -= 10.0

    score = max(0.0, min(100.0, score))

    # Status thresholds
    if score >= 90.0:
        status: SystemStatus = "healthy"
    elif score >= 70.0:
        status = "degraded"
    else:
        status = "critical"

    critical_issues: List[str] = [
        f"{s.name}: {s.status}"
        for s in critical_services
        if s.status != "healthy"
    ]

    recommendations: List[str] = []
    if critical_issues:
        recommendations.append("Investigate critical services immediately.")
    if avg_response > 500.0:
        recommendations.append("High average response time; check system load.")
    if status == "degraded" and not critical_issues:
        recommendations.append("Monitor degraded services and plan maintenance.")

    return SystemHealth(
        status=status,
        overall_score=score,
        timestamp=started,
        services=services,
        critical_issues=critical_issues,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Synchronous wrapper (public API)
# ---------------------------------------------------------------------------

def health_check(timeout: float = 2.0, critical_only: bool = False) -> SystemHealth:
    """
    Comprehensive ALPHA-PRIME system health check.

    Args:
        timeout: Max seconds for all service checks combined.
        critical_only: If True, skip non-critical services.

    Returns:
        SystemHealth dataclass with service-level details.

    Raises:
        HealthCheckTimeout: If total execution exceeds `timeout`.
    """
    return asyncio.run(health_check_async(timeout=timeout, critical_only=critical_only))


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def system_health_to_dict(h: SystemHealth) -> Dict[str, Any]:
    return {
        "status": h.status,
        "overall_score": h.overall_score,
        "timestamp": h.timestamp.isoformat() + "Z",
        "services": {
            name: {
                "status": svc.status,
                "response_time_ms": svc.response_time_ms,
                "critical": svc.critical,
                "details": svc.details,
            }
            for name, svc in h.services.items()
        },
        "critical_issues": h.critical_issues,
        "recommendations": h.recommendations,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ops.health_check",
        description="ALPHA-PRIME system health check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON (for monitoring systems)",
    )
    parser.add_argument(
        "--critical",
        action="store_true",
        help="Check critical services only (fast probe)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Max seconds for all checks (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include per-service details in human-readable output",
    )

    args = parser.parse_args(argv)

    try:
        result = health_check(timeout=args.timeout, critical_only=args.critical)
    except HealthCheckTimeout as exc:
        if args.json:
            payload = {
                "status": "critical",
                "overall_score": 0.0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(exc),
            }
            print(json.dumps(payload))
        else:
            print(f"Health check timeout: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(system_health_to_dict(result)))
        return 0 if result.is_healthy else 1

    # Human-readable output
    print(f"Status: {result.status} (Score: {result.overall_score:.1f})")
    print(f"Timestamp: {result.timestamp.isoformat()}Z")
    print()

    for svc in sorted(result.services.values(), key=lambda s: (not s.critical, s.name)):
        line = f"- {svc.name}: {svc.status} ({svc.response_time_ms:.1f}ms"
        if args.verbose and svc.details:
            line += f", details={svc.details}"
        line += ")"
        print(line)

    if result.critical_issues:
        print()
        print("Critical issues:")
        for issue in result.critical_issues:
            print(f"  * {issue}")

    if result.recommendations:
        print()
        print("Recommendations:")
        for r in result.recommendations:
            print(f"  - {r}")

    return 0 if result.is_healthy else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(_cli())


__all__ = [
    "SystemHealth",
    "ServiceHealth",
    "HealthCheckTimeout",
    "health_check",
    "health_check_async",
    "system_health_to_dict",
]
