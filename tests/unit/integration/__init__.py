"""
ALPHA-PRIME v2.0 Integration Test Suite
========================================
Tests requiring external services (PostgreSQL, Redis, broker APIs).

Definition:
- Multi-component testing (DB + cache + broker + API).
- Real external services (PostgreSQL, Redis, broker sandbox).
- Real transactions with rollback and cache behavior.
- Slower execution (<5s/test, <60s suite).
- Isolated test environment (dedicated test DB/Redis).
"""

from __future__ import annotations

import asyncio
import importlib
import os
import socket
from typing import Any, Dict, List, Optional

__version__ = "2.0.0"

INTEGRATION_TEST_DEFINITION: str = """
Integration tests in ALPHA-PRIME v2.0:
- Test interactions between multiple components
- Require external services (PostgreSQL, Redis, broker APIs)
- Test real database transactions (with rollback)
- Test cache behavior, API integrations, message queues
- Slower execution (<5s per test, <60s suite)
- Environment-specific (test DB, test Redis instance)
""".strip()

REQUIRED_SERVICES: Dict[str, Dict[str, Any]] = {
    "postgres": {
        "host": os.getenv("TEST_POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("TEST_POSTGRES_PORT", "5432")),
        "test_db": os.getenv("TEST_POSTGRES_DB", "alpha_prime_test"),
    },
    "redis": {
        "host": os.getenv("TEST_REDIS_HOST", "localhost"),
        "port": int(os.getenv("TEST_REDIS_PORT", "6379")),
        "test_db": int(os.getenv("TEST_REDIS_DB", "15")),
    },
    "broker_api": {
        "sandbox": True,
        "timeout": int(os.getenv("TEST_BROKER_TIMEOUT", "10")),
        "base_url": os.getenv("TEST_BROKER_BASE_URL", "https://sandbox-broker.example.com"),
    },
}

INTEGRATION_TEST_SCOPE: Dict[str, List[str]] = {
    "database": [
        "Transaction commits/rollbacks",
        "Query performance",
        "Concurrent access",
        "Migration testing",
    ],
    "cache": [
        "Redis operations (get/set/delete)",
        "Cache invalidation",
        "Distributed locking",
        "Pub/sub messaging",
    ],
    "api": [
        "FastAPI endpoint testing",
        "Authentication/authorization",
        "Request/response validation",
        "Error handling",
    ],
    "broker": [
        "Order placement/cancellation",
        "Quote fetching",
        "Position updates",
        "WebSocket streams",
    ],
    "workflow": [
        "Strategy execution end-to-end",
        "Backtest → deployment flow",
        "Alert → notification pipeline",
    ],
}


# ---------------------------------------------------------------------------
# Service utilities
# ---------------------------------------------------------------------------

def _check_tcp_service(host: str, port: int, timeout: float) -> bool:
    """Lightweight TCP connectivity check."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def check_services_available(
    services: Optional[List[str]] = None,
    timeout: float = 5.0,
) -> Dict[str, bool]:
    """
    Check if required services are reachable.

    Args:
        services: List of services to check (default: all).
        timeout: Connection timeout per service (seconds).

    Returns:
        Dict mapping service name to availability status.
    """
    selected = services or list(REQUIRED_SERVICES.keys())
    status: Dict[str, bool] = {}

    for name in selected:
        cfg = REQUIRED_SERVICES.get(name, {})
        if name == "postgres":
            host = cfg.get("host", "localhost")
            port = int(cfg.get("port", 5432))
            status[name] = _check_tcp_service(host, port, timeout)
        elif name == "redis":
            host = cfg.get("host", "localhost")
            port = int(cfg.get("port", 6379))
            status[name] = _check_tcp_service(host, port, timeout)
        elif name == "broker_api":
            # For HTTP APIs we only check TCP port on hostname for simplicity.
            base_url: str = cfg.get("base_url", "https://sandbox-broker.example.com")
            # Avoid heavy HTTP call here; optional TCP check if hostname/port obvious.
            status[name] = True
        else:
            status[name] = False

    return status


async def wait_for_services(
    services: Optional[List[str]] = None,
    max_wait: float = 30.0,
    poll_interval: float = 1.0,
) -> bool:
    """
    Wait for services to become available.

    Args:
        services: Services to wait for (default: all).
        max_wait: Maximum total wait time in seconds.
        poll_interval: Poll interval in seconds.

    Returns:
        True if all services available, False if timeout reached.
    """
    remaining = max_wait
    target = services or list(REQUIRED_SERVICES.keys())

    while remaining >= 0:
        status = check_services_available(target)
        if all(status.values()):
            return True
        await asyncio.sleep(poll_interval)
        remaining -= poll_interval

    return False


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def list_integration_test_modules() -> List[str]:
    """
    Return all test_*.py modules in tests/integration/.

    Returns:
        List of module names without directory or extension (e.g. 'test_trading_flow').
    """
    base_dir = os.path.dirname(__file__)
    modules: List[str] = []
    for entry in os.listdir(base_dir):
        if not entry.startswith("test_"):
            continue
        if not entry.endswith(".py"):
            continue
        if entry == "__init__.py":
            continue
        modules.append(entry[:-3])
    modules.sort()
    return modules


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _lazy_import_pytest():
    """Lazy import pytest to keep import-time side effects minimal."""
    try:
        return importlib.import_module("pytest")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pytest is required to run integration tests. "
            "Install development/test dependencies and retry."
        ) from exc


def _build_pytest_args(
    module: Optional[str],
    markers: Optional[List[str]],
) -> List[str]:
    """Build pytest CLI args for integration tests."""
    base_path = os.path.dirname(__file__)
    args: List[str] = [base_path]

    if module:
        args.append(os.path.join(base_path, f"{module}.py"))

    mark_expr_parts: List[str] = ["integration"]
    if markers:
        # Combine additional markers with 'and' semantics.
        mark_expr_parts.extend(markers)
    mark_expr = " and ".join(mark_expr_parts)
    args.extend(["-m", mark_expr])

    # Less verbose by default; integration runs can be noisy.
    args.append("-v")
    return args


def run_integration_tests(
    module: Optional[str] = None,
    markers: Optional[List[str]] = None,
    skip_if_services_unavailable: bool = True,
) -> int:
    """
    Run integration tests with service checks.

    Args:
        module:
            Specific module name (e.g., 'test_trading_flow') relative
            to tests/integration/. If None, runs all integration tests.
        markers:
            Additional pytest markers to filter (e.g., ['requires_db']).
        skip_if_services_unavailable:
            If True, skip tests when required services are down and
            return exit code 0. If False, fail with non-zero exit.

    Returns:
        Exit code (0 = success).
    """
    pytest = _lazy_import_pytest()

    # Service check (synchronous, fast TCP-level).
    status = check_services_available()
    all_ok = all(status.values())

    if not all_ok and skip_if_services_unavailable:
        # Emulate a "skipped" run: print info and return success.
        missing = [name for name, ok in status.items() if not ok]
        print(f"[integration] Required services unavailable, skipping: {', '.join(missing)}")
        return 0

    if not all_ok and not skip_if_services_unavailable:
        missing = [name for name, ok in status.items() if not ok]
        print(f"[integration] Required services unavailable: {', '.join(missing)}")
        return 1

    args = _build_pytest_args(module=module, markers=markers)
    return pytest.main(args)


__all__ = [
    "__version__",
    "INTEGRATION_TEST_DEFINITION",
    "REQUIRED_SERVICES",
    "INTEGRATION_TEST_SCOPE",
    "check_services_available",
    "wait_for_services",
    "list_integration_test_modules",
    "run_integration_tests",
]
