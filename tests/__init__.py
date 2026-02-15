"""
ALPHA-PRIME v2.0 Test Suite Orchestration
=========================================

Central orchestration module for the ALPHA-PRIME v2.0 test suite.

This module:

- Exposes a stable public API for running logical subsets of tests
  (unit, integration, e2e, smoke, full suite).
- Provides shared metadata (markers, coverage targets, suite name).
- Is safe to import from anywhere (no side effects, no test runs on import).

All interaction with pytest is opt-in via the helper functions below.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import time


__version__ = "2.0.0"
TEST_SUITE_NAME = "ALPHA-PRIME v2.0 Test Suite"
MIN_COVERAGE_BACKEND: float = 0.90   # 90%
MIN_COVERAGE_DASHBOARD: float = 0.80 # 80%
MIN_COVERAGE_OPS: float = 0.85       # 85%


# ---------------------------------------------------------------------------
# Standard pytest markers (documentation-only registry)
# ---------------------------------------------------------------------------

PYTEST_MARKERS: Dict[str, str] = {
    "unit": "Fast-running unit tests, no external services.",
    "integration": "Tests hitting DB, cache, API, or filesystem.",
    "e2e": "End-to-end tests exercising the full system.",
    "performance": "Performance and load tests; may be slow.",
    "slow": "Slow tests not run by default in CI.",
    "smoke": "Minimal tests for quick health checks.",
    "requires_db": "Tests requiring a real or test database.",
    "requires_redis": "Tests requiring Redis.",
    "requires_broker": "Tests requiring external broker APIs.",
}


# ---------------------------------------------------------------------------
# Lazy import infrastructure
# ---------------------------------------------------------------------------

_LAZY_MODULES: Dict[str, Callable[[], ModuleType]] = {}


def _lazy_import(module_path: str) -> ModuleType:
    """
    Lazily import a module used only for testing.

    Raises:
        RuntimeError: If the module cannot be imported, with guidance to
        install development/test dependencies.
    """
    import importlib

    try:
        return importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Required test dependency '{module_path}' is not available. "
            "Install the 'dev' or 'test' extras to run this test suite."
        ) from exc


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class TestRunResult:
    """Summary of a single pytest run."""
    success: bool
    exit_code: int
    tests_run: int
    tests_failed: int
    tests_errored: int
    duration_seconds: float
    markers: List[str]
    paths: List[str]
    extra: Dict[str, Any]


# ---------------------------------------------------------------------------
# Internal runner
# ---------------------------------------------------------------------------

def _run_pytest(args: List[str], markers: Optional[Iterable[str]] = None) -> TestRunResult:
    """
    Internal helper to invoke pytest with the given arguments.

    Notes:
        - Uses a lazy import of pytest.
        - Does not introspect individual test counts; these are reported as -1.
    """
    pytest = _lazy_import("pytest")
    start = time.time()
    exit_code = pytest.main(args)
    duration = time.time() - start

    marker_list = list(markers or [])
    paths = [a for a in args if not a.startswith("-")]

    return TestRunResult(
        success=exit_code == 0,
        exit_code=exit_code,
        tests_run=-1,      # Unknown without plugins
        tests_failed=-1,   # Unknown without plugins
        tests_errored=-1,  # Unknown without plugins
        duration_seconds=duration,
        markers=marker_list,
        paths=paths,
        extra={"args": args},
    )


# ---------------------------------------------------------------------------
# Public helpers for running subsets of the suite
# ---------------------------------------------------------------------------

def run_unit_tests(markers: Optional[Iterable[str]] = None) -> TestRunResult:
    """
    Run unit tests only (fast, no external services).

    Args:
        markers: Optional iterable of additional markers to filter by.

    Returns:
        TestRunResult with pytest exit code and basic context.
    """
    base_args = ["tests/unit"]
    marker_expr_parts: List[str] = ["unit"]
    if markers:
        marker_expr_parts.extend(markers)
    marker_expr = " and ".join(marker_expr_parts)
    args = base_args + ["-m", marker_expr]
    return _run_pytest(args, markers=list(marker_expr_parts))


def run_integration_tests(markers: Optional[Iterable[str]] = None) -> TestRunResult:
    """
    Run integration tests (DB/cache/API/filesystem).

    Args:
        markers: Optional iterable of additional markers to filter by.

    Returns:
        TestRunResult with pytest exit code and basic context.
    """
    base_args = ["tests/integration"]
    marker_expr_parts: List[str] = ["integration"]
    if markers:
        marker_expr_parts.extend(markers)
    marker_expr = " and ".join(marker_expr_parts)
    args = base_args + ["-m", marker_expr]
    return _run_pytest(args, markers=list(marker_expr_parts))


def run_e2e_tests(markers: Optional[Iterable[str]] = None) -> TestRunResult:
    """
    Run end-to-end tests (full-system scenarios).

    Args:
        markers: Optional iterable of additional markers to filter by.

    Returns:
        TestRunResult with pytest exit code and basic context.
    """
    base_args = ["tests/e2e"]
    marker_expr_parts: List[str] = ["e2e"]
    if markers:
        marker_expr_parts.extend(markers)
    marker_expr = " and ".join(marker_expr_parts)
    args = base_args + ["-m", marker_expr]
    return _run_pytest(args, markers=list(marker_expr_parts))


def run_smoke_tests() -> TestRunResult:
    """
    Run smoke tests: minimal, fast checks to verify basic health.

    Returns:
        TestRunResult with pytest exit code and basic context.
    """
    args = ["-m", "smoke"]
    return _run_pytest(args, markers=["smoke"])


def run_full_suite(fail_fast: bool = False) -> TestRunResult:
    """
    Run the full test suite (all test directories).

    Args:
        fail_fast: If True, stop after the first failure (pytest -x).

    Returns:
        TestRunResult summarizing the overall run.
    """
    paths = get_default_test_paths()
    args: List[str] = paths.copy()
    if fail_fast:
        args.append("-x")
    return _run_pytest(args, markers=[])


def run_marked(mark_expr: str, paths: Optional[Iterable[str]] = None) -> TestRunResult:
    """
    Run tests matching a pytest -m marker expression.

    Args:
        mark_expr: Marker expression, e.g. 'unit and not slow'.
        paths: Optional iterable of paths to restrict the run to.

    Returns:
        TestRunResult with pytest exit code and basic context.
    """
    paths_list = list(paths) if paths is not None else get_default_test_paths()
    args = paths_list + ["-m", mark_expr]
    # We don't parse the expression back into a list; report as a single string.
    return _run_pytest(args, markers=[mark_expr])


# ---------------------------------------------------------------------------
# Suite metadata & utilities
# ---------------------------------------------------------------------------

def list_standard_markers() -> Dict[str, str]:
    """
    Return a copy of the standard pytest markers and descriptions.

    This is the single source of truth for marker documentation;
    actual registration should occur in pytest.ini or conftest.py.
    """
    return dict(PYTEST_MARKERS)


def get_default_test_paths() -> List[str]:
    """
    Return the default top-level test directories used by this project.

    This is used by run_full_suite and other helpers as the baseline
    set of test roots.
    """
    return ["tests/unit", "tests/integration", "tests/e2e", "tests/performance"]


def print_quickstart() -> None:
    """
    Print a short quickstart guide for running tests locally.

    This function has no side effects beyond printing to stdout and is
    never executed on import.
    """
    guide = f"""
{TEST_SUITE_NAME} Quickstart
--------------------------------------
- Run all tests:        pytest
- Unit tests only:      pytest tests/unit -m "unit"
- Integration tests:    pytest tests/integration -m "integration"
- E2E tests:            pytest tests/e2e -m "e2e"
- Smoke tests (fast):   pytest -m "smoke"
- With coverage:        pytest --cov=alpha_prime --cov-report=term-missing
"""
    print(guide.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "TEST_SUITE_NAME",
    "__version__",
    "MIN_COVERAGE_BACKEND",
    "MIN_COVERAGE_DASHBOARD",
    "MIN_COVERAGE_OPS",
    "TestRunResult",
    "PYTEST_MARKERS",
    "run_unit_tests",
    "run_integration_tests",
    "run_e2e_tests",
    "run_smoke_tests",
    "run_full_suite",
    "run_marked",
    "list_standard_markers",
    "get_default_test_paths",
    "print_quickstart",
]
