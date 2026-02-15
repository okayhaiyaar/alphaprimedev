"""
ALPHA-PRIME v2.0 Unit Test Suite
=================================

Fast, isolated tests for pure business logic.

Definition:
- Test pure functions, business logic, and calculations.
- No database, Redis, file I/O, or network calls.
- Must complete in <100ms per test.
- Fully deterministic (no random, no datetime.now()).
- Use mocks/stubs for all external dependencies.

This module provides:
- Metadata describing what counts as a unit test.
- Discovery helpers for unit test modules.
- A thin execution wrapper around pytest for unit tests only.
- Utilities for enforcing speed constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import os
import time


__version__ = "2.0.0"

UNIT_TEST_DEFINITION: str = """
Unit tests in ALPHA-PRIME v2.0:
- Test pure functions, business logic, calculations
- No database, Redis, file I/O, or network calls
- Must complete in <100ms per test
- Fully deterministic (no random, no datetime.now())
- Use mocks/stubs for all external dependencies
""".strip()

UNIT_TEST_SCOPE: Dict[str, List[str]] = {
    "utils": [
        "math utilities (Sharpe, Sortino, Calmar)",
        "date/time helpers",
        "string formatting",
        "validation functions",
    ],
    "core": [
        "signal generation logic",
        "position sizing calculations",
        "risk metrics (no DB)",
        "feature engineering (pure functions)",
    ],
    "models": [
        "strategy parameter validation",
        "data transformations",
        "model serialization/deserialization",
    ],
}

MAX_TEST_DURATION_MS: float = 100.0


@dataclass
class UnitTestRunResult:
    """Summary of a unit-test-only run."""
    success: bool
    exit_code: int
    duration_seconds: float
    modules: List[str]


def _lazy_import_pytest():
    """Lazy import of pytest to keep imports side-effect-free."""
    import importlib
    try:
        return importlib.import_module("pytest")
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "pytest is required to run unit tests. "
            "Install development/test dependencies and retry."
        ) from exc


def list_unit_test_modules() -> List[str]:
    """
    Return all test_*.py modules in tests/unit/.

    The returned values are module-like names without the .py extension
    and without directory prefixes (e.g. 'test_utils_math').
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


def _unit_test_args(module: Optional[str], verbose: bool) -> List[str]:
    """
    Build pytest CLI arguments for unit tests.

    - Restricts to tests/unit.
    - Filters to tests marked 'unit' and not 'slow' or 'integration'/'e2e'/'performance'.
    """
    base_path = os.path.dirname(__file__)
    args: List[str] = [base_path]

    if module:
        args.append(os.path.join(base_path, f"{module}.py"))

    # Marker expression: unit tests only, avoid known slow / non-unit markers.
    mark_expr = "unit and not slow and not integration and not e2e and not performance"
    args.extend(["-m", mark_expr])

    if verbose:
        args.append("-vv")
    else:
        args.append("-q")

    return args


def run_unit_tests(
    module: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """
    Run unit tests with strict fast-only settings.

    Args:
        module:
            Specific module name (e.g., 'test_utils_math') relative to
            tests/unit/. If None, runs all unit tests.
        verbose:
            If True, run pytest with -vv, otherwise -q.

    Returns:
        Exit code (0 = success, non-zero = failures or errors).
    """
    pytest = _lazy_import_pytest()
    args = _unit_test_args(module=module, verbose=verbose)
    start = time.time()
    exit_code = pytest.main(args)
    _ = UnitTestRunResult(
        success=exit_code == 0,
        exit_code=exit_code,
        duration_seconds=time.time() - start,
        modules=[module] if module else list_unit_test_modules(),
    )
    return exit_code


def is_fast_test(test_duration_ms: float) -> bool:
    """Check if test meets unit test speed requirements (<100ms)."""
    return test_duration_ms < MAX_TEST_DURATION_MS


__all__ = [
    "UNIT_TEST_DEFINITION",
    "UNIT_TEST_SCOPE",
    "MAX_TEST_DURATION_MS",
    "list_unit_test_modules",
    "run_unit_tests",
    "is_fast_test",
    "__version__",
]
