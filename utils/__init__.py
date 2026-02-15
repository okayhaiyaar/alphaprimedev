"""
ALPHA-PRIME v2.0 - Central Utilities Hub
========================================
Safe, production-grade utilities for trading systems.

This module exposes a curated set of utilities via:

TIER 1 - Essentials (from utils import *)
    - Logging, config, paths
    - Data I/O helpers
    - Date & market utilities
    - Core type aliases

TIER 2 - Core Trading (from utils import *)  -- limited curated subset
    - Performance/risk metrics
    - Position sizing
    - Signal helpers
    - Portfolio helpers

TIER 3 - Advanced (lazy, explicit only)
    - ML utilities
    - Optimisation utilities
    - Plotting/visualisation utilities

Design Goals:
- Safe star imports: only vetted symbols in __all__.
- Lazy loading for optional heavy modules.
- Developer-friendly discovery & quickstart helpers.
- Type-checker compatible (mypy-friendly).
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config import get_logger, get_settings

# Public versioning
__version__ = "2.0.0"
__utils_version__ = "2026.02"

logger = get_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────

Price = float | np.ndarray
Returns = np.ndarray
Signal = Literal[-1, 0, 1]
DateRange = tuple[datetime, datetime]


# ──────────────────────────────────────────────────────────
# TIER 1 - ESSENTIALS
# ──────────────────────────────────────────────────────────

# --- Paths & config ---------------------------------------------------------


def get_project_root() -> str:
    """Return project root directory as configured in settings or fallback."""
    root = getattr(settings, "project_root", ".")
    return str(root)


def _join_path(base: str, *parts: str) -> str:
    import os

    return os.path.join(base, *parts)


def get_data_dir() -> str:
    """Return path to the data directory."""
    return _join_path(get_project_root(), getattr(settings, "data_dir", "data"))


def get_cache_dir() -> str:
    """Return path to the cache directory."""
    return _join_path(get_project_root(), getattr(settings, "cache_dir", "cache"))


def get_logs_dir() -> str:
    """Return path to the logs directory."""
    return _join_path(get_project_root(), getattr(settings, "logs_dir", "logs"))


def ensure_dir_exists(path: str) -> str:
    """Ensure directory exists; create it if necessary and return path."""
    import os

    os.makedirs(path, exist_ok=True)
    return path


# --- Data I/O ---------------------------------------------------------------


def read_parquet_safe(path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Safely read a parquet file, returning empty DataFrame on failure.

    Args:
        path: Path to parquet file.
        **kwargs: Passed to pandas.read_parquet.

    Returns:
        DataFrame (possibly empty).
    """
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read parquet %s: %s", path, exc)
        return pd.DataFrame()


def write_parquet_safe(df: pd.DataFrame, path: str, **kwargs: Any) -> bool:
    """
    Safely write DataFrame to parquet, returning success flag.

    Args:
        df: DataFrame to write.
        path: Output path.
        **kwargs: Passed to DataFrame.to_parquet.

    Returns:
        True if write succeeded, False otherwise.
    """
    try:
        ensure_dir_exists(importlib.import_module("os").path.dirname(path) or ".")
        df.to_parquet(path, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write parquet %s: %s", path, exc)
        return False


def load_csv_safe(path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Safely load CSV, returning empty DataFrame on failure.

    Args:
        path: CSV path.
        **kwargs: Passed to pandas.read_csv.

    Returns:
        DataFrame (possibly empty).
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read csv %s: %s", path, exc)
        return pd.DataFrame()


def validate_dataframe_schema(
    df: pd.DataFrame, required_columns: Iterable[str]
) -> bool:
    """
    Validate DataFrame has all required columns.

    Args:
        df: DataFrame to check.
        required_columns: Required column names.

    Returns:
        True if schema is valid, False otherwise.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.warning("DataFrame missing columns: %s", missing)
        return False
    return True


# --- Date / trading calendar -------------------------------------------------


def is_trading_day(d: date) -> bool:
    """
    Heuristic trading-day check (Mon–Fri, not strict holiday-aware).

    Integrate with a proper calendar for production if required.
    """
    return d.weekday() < 5


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (inclusive of start, exclusive of end)."""
    if start > end:
        start, end = end, start
    days = 0
    cur = start
    while cur < end:
        if is_trading_day(cur):
            days += 1
        cur += timedelta(days=1)
    return days


def business_days_ahead(start: date, n: int) -> date:
    """Return date N trading days ahead of start."""
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = start
    while remaining > 0:
        cur += timedelta(days=step)
        if is_trading_day(cur):
            remaining -= 1
    return cur


def next_friday_close(start: datetime) -> datetime:
    """Return next Friday at 16:00 local time (simplified)."""
    cur = start
    while cur.weekday() != 4:  # Friday
        cur += timedelta(days=1)
    return cur.replace(hour=16, minute=0, second=0, microsecond=0)


def get_next_rebalance_date(start: datetime, frequency_days: int = 30) -> datetime:
    """Compute next rebalance date by adding frequency_days trading days."""
    target_date = business_days_ahead(start.date(), frequency_days)
    return datetime.combine(target_date, datetime.min.time())


# --- Market utils ------------------------------------------------------------


def is_market_open(now: Optional[datetime] = None) -> bool:
    """
    Very rough market-open check for US equities (9:30–16:00 Mon–Fri).

    Args:
        now: Datetime to check; defaults to current local time.

    Returns:
        True if within assumed regular session.
    """
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return False
    return 9 <= now.hour < 16 or (now.hour == 16 and now.minute == 0)


def get_trading_session(now: Optional[datetime] = None) -> str:
    """
    Return simple trading session label: 'PRE', 'REGULAR', 'POST', 'CLOSED'.
    """
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return "CLOSED"
    if 4 <= now.hour < 9:
        return "PRE"
    if 9 <= now.hour < 16:
        return "REGULAR"
    if 16 <= now.hour < 20:
        return "POST"
    return "CLOSED"


def market_regime(volatility: float) -> str:
    """
    Map volatility to a simple market regime label.

    Args:
        volatility: Annualised volatility as decimal (e.g. 0.2).

    Returns:
        'CALM' | 'NORMAL' | 'VOLATILE' | 'EXTREME'.
    """
    if volatility < 0.15:
        return "CALM"
    if volatility < 0.25:
        return "NORMAL"
    if volatility < 0.4:
        return "VOLATILE"
    return "EXTREME"


# ──────────────────────────────────────────────────────────
# TIER 2 - CORE TRADING UTILITIES (lightweight implementations)
# ──────────────────────────────────────────────────────────


# --- Performance / risk ------------------------------------------------------


def annualized_return(returns: Returns, periods_per_year: int = 252) -> float:
    """Compute annualised return from period returns."""
    if returns.size == 0:
        return 0.0
    r = 1.0 + returns
    total = float(np.prod(r))
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: Returns, periods_per_year: int = 252) -> float:
    """Annualised volatility."""
    if returns.size == 0:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def calculate_sharpe(returns: Returns, risk_free: float = 0.0) -> float:
    """
    Compute Sharpe ratio.

    Args:
        returns: Period returns array.
        risk_free: Risk-free rate per period (not annualised).

    Returns:
        Sharpe ratio.
    """
    if returns.size == 0:
        return 0.0
    excess = returns - risk_free
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1))
    if sigma == 0:
        return 0.0
    return mu / sigma * np.sqrt(252)


def calculate_sortino(returns: Returns, risk_free: float = 0.0) -> float:
    """Compute Sortino ratio."""
    if returns.size == 0:
        return 0.0
    excess = returns - risk_free
    downside = excess[excess < 0]
    if downside.size == 0:
        return np.inf
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0:
        return np.inf
    mu = float(np.mean(excess))
    return mu / downside_std * np.sqrt(252)


def max_drawdown(equity: Price) -> float:
    """Compute maximum drawdown (as negative fraction)."""
    arr = np.array(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    cummax = np.maximum.accumulate(arr)
    dd = arr / cummax - 1.0
    return float(np.min(dd))


def calmar_ratio(returns: Returns, equity: Price) -> float:
    """Compute Calmar ratio (CAGR / |max drawdown|)."""
    cagr = annualized_return(returns)
    mdd = max_drawdown(equity)
    if mdd >= 0:
        return 0.0
    return cagr / abs(mdd)


# --- Risk measures -----------------------------------------------------------


def var_historical(returns: Returns, alpha: float = 0.95) -> float:
    """Historical VaR at given confidence level (positive number)."""
    if returns.size == 0:
        return 0.0
    q = float(np.quantile(returns, 1 - alpha))
    return -q


def var_parametric(returns: Returns, alpha: float = 0.95) -> float:
    """Parametric (Gaussian) VaR."""
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    from math import sqrt

    from math import erf

    z = sqrt(2) * _erfinv(2 * alpha - 1)
    return -(mu + z * sigma)


def _erfinv(x: float) -> float:
    """Approximate inverse error function."""
    a = 0.147
    sign = 1 if x >= 0 else -1
    ln = np.log(1 - x**2)
    term = 2 / (np.pi * a) + ln / 2
    return sign * np.sqrt(np.sqrt(term**2 - ln / a) - term)


def expected_shortfall(returns: Returns, alpha: float = 0.95) -> float:
    """Expected shortfall (CVaR) at given confidence level."""
    if returns.size == 0:
        return 0.0
    thresh = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= thresh]
    if tail.size == 0:
        return 0.0
    return float(-np.mean(tail))


def ulcer_index(equity: Price) -> float:
    """Ulcer index: RMS of percentage drawdowns."""
    arr = np.array(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    cummax = np.maximum.accumulate(arr)
    dd = (arr - cummax) / cummax * 100.0
    return float(np.sqrt(np.mean(dd**2)))


def pain_index(returns: Returns) -> float:
    """Pain index: average drawdown magnitude."""
    arr = np.array(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    return float(-np.mean(dd))


# --- Position sizing ---------------------------------------------------------


def kelly_criterion(edge: float, win_prob: float, payoff_ratio: float) -> float:
    """
    Kelly fraction for simple Bernoulli outcome.

    Args:
        edge: Expected edge per trade (in return units).
        win_prob: Probability of win.
        payoff_ratio: Average win / average loss.

    Returns:
        Fraction of capital to risk per trade.
    """
    if payoff_ratio <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    return float(max(0.0, min(1.0, (edge / payoff_ratio))))


def fixed_fractional(capital: float, risk_fraction: float, stop_distance: float) -> float:
    """
    Fixed fractional position sizing.

    Args:
        capital: Current capital.
        risk_fraction: Fraction of capital to risk.
        stop_distance: Distance to stop in price terms.

    Returns:
        Position size (shares).
    """
    if stop_distance <= 0 or risk_fraction <= 0:
        return 0.0
    risk_amount = capital * risk_fraction
    return float(risk_amount / stop_distance)


def volatility_parity(vols: Sequence[float]) -> np.ndarray:
    """Compute volatility-parity weights (inverse-vol normalised)."""
    v = np.array(vols, dtype=float)
    v[v <= 0] = np.nan
    inv = 1.0 / v
    if np.all(np.isnan(inv)):
        return np.zeros_like(inv)
    inv = np.nan_to_num(inv, nan=0.0)
    w = inv / inv.sum()
    return w


def optimal_f_position(expected_return: float, variance: float) -> float:
    """
    Simplified optimal-f style sizing: f* ~ mu / sigma^2.

    Args:
        expected_return: Expected per-period return.
        variance: Variance of return.

    Returns:
        Fraction of capital to allocate.
    """
    if variance <= 0:
        return 0.0
    return float(max(0.0, min(1.0, expected_return / variance)))


# --- Signals / portfolio -----------------------------------------------------


def signal_to_direction(signal: Signal) -> int:
    """Convert Signal type to trading direction (-1,0,1)."""
    return int(signal)


def debounce_signals(
    signals: np.ndarray, min_gap: int = 1
) -> np.ndarray:
    """
    Debounce signals by enforcing a minimum gap between non-zero events.

    Args:
        signals: Array of signals (-1,0,1).
        min_gap: Minimum bars between distinct non-zero signals.

    Returns:
        Debounced signal array.
    """
    sig = np.array(signals, dtype=int)
    last_idx = -np.inf
    for i, val in enumerate(sig):
        if val != 0:
            if i - last_idx <= min_gap:
                sig[i] = 0
            else:
                last_idx = i
    return sig


def signal_strength(returns_forward: np.ndarray) -> np.ndarray:
    """
    Compute simple signal strength as forward return z-score.

    Args:
        returns_forward: Array of forward returns.

    Returns:
        Z-scored signal strength array.
    """
    r = np.array(returns_forward, dtype=float)
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    if sigma == 0:
        return np.zeros_like(r)
    return (r - mu) / sigma


def smooth_signals(signals: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply simple moving average smoothing to signal series."""
    sig = np.array(signals, dtype=float)
    if window <= 1 or sig.size == 0:
        return sig
    kernel = np.ones(window) / window
    sm = np.convolve(sig, kernel, mode="same")
    return sm


def portfolio_weights(returns_cov: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
    """
    Compute naive mean-variance portfolio weights given covariance matrix.

    Args:
        returns_cov: Covariance matrix.
        risk_aversion: Risk aversion coefficient.

    Returns:
        Weights summing to 1.
    """
    n = returns_cov.shape[0]
    if n == 0:
        return np.array([])
    cov = np.array(returns_cov, dtype=float)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    ones = np.ones((n, 1))
    w = inv_cov @ ones
    w = w / (ones.T @ w)
    return w.ravel()


def portfolio_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Compute marginal risk contributions for each asset.

    Args:
        weights: Portfolio weights.
        cov: Covariance matrix.

    Returns:
        Risk contribution per asset.
    """
    w = np.array(weights, dtype=float)
    c = np.array(cov, dtype=float)
    total_var = float(w.T @ c @ w)
    if total_var <= 0:
        return np.zeros_like(w)
    mrc = c @ w
    rc = w * mrc / total_var
    return rc


# ──────────────────────────────────────────────────────────
# TIER 3 - ADVANCED: LAZY IMPORTS
# ──────────────────────────────────────────────────────────


class LazyImports:
    """
    Lazy loader for advanced utility modules.

    Access via:
        utils.lazy.plotting.plot_equity_curve(...)
        utils.lazy.ml.robust_scaler(...)
        utils.lazy.optimization.sharpe_objective(...)
    """

    plotting: Any = None
    ml: Any = None
    optimization: Any = None

    def __getattr__(self, name: str) -> Any:
        if name in {"plotting", "ml", "optimization"}:
            try:
                mod = importlib.import_module(f"utils.{name}")
            except ImportError as exc:  # noqa: BLE001
                warnings.warn(
                    f"Optional utils.{name} module not available: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                mod = None
            setattr(self, name, mod)
            return mod
        raise AttributeError(f"'LazyImports' object has no attribute '{name}'")


lazy = LazyImports()


# ──────────────────────────────────────────────────────────
# UTILITY REGISTRY & DISCOVERY
# ──────────────────────────────────────────────────────────

UTILITY_GROUPS: Dict[str, List[str]] = {
    "essentials": [
        "Price",
        "Returns",
        "Signal",
        "DateRange",
        "read_parquet_safe",
        "write_parquet_safe",
        "load_csv_safe",
        "validate_dataframe_schema",
        "get_project_root",
        "get_data_dir",
        "get_cache_dir",
        "get_logs_dir",
        "ensure_dir_exists",
        "trading_days_between",
        "business_days_ahead",
        "next_friday_close",
        "is_trading_day",
        "get_next_rebalance_date",
        "is_market_open",
        "get_trading_session",
        "market_regime",
    ],
    "performance": [
        "annualized_return",
        "annualized_volatility",
        "calculate_sharpe",
        "calculate_sortino",
        "max_drawdown",
        "calmar_ratio",
    ],
    "risk": [
        "var_historical",
        "var_parametric",
        "expected_shortfall",
        "ulcer_index",
        "pain_index",
    ],
    "sizing": [
        "kelly_criterion",
        "fixed_fractional",
        "volatility_parity",
        "optimal_f_position",
    ],
    "signals": [
        "signal_to_direction",
        "debounce_signals",
        "signal_strength",
        "smooth_signals",
    ],
    "portfolio": [
        "portfolio_weights",
        "portfolio_risk_contribution",
    ],
    "advanced": [
        "lazy.plotting.plot_equity_curve",
        "lazy.plotting.plot_feature_heatmap",
        "lazy.ml.robust_scaler",
        "lazy.ml.ts_freshness_filter",
        "lazy.optimization.sharpe_objective",
        "lazy.optimization.hyperopt_wrapper",
    ],
}


# ──────────────────────────────────────────────────────────
# VERSION & COMPATIBILITY CHECKS
# ──────────────────────────────────────────────────────────


def check_compatibility() -> None:
    """
    Warn if utilities are used with incompatible ALPHA-PRIME core.

    This is a light-touch check; core should define alpha_prime_version
    or similar for stricter checks.
    """
    try:
        import alpha_prime  # type: ignore[import-not-found]

        core_version = getattr(alpha_prime, "__version__", None)
        if core_version is None:
            warnings.warn(
                "ALPHA-PRIME core detected but version missing; "
                "compatibility not guaranteed.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Additional semantic version checks could be implemented here.
    except ImportError:
        warnings.warn(
            "ALPHA-PRIME core not detected; utilities may be used standalone "
            "but some integrations could be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )


# ──────────────────────────────────────────────────────────
# DEVELOPER EXPERIENCE HELPERS
# ──────────────────────────────────────────────────────────


def quickstart_guide() -> None:
    """Prints most common utilities grouped by use case."""
    msg = """
🚀 ALPHA-PRIME UTILS QUICKSTART

Essentials:
  - IO:       read_parquet_safe(), load_csv_safe(), validate_dataframe_schema()
  - Paths:    get_project_root(), get_data_dir(), get_cache_dir(), get_logs_dir()
  - Dates:    trading_days_between(), get_next_rebalance_date()
  - Market:   is_market_open(), get_trading_session(), market_regime()

Performance & Risk:
  - Metrics:  calculate_sharpe(), max_drawdown(), annualized_volatility()
  - Risk:     var_historical(), expected_shortfall(), ulcer_index()

Position Sizing & Signals:
  - Sizing:   kelly_criterion(), fixed_fractional(), volatility_parity()
  - Signals:  signal_strength(), debounce_signals(), smooth_signals()

Advanced (lazy-loaded):
  - Plotting: utils.lazy.plotting.plot_equity_curve(equity)
  - ML:       utils.lazy.ml.robust_scaler(X)
  - Opt:      utils.lazy.optimization.sharpe_objective(params)
"""
    print(msg)


def list_all_utils(group: Optional[str] = None) -> List[str]:
    """
    List available utilities by category.

    Args:
        group: Optional group key in UTILITY_GROUPS; if None, returns all.

    Returns:
        List of utility names (strings).
    """
    if group is None:
        all_names: List[str] = []
        for names in UTILITY_GROUPS.values():
            all_names.extend(names)
        # Strip 'lazy.' prefix for presentation
        all_names = [n.replace("lazy.", "") for n in all_names]
        return sorted(set(all_names))
    return UTILITY_GROUPS.get(group, []).copy()


# ──────────────────────────────────────────────────────────
# SAFE STAR IMPORTS (__all__)
# ──────────────────────────────────────────────────────────

# Only these vetted symbols are exported via `from utils import *`.
__all__ = [
    # Tier 1: essentials (around 20)
    "Price",
    "Returns",
    "Signal",
    "DateRange",
    "read_parquet_safe",
    "write_parquet_safe",
    "load_csv_safe",
    "validate_dataframe_schema",
    "get_project_root",
    "get_data_dir",
    "get_cache_dir",
    "get_logs_dir",
    "ensure_dir_exists",
    "trading_days_between",
    "business_days_ahead",
    "next_friday_close",
    "is_trading_day",
    "get_next_rebalance_date",
    "is_market_open",
    "get_trading_session",
    "market_regime",
    # Tier 2: core trading (curated subset)
    "annualized_return",
    "annualized_volatility",
    "calculate_sharpe",
    "calculate_sortino",
    "max_drawdown",
    "calmar_ratio",
    "var_historical",
    "expected_shortfall",
    "ulcer_index",
    "kelly_criterion",
    "fixed_fractional",
    "volatility_parity",
    "signal_to_direction",
    "debounce_signals",
    "signal_strength",
    "smooth_signals",
    "portfolio_weights",
    "portfolio_risk_contribution",
    # Dev helpers
    "lazy",
    "quickstart_guide",
    "list_all_utils",
    "check_compatibility",
    "__version__",
    "__utils_version__",
]

# Run a lightweight compatibility check on first import (non-fatal)
try:
    check_compatibility()
except Exception:  # noqa: BLE001
    # Do not fail imports due to compatibility warnings
    pass
