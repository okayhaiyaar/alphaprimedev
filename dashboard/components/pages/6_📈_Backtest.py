"""
ALPHA-PRIME v2.0 - Backtesting & Validation Dashboard (Python Facade)
====================================================================

Python-side contract for the ALPHA-PRIME Backtest page.

This module does NOT implement the React 18 + TypeScript UI. Instead it:

- Defines page props and core data structures for backtests, validation,
  Monte Carlo, walk-forward, and comparison.
- Encodes the tabbed layout (Results → Validation → Monte Carlo →
  Walk-Forward → Compare).
- Provides configuration for APIs and WebSocket endpoints.

The React implementation should live in a TS/TSX file, e.g.:

  dashboard/pages/6_📈_Backtest.tsx

and mirror the types and IDs defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Page props
# ---------------------------------------------------------------------------

class BacktestPageProps(TypedDict, total=False):
    backtestId: str
    strategyId: str
    compareIds: List[str]


# ---------------------------------------------------------------------------
# Core backtest data types
# ---------------------------------------------------------------------------

class PeriodRange(TypedDict):
    start: datetime
    end: datetime


class EquityPoint(TypedDict):
    timestamp: datetime
    equity: float
    drawdown_pct: float


class Trade(TypedDict, total=False):
    id: str
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    duration_minutes: float


class PerformanceMetrics(TypedDict, total=False):
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    cagr_pct: float
    annual_vol_pct: float
    alpha: float
    beta: float


class MonthlyReturn(TypedDict):
    year: int
    month: int
    return_pct: float


class RegimeBreakdownRow(TypedDict):
    regime: str
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float


class DetailedStats(TypedDict, total=False):
    is_metrics: PerformanceMetrics
    oos_metrics: PerformanceMetrics
    monthly_returns: List[MonthlyReturn]
    regime_breakdown: List[RegimeBreakdownRow]


# ---------------------------------------------------------------------------
# Validation report (tab 2)
# ---------------------------------------------------------------------------

ValidationSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "INFO"]
ValidationStatus = Literal["FAIL", "WARN", "PASS"]


class ValidationCheck(TypedDict, total=False):
    id: str
    name: str
    severity: ValidationSeverity
    status: ValidationStatus
    message: str
    recommendation: str | None


class ValidationReport(TypedDict, total=False):
    score: float            # 0-100
    deployable: bool
    checks: List[ValidationCheck]


# ---------------------------------------------------------------------------
# Monte Carlo (tab 3)
# ---------------------------------------------------------------------------

class MonteCarloDistribution(TypedDict, total=False):
    sharpe_samples: List[float]
    dd_samples: List[float]
    ruin_probability: float
    prob_superior_vs_bench: float
    sharpe_ci_95: List[float]      # [lower, upper]
    dd_ci_95: List[float]          # [lower, upper]


class MonteCarloResult(TypedDict, total=False):
    simulations: int
    distribution: MonteCarloDistribution
    generated_at: datetime


# ---------------------------------------------------------------------------
# Walk-forward (tab 4)
# ---------------------------------------------------------------------------

class WalkForwardWindow(TypedDict, total=False):
    window_id: str
    period: PeriodRange
    params: Dict[str, Any]
    is_metrics: PerformanceMetrics
    oos_metrics: PerformanceMetrics
    capacity_ratio: float
    regime_min_sharpe: float


class WalkForwardResult(TypedDict, total=False):
    windows: List[WalkForwardWindow]
    parameter_stability: Dict[str, float]   # param -> stability score
    deployment_recommendation: str          # e.g. "deploy", "review", "reject"


# ---------------------------------------------------------------------------
# A/B comparison (tab 5)
# ---------------------------------------------------------------------------

class CompareMetricRow(TypedDict, total=False):
    metric: str
    values: Dict[str, float]       # strategy_id -> value
    benchmark_value: float | None


class CompareStatTest(TypedDict, total=False):
    test_name: str                 # e.g. "paired_t_test"
    p_value: float
    significant: bool


class CompareResult(TypedDict, total=False):
    strategy_ids: List[str]
    metric_rows: List[CompareMetricRow]
    stat_tests: List[CompareStatTest]
    trade_level_summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# BacktestResult aggregate (used across tabs)
# ---------------------------------------------------------------------------

class BacktestResult(TypedDict, total=False):
    id: str
    strategy: str
    period: PeriodRange
    metrics: PerformanceMetrics
    equity_curve: List[EquityPoint]
    benchmark_curve: List[EquityPoint]
    trades: List[Trade]
    detailed_stats: DetailedStats
    validation: ValidationReport
    walk_forward: List[WalkForwardResult]
    monte_carlo: MonteCarloResult


# ---------------------------------------------------------------------------
# Tabbed layout model
# ---------------------------------------------------------------------------

TabId = Literal["results", "validation", "montecarlo", "walkforward", "compare"]


@dataclass(frozen=True)
class TabMeta:
    id: TabId
    label: str
    icon: str          # Lucide icon name
    aria_label: str
    order: int


TABS: List[TabMeta] = [
    TabMeta(
        id="results",
        label="Results",
        icon="BarChart3",
        aria_label="Backtest results",
        order=1,
    ),
    TabMeta(
        id="validation",
        label="Validation",
        icon="CheckCircle2",
        aria_label="Validation checks",
        order=2,
    ),
    TabMeta(
        id="montecarlo",
        label="Monte Carlo",
        icon="Dice5",
        aria_label="Monte Carlo simulations",
        order=3,
    ),
    TabMeta(
        id="walkforward",
        label="Walk-Forward",
        icon="FastForward",
        aria_label="Walk-forward analysis",
        order=4,
    ),
    TabMeta(
        id="compare",
        label="Compare",
        icon="Scale",
        aria_label="Strategy comparison",
        order=5,
    ),
]


# ---------------------------------------------------------------------------
# Validation scoring model
# ---------------------------------------------------------------------------

class ValidationScoreComponents(TypedDict, total=False):
    stats: float
    validation: float
    walk_forward: float
    monte_carlo: float
    regime: float


class ValidationGradeBreakdown(TypedDict, total=False):
    total_score: float
    grade: str         # e.g. "PREMIUM", "PRODUCTION READY"
    components: ValidationScoreComponents
    deployable: bool
    message: str


def compute_validation_grade(score: float) -> str:
    """
    Map an overall score to a grade.

    95-100: PREMIUM
    85-95:  PRODUCTION READY
    75-85:  REVIEW REQUIRED
    <75:    REJECTED
    """
    if score >= 95:
        return "PREMIUM"
    if score >= 85:
        return "PRODUCTION READY"
    if score >= 75:
        return "REVIEW REQUIRED"
    return "REJECTED"


# ---------------------------------------------------------------------------
# API & WebSocket configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestApiConfig:
    get_backtest_endpoint: str
    get_montecarlo_endpoint: str
    get_walkforward_endpoint: str
    compare_endpoint: str


BACKTEST_API_CONFIG = BacktestApiConfig(
    get_backtest_endpoint="/api/v2/backtests/{id}",
    get_montecarlo_endpoint="/api/v2/backtests/{id}/montecarlo",
    get_walkforward_endpoint="/api/v2/backtests/{id}/walkforward",
    compare_endpoint="/api/v2/backtests/compare",
)


@dataclass(frozen=True)
class BacktestRealtimeConfig:
    websocket_endpoint_format: str  # e.g. "/ws/backtest/{id}/live"


BACKTEST_REALTIME_CONFIG = BacktestRealtimeConfig(
    websocket_endpoint_format="/ws/backtest/{id}/live",
)


# ---------------------------------------------------------------------------
# Hook configuration facades (for TS hooks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestDataHookConfig:
    stale_time_ms: int
    refetch_interval_ms: int


@dataclass(frozen=True)
class MonteCarloHookConfig:
    debounce_ms: int


@dataclass(frozen=True)
class WalkForwardHookConfig:
    debounce_ms: int


USE_BACKTEST_DATA_CONFIG = BacktestDataHookConfig(
    stale_time_ms=60_000,
    refetch_interval_ms=0,
)

USE_MONTECARLO_CONFIG = MonteCarloHookConfig(
    debounce_ms=500,
)

USE_WALKFORWARD_CONFIG = WalkForwardHookConfig(
    debounce_ms=500,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_tabs() -> List[TabId]:
    """Return all tab identifiers for the Backtest page."""
    return [tab.id for tab in TABS]


def get_tab_meta(tab_id: str) -> TabMeta | None:
    """Return metadata for a given tab, if defined."""
    for tab in TABS:
        if tab.id == tab_id:
            return tab
    return None


__all__: List[str] = [
    "BacktestPageProps",
    "PeriodRange",
    "EquityPoint",
    "Trade",
    "PerformanceMetrics",
    "MonthlyReturn",
    "RegimeBreakdownRow",
    "DetailedStats",
    "ValidationSeverity",
    "ValidationStatus",
    "ValidationCheck",
    "ValidationReport",
    "MonteCarloDistribution",
    "MonteCarloResult",
    "WalkForwardWindow",
    "WalkForwardResult",
    "CompareMetricRow",
    "CompareStatTest",
    "CompareResult",
    "BacktestResult",
    "TabId",
    "TabMeta",
    "TABS",
    "ValidationScoreComponents",
    "ValidationGradeBreakdown",
    "compute_validation_grade",
    "BacktestApiConfig",
    "BACKTEST_API_CONFIG",
    "BacktestRealtimeConfig",
    "BACKTEST_REALTIME_CONFIG",
    "BacktestDataHookConfig",
    "MonteCarloHookConfig",
    "WalkForwardHookConfig",
    "USE_BACKTEST_DATA_CONFIG",
    "USE_MONTECARLO_CONFIG",
    "USE_WALKFORWARD_CONFIG",
    "list_tabs",
    "get_tab_meta",
]
