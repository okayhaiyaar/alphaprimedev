"""
ALPHA-PRIME v2.0 - Strategy Performance Analytics (Python Facade)
=================================================================

Python-side contract for the ALPHA-PRIME Performance page.

This module does NOT implement the React 18 + TypeScript UI. Instead it:

- Defines page props and filter types used by the frontend.
- Describes all core data structures (equity curves, regimes, scatter,
  stats table, detail panel).
- Encodes layout sections (sidebar filters, main charts, detail panel).
- Provides query + WebSocket configuration for the performance workspace.

The React implementation should live in a TS/TSX file, e.g.:

  dashboard/pages/2_📊_Performance.tsx

and mirror the types and IDs defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Page props & filters (Python view of TS types)
# ---------------------------------------------------------------------------

TimeRangeId = Literal["1M", "3M", "6M", "YTD", "1Y", "All"]

BenchmarkId = Literal["SPY", "QQQ", "None"]


class PerformanceFilters(TypedDict, total=False):
    timeRange: TimeRangeId
    strategies: List[str]
    benchmark: BenchmarkId
    regimes: List[str]
    metrics: List[str]
    search: str


class PerformancePageProps(TypedDict, total=False):
    initialFilters: PerformanceFilters
    realtime: bool
    userRole: str


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class EquityPoint(TypedDict):
    timestamp: datetime
    equity: float
    drawdown_pct: float


class EquitySeries(TypedDict):
    strategy_id: str
    label: str
    is_live: bool           # live vs backtest
    points: List[EquityPoint]


class RegimeCell(TypedDict):
    strategy_id: str
    regime: str             # "Bull", "Bear", "Sideways", "Volatile", "Crisis"
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int


class ScatterPoint(TypedDict):
    strategy_id: str
    label: str
    volatility_pct: float
    annual_return_pct: float
    sharpe: float
    regime_consistency: float   # 0-1
    alpha: float
    beta: float
    information_ratio: float


class StatsRow(TypedDict):
    strategy: str
    is_sharpe: float
    oos_sharpe: float
    capacity: float
    p_value: float
    alpha: float
    beta: float
    regime_min: float


class DetailMetrics(TypedDict):
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int


class DetailTradeSummary(TypedDict):
    total_trades: int
    avg_hold_days: float
    winners: int
    losers: int
    expectancy_pct: float


class DetailStatTests(TypedDict):
    t_test_p_value: float
    monte_carlo_p_value: float
    bootstrap_p_value: float | None


class StrategyDetail(TypedDict):
    strategy_id: str
    label: str
    metrics: DetailMetrics
    equity: EquitySeries
    drawdown_points: List[EquityPoint]
    trade_summary: DetailTradeSummary
    stat_tests: DetailStatTests


class PerformanceData(TypedDict):
    """
    Aggregate payload returned by GET /api/v2/performance/backtests?filters=...
    """
    filters: PerformanceFilters
    equity_series: List[EquitySeries]
    regimes: List[RegimeCell]
    scatter_points: List[ScatterPoint]
    stats_rows: List[StatsRow]
    selected_strategy_id: str | None
    strategy_detail: StrategyDetail | None
    generated_at: datetime


# ---------------------------------------------------------------------------
# Layout sections (sidebar, main content, detail panel)
# ---------------------------------------------------------------------------

SectionId = Literal[
    "sidebar_filters",
    "equity_curves",
    "regime_matrix",
    "performance_scatter",
    "stats_table",
    "detail_panel",
    "metrics_summary",
]


@dataclass(frozen=True)
class SectionMeta:
    id: SectionId
    title: str
    description: str
    aria_role: str       # e.g. "complementary", "main"
    column: int          # 1=sidebar, 2=main, 3=detail
    order: int           # ordering within column


SECTIONS: List[SectionMeta] = [
    SectionMeta(
        id="sidebar_filters",
        title="Filters",
        description="Time range, strategies, benchmarks, regimes, metrics.",
        aria_role="complementary",
        column=1,
        order=1,
    ),
    SectionMeta(
        id="equity_curves",
        title="Equity Curves",
        description="Live vs backtest equity curves with benchmark overlays.",
        aria_role="main",
        column=2,
        order=1,
    ),
    SectionMeta(
        id="regime_matrix",
        title="Regime Performance Matrix",
        description="Sharpe by regime for top strategies (heatmap).",
        aria_role="main",
        column=2,
        order=2,
    ),
    SectionMeta(
        id="performance_scatter",
        title="Performance Scatter",
        description="Volatility vs annual return with Sharpe-sized bubbles.",
        aria_role="main",
        column=2,
        order=3,
    ),
    SectionMeta(
        id="stats_table",
        title="Statistical Table",
        description="In-sample vs out-of-sample stats and significance.",
        aria_role="region",
        column=2,
        order=4,
    ),
    SectionMeta(
        id="detail_panel",
        title="Strategy Detail",
        description="Deep dive into selected strategy performance.",
        aria_role="complementary",
        column=3,
        order=1,
    ),
    SectionMeta(
        id="metrics_summary",
        title="Metrics Summary",
        description="Sharpe/Sortino/DD/win-rate KPI grid.",
        aria_role="region",
        column=2,
        order=0,
    ),
]


# ---------------------------------------------------------------------------
# Query & WebSocket configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerformanceQueryConfig:
    query_key_prefix: str
    stale_time_ms: int
    keep_previous_data: bool


PERFORMANCE_QUERY_CONFIG = PerformanceQueryConfig(
    query_key_prefix="performance",
    stale_time_ms=60_000,
    keep_previous_data=True,
)


@dataclass(frozen=True)
class PerformanceRealtimeConfig:
    websocket_endpoint: str
    channel_prefix: str
    rolling_sharpe_interval_sec: int


PERFORMANCE_REALTIME_CONFIG = PerformanceRealtimeConfig(
    websocket_endpoint="/ws/performance/live",
    channel_prefix="performance",
    rolling_sharpe_interval_sec=60,
)


# ---------------------------------------------------------------------------
# Table metadata (for TanStack Table on the frontend)
# ---------------------------------------------------------------------------

STATS_TABLE_COLUMNS: List[str] = [
    "strategy",
    "is_sharpe",
    "oos_sharpe",
    "capacity",
    "p_value",
    "alpha",
    "beta",
    "regime_min",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_sections() -> List[SectionId]:
    """Return all section identifiers for the Performance page."""
    return [section.id for section in SECTIONS]


def get_section_meta(section_id: str) -> SectionMeta | None:
    """Return metadata for a given section, if defined."""
    for section in SECTIONS:
        if section.id == section_id:
            return section
    return None


def is_significant(p_value: float, threshold: float = 0.05) -> bool:
    """Helper for `p < 0.05` statistical badges."""
    return p_value < threshold


__all__: List[str] = [
    "TimeRangeId",
    "BenchmarkId",
    "PerformanceFilters",
    "PerformancePageProps",
    "EquityPoint",
    "EquitySeries",
    "RegimeCell",
    "ScatterPoint",
    "StatsRow",
    "DetailMetrics",
    "DetailTradeSummary",
    "DetailStatTests",
    "StrategyDetail",
    "PerformanceData",
    "SectionId",
    "SectionMeta",
    "SECTIONS",
    "PerformanceQueryConfig",
    "PERFORMANCE_QUERY_CONFIG",
    "PerformanceRealtimeConfig",
    "PERFORMANCE_REALTIME_CONFIG",
    "STATS_TABLE_COLUMNS",
    "list_sections",
    "get_section_meta",
    "is_significant",
]
