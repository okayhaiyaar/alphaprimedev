"""
ALPHA-PRIME v2.0 - Executive Overview Dashboard (Python Facade)
================================================================

Python-side contract for the ALPHA-PRIME executive "Home" dashboard.

This module does NOT implement the React 18 + TypeScript page. Instead,
it provides:

- Stable identifiers and metadata for the 12 KPI cards rendered on the
  executive overview page.
- Type-like structures for the main dashboard metrics payload returned
  by `/api/v2/dashboard/metrics`.
- Realtime/WebSocket configuration for the homepage (PnL, signals, health).
- Layout hints (hero metrics, rows/sections) that the frontend can mirror.

The actual React page should live in a TS/TSX file such as:

  dashboard/pages/1_🏠_Home.tsx

and mirror the types and IDs defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Page props (Python view of HomePageProps)
# ---------------------------------------------------------------------------

class HomePageProps(TypedDict, total=False):
    userRole: str
    realtime: bool
    timezone: str  # e.g. "America/New_York"


# ---------------------------------------------------------------------------
# KPI identifiers and basic types
# ---------------------------------------------------------------------------

KpiId = Literal[
    "pnl_overview",
    "portfolio_health",
    "active_signals",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "drift_score",
    "exposure",
    "var_cvar",
    "beta_vol",
    "top_winner_loser",
    "system_health",
]

MetricTrend = Literal["up", "down", "flat"]
MetricColor = Literal["default", "green", "yellow", "red", "gradient"]


class TrendInfo(TypedDict, total=False):
    delta: float
    direction: MetricTrend


class SparklineData(TypedDict):
    values: List[float]
    min: float
    max: float
    current: float


# ---------------------------------------------------------------------------
# KPI payload shapes (what /api/v2/dashboard/metrics returns)
# ---------------------------------------------------------------------------

class PnlOverviewKpi(TypedDict):
    today: float
    today_pct: float
    mtd: float
    ytd: float
    sparkline: SparklineData
    trend: TrendInfo


class PortfolioHealthKpi(TypedDict):
    score: float          # 0-100
    grade: str            # e.g. "A"
    sharpe: float
    max_drawdown: float
    win_rate: float


class ActiveSignalsKpi(TypedDict):
    pending: int
    executed_today: int
    rejected_today: int
    avg_confidence: float
    avg_rr: float


class SharpeRatioKpi(TypedDict):
    current: float
    rolling_90d: SparklineData
    alpha_vs_bench: float
    bull_sharpe: float
    bear_sharpe: float


class MaxDrawdownKpi(TypedDict):
    current: float
    peak: float
    recovery_days: int
    time_underwater_pct: float


class WinRateKpi(TypedDict):
    win_rate: float
    profit_factor: float
    expectancy_pct: float
    recent_24h: float


class DriftScoreKpi(TypedDict):
    composite_pct: float
    data_psi_pct: float
    error_ratio: float
    last_check_at: datetime
    retrain_overdue: bool


class ExposureKpi(TypedDict):
    long_pct: float
    short_pct: float
    net_pct: float
    top_holding_symbol: str
    top_holding_pct: float
    concentration_warning: bool


class VaRCVaRKpi(TypedDict):
    var_95_pct: float
    cvar_95_pct: float
    stress_2020_pct: float
    stress_limit_pct: float


class BetaVolKpi(TypedDict):
    beta_vs_spy: float
    volatility_pct: float
    skew: float
    tail_ratio: float


class TopWinnerLoserKpi(TypedDict):
    winner_symbol: str
    winner_pct: float
    winner_pnl: float
    loser_symbol: str
    loser_pct: float
    loser_pnl: float


class SystemHealthKpi(TypedDict):
    uptime_pct: float
    latency_p95_ms: float
    alerts_total: int
    alerts_warning: int
    cache_hit_pct: float
    last_heartbeat: datetime


class DashboardMetrics(TypedDict):
    """Aggregate payload returned by GET /api/v2/dashboard/metrics."""
    pnl_overview: PnlOverviewKpi
    portfolio_health: PortfolioHealthKpi
    active_signals: ActiveSignalsKpi
    sharpe_ratio: SharpeRatioKpi
    max_drawdown: MaxDrawdownKpi
    win_rate: WinRateKpi
    drift_score: DriftScoreKpi
    exposure: ExposureKpi
    var_cvar: VaRCVaRKpi
    beta_vol: BetaVolKpi
    top_winner_loser: TopWinnerLoserKpi
    system_health: SystemHealthKpi
    generated_at: datetime


# ---------------------------------------------------------------------------
# Visual/layout metadata for each KPI (to mirror grid layout)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KpiLayoutMeta:
    """Layout hint for a KPI card in the 4xN grid."""

    id: KpiId
    title: str
    icon: str           # Lucide icon name in the frontend
    color: MetricColor
    row: int            # 1-based row index in desktop grid
    col: int            # 1-based column index in desktop grid


KPI_LAYOUT: List[KpiLayoutMeta] = [
    # Row 1: PnL Today | PnL MTD | PnL YTD | Active Signals
    KpiLayoutMeta(
        id="pnl_overview",
        title="PnL Overview",
        icon="DollarSign",
        color="gradient",
        row=1,
        col=1,
    ),
    KpiLayoutMeta(
        id="portfolio_health",
        title="Portfolio Health",
        icon="HeartPulse",
        color="green",
        row=1,
        col=2,
    ),
    KpiLayoutMeta(
        id="active_signals",
        title="Active Signals",
        icon="Activity",
        color="yellow",
        row=1,
        col=4,
    ),
    # Row 2: Sharpe (All) | Max DD | Win Rate | Drift Score
    KpiLayoutMeta(
        id="sharpe_ratio",
        title="Sharpe Ratio",
        icon="TrendingUp",
        color="green",
        row=2,
        col=1,
    ),
    KpiLayoutMeta(
        id="max_drawdown",
        title="Max Drawdown",
        icon="ArrowDown",
        color="red",
        row=2,
        col=2,
    ),
    KpiLayoutMeta(
        id="win_rate",
        title="Win Rate",
        icon="Percent",
        color="green",
        row=2,
        col=3,
    ),
    KpiLayoutMeta(
        id="drift_score",
        title="Drift Score",
        icon="Radar",
        color="yellow",
        row=2,
        col=4,
    ),
    # Row 3: Long/Short | VaR 95% | Beta | System Uptime
    KpiLayoutMeta(
        id="exposure",
        title="Exposure & Concentration",
        icon="PieChart",
        color="default",
        row=3,
        col=1,
    ),
    KpiLayoutMeta(
        id="var_cvar",
        title="VaR & Stress Tests",
        icon="AlertTriangle",
        color="red",
        row=3,
        col=2,
    ),
    KpiLayoutMeta(
        id="beta_vol",
        title="Beta & Volatility",
        icon="BarChart2",
        color="default",
        row=3,
        col=3,
    ),
    KpiLayoutMeta(
        id="system_health",
        title="System Health",
        icon="Server",
        color="default",
        row=3,
        col=4,
    ),
    # Row 4: Top Winner | Top Loser | Net Exposure | Health Status
    KpiLayoutMeta(
        id="top_winner_loser",
        title="Top Winner / Loser",
        icon="Trophy",
        color="gradient",
        row=4,
        col=1,
    ),
]


# ---------------------------------------------------------------------------
# Realtime & WebSocket configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealtimeConfig:
    websocket_endpoint: str
    pnl_channel: str
    signals_channel: str
    health_channel: str
    stale_threshold_seconds: int
    smooth_transition_seconds: float


REALTIME_CONFIG = RealtimeConfig(
    websocket_endpoint="/ws/dashboard",
    pnl_channel="pnl",
    signals_channel="signals",
    health_channel="system_health",
    stale_threshold_seconds=30,
    smooth_transition_seconds=0.3,
)


# ---------------------------------------------------------------------------
# Hook configuration (for TS hooks like useDashboardMetrics)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardMetricsQueryConfig:
    query_key: str
    stale_time_ms: int
    refetch_interval_ms_default: int


DASHBOARD_METRICS_QUERY_CONFIG = DashboardMetricsQueryConfig(
    query_key="dashboard.metrics",
    stale_time_ms=30_000,
    refetch_interval_ms_default=5_000,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_kpis() -> List[KpiId]:
    """Return all KPI identifiers used on the Home page."""
    return [meta.id for meta in KPI_LAYOUT]


def get_kpi_layout(kpi_id: str) -> KpiLayoutMeta | None:
    """Return layout metadata for a KPI, if defined."""
    for meta in KPI_LAYOUT:
        if meta.id == kpi_id:
            return meta
    return None


__all__: List[str] = [
    "HomePageProps",
    "KpiId",
    "MetricTrend",
    "MetricColor",
    "TrendInfo",
    "SparklineData",
    "PnlOverviewKpi",
    "PortfolioHealthKpi",
    "ActiveSignalsKpi",
    "SharpeRatioKpi",
    "MaxDrawdownKpi",
    "WinRateKpi",
    "DriftScoreKpi",
    "ExposureKpi",
    "VaRCVaRKpi",
    "BetaVolKpi",
    "TopWinnerLoserKpi",
    "SystemHealthKpi",
    "DashboardMetrics",
    "KpiLayoutMeta",
    "KPI_LAYOUT",
    "RealtimeConfig",
    "REALTIME_CONFIG",
    "DashboardMetricsQueryConfig",
    "DASHBOARD_METRICS_QUERY_CONFIG",
    "list_kpis",
    "get_kpi_layout",
]
