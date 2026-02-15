"""
ALPHA-PRIME v2.0 - Trading Metrics Cards (Python Facade)
========================================================

Python-side model of the React/TypeScript metric cards used in the
ALPHA-PRIME dashboard.

The real UI components (React 18 + TS strict, shadcn/ui, Tailwind,
Recharts, Framer Motion, TanStack Query, Lucide) live in the frontend
codebase, e.g. `dashboard/components/metrics.tsx`.

This module provides:

- Stable metric card names and types.
- Props/shape definitions for each metric card.
- Metadata for trends, colors, sparklines, and realtime behaviour.
- Hook configuration descriptors (`useMetricData`, `useSparkline`,
  `useMetricTrend`) for the TS layer to implement.

It contains only valid Python so it can safely live alongside backend
code without breaking Python tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Mapping, TypedDict


# ---------------------------------------------------------------------------
# Core types (Python view of TS interfaces)
# ---------------------------------------------------------------------------

MetricTrend = Literal["up", "down", "flat"]
MetricColor = Literal["default", "green", "yellow", "red", "gradient"]
SharpeGrade = Literal["F", "D", "C", "B", "A", "A+"]

MetricId = Literal[
    "sharpe_ratio",
    "drawdown",
    "win_rate",
    "drift_score",
    "pnl",
    "exposure",
    "var_cvar",
    "risk_metrics",
    "signal_queue",
    "active_positions",
    "universe_score",
    "system_health",
]


class TrendInfo(TypedDict, total=False):
    delta: float
    direction: MetricTrend


class SparklineData(TypedDict):
    values: List[float]
    min: float
    max: float
    current: float


class MetricCardProps(TypedDict, total=False):
    """
    Backend-side representation of the TS `MetricCardProps`.

    The React implementation should mirror this interface.
    """

    title: str
    value: str | float | int
    trend: TrendInfo
    sparklineData: SparklineData
    color: MetricColor
    realtime: bool
    className: str


# ---------------------------------------------------------------------------
# Metric-specific payloads
# ---------------------------------------------------------------------------


class SharpeRatioPayload(TypedDict):
    value: float
    trend_delta_24h: float
    rolling_90d: List[float]
    bull_sharpe: float
    bear_sharpe: float
    grade: SharpeGrade


class DrawdownPayload(TypedDict):
    current: float
    max_drawdown: float
    recovery_days: int
    time_underwater_days: int
    recovery_factor: float


class WinRatePayload(TypedDict):
    overall: float
    recent_24h: float
    bull_win_rate: float
    bear_win_rate: float
    profit_factor: float


class DriftScorePayload(TypedDict):
    composite_pct: float
    data_psi_pct: float
    model_error_pct: float
    last_check_at: datetime
    retrain_overdue: bool


class PnLPayload(TypedDict):
    today: float
    mtd: float
    ytd: float
    wins: int
    losses: int
    realtime: bool


class ExposurePayload(TypedDict):
    long_pct: float
    short_pct: float
    net_pct: float
    top_positions: List[Dict[str, Any]]
    concentration_warning: bool


class VaRCVaRPayload(TypedDict):
    var_95: float
    cvar_95: float
    stress_2020: float
    limit_breached: bool


class RiskMetricsPayload(TypedDict):
    volatility_ann: float
    beta_vs_spy: float
    skew: float
    tail_ratio: float


class SignalQueuePayload(TypedDict):
    pending: int
    executed: int
    rejected: int
    avg_rr: float
    confidence_histogram: List[float]


class ActivePositionsPayload(TypedDict):
    open_count: int
    avg_hold_days: float
    winners: int
    pnl_pct: float
    trailing_stop_ok: bool


class UniverseScorePayload(TypedDict):
    top_signals: List[Dict[str, Any]]
    coverage: int
    universe_size: int
    sector_bias: Dict[str, float]


class SystemHealthPayload(TypedDict):
    uptime_pct: float
    latency_p95_ms: float
    alert_count: int
    warning_count: int


# ---------------------------------------------------------------------------
# Metadata for metric cards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricVisualProfile:
    """Visual and layout hints for a metric card."""

    primary_color: MetricColor
    sparkline_height_px: int
    show_sparkline_mobile: bool
    show_trend: bool
    emphasize_value: bool


@dataclass(frozen=True)
class MetricRealtimeProfile:
    """Realtime behaviour hints."""

    supports_realtime: bool
    stale_time_ms: int
    websocket_channel: str | None
    aria_live: Literal["off", "polite", "assertive"]


@dataclass(frozen=True)
class MetricCardInfo:
    """
    High-level description of a metric card.

    Used by backend config/docs and can be reflected into TS.
    """

    id: MetricId
    title: str
    description: str
    visual: MetricVisualProfile
    realtime: MetricRealtimeProfile
    # Name of the TS React component, e.g. SharpeRatioCard
    component_name: str


METRIC_CARDS: List[MetricCardInfo] = [
    MetricCardInfo(
        id="sharpe_ratio",
        title="Sharpe Ratio",
        description="Overall and regime Sharpe with rolling 90d sparkline.",
        component_name="SharpeRatioCard",
        visual=MetricVisualProfile(
            primary_color="green",
            sparkline_height_px=24,
            show_sparkline_mobile=False,
            show_trend=True,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=False,
            stale_time_ms=5_000,
            websocket_channel=None,
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="drawdown",
        title="Drawdown",
        description="Current and maximum drawdown with recovery stats.",
        component_name="DrawdownCard",
        visual=MetricVisualProfile(
            primary_color="red",
            sparkline_height_px=18,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=False,
            stale_time_ms=30_000,
            websocket_channel=None,
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="win_rate",
        title="Win Rate",
        description="Overall and recent win rate with regime breakdown.",
        component_name="WinRateCard",
        visual=MetricVisualProfile(
            primary_color="green",
            sparkline_height_px=18,
            show_sparkline_mobile=False,
            show_trend=True,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=False,
            stale_time_ms=30_000,
            websocket_channel=None,
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="drift_score",
        title="Drift Score",
        description="Composite data/model drift score with retrain indicator.",
        component_name="DriftScoreCard",
        visual=MetricVisualProfile(
            primary_color="yellow",
            sparkline_height_px=18,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=True,
            stale_time_ms=10_000,
            websocket_channel="drift_updates",
            aria_live="polite",
        ),
    ),
    MetricCardInfo(
        id="pnl",
        title="P&L",
        description="Today/MTD/YTD PnL with gradient color and realtime ticker.",
        component_name="PnLCard",
        visual=MetricVisualProfile(
            primary_color="gradient",
            sparkline_height_px=24,
            show_sparkline_mobile=True,
            show_trend=True,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=True,
            stale_time_ms=5_000,
            websocket_channel="pnl_stream",
            aria_live="polite",
        ),
    ),
    MetricCardInfo(
        id="exposure",
        title="Exposure",
        description="Long/short/net exposure and top positions snippet.",
        component_name="ExposureCard",
        visual=MetricVisualProfile(
            primary_color="default",
            sparkline_height_px=16,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=False,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=True,
            stale_time_ms=10_000,
            websocket_channel="exposure_updates",
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="var_cvar",
        title="VaR & CVaR",
        description="Risk-at-risk and conditional VaR with stress scenarios.",
        component_name="VaRCVaRCard",
        visual=MetricVisualProfile(
            primary_color="red",
            sparkline_height_px=16,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=False,
            stale_time_ms=60_000,
            websocket_channel=None,
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="risk_metrics",
        title="Risk Metrics",
        description="Volatility, beta, skew, and tail ratio snapshot.",
        component_name="RiskMetricsCard",
        visual=MetricVisualProfile(
            primary_color="default",
            sparkline_height_px=0,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=False,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=False,
            stale_time_ms=60_000,
            websocket_channel=None,
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="signal_queue",
        title="Signal Queue",
        description="Pending/executed/rejected signals and average R:R.",
        component_name="SignalQueueCard",
        visual=MetricVisualProfile(
            primary_color="yellow",
            sparkline_height_px=18,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=False,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=True,
            stale_time_ms=5_000,
            websocket_channel="signal_queue",
            aria_live="polite",
        ),
    ),
    MetricCardInfo(
        id="active_positions",
        title="Active Positions",
        description="Open positions, winner ratio, trailing stop status.",
        component_name="ActivePositionsCard",
        visual=MetricVisualProfile(
            primary_color="green",
            sparkline_height_px=18,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=False,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=True,
            stale_time_ms=10_000,
            websocket_channel="positions_updates",
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="universe_score",
        title="Universe Score",
        description="Top signals, coverage, and sector bias detection.",
        component_name="UniverseScoreCard",
        visual=MetricVisualProfile(
            primary_color="gradient",
            sparkline_height_px=18,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=False,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=False,
            stale_time_ms=60_000,
            websocket_channel=None,
            aria_live="off",
        ),
    ),
    MetricCardInfo(
        id="system_health",
        title="System Health",
        description="Uptime, latency and alert summary.",
        component_name="SystemHealthCard",
        visual=MetricVisualProfile(
            primary_color="default",
            sparkline_height_px=0,
            show_sparkline_mobile=False,
            show_trend=False,
            emphasize_value=True,
        ),
        realtime=MetricRealtimeProfile(
            supports_realtime=True,
            stale_time_ms=15_000,
            websocket_channel="system_health",
            aria_live="polite",
        ),
    ),
]

_METRIC_CARD_MAP: Dict[MetricId, MetricCardInfo] = {m.id: m for m in METRIC_CARDS}


# ---------------------------------------------------------------------------
# Hook configuration facades (for TS hooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDataHookConfig:
    """
    Configuration for the frontend `useMetricData(metricId, timeframe?)` hook.
    """

    default_timeframe: str
    stale_time_ms: int
    refetch_interval_ms: int
    enable_websocket: bool


@dataclass(frozen=True)
class SparklineHookConfig:
    """Configuration for the `useSparkline` hook."""

    default_window: int
    normalize: bool


@dataclass(frozen=True)
class MetricTrendHookConfig:
    """Configuration for the `useMetricTrend` hook."""

    lookback_hours: int


USE_METRIC_DATA_CONFIG = MetricDataHookConfig(
    default_timeframe="1D",
    stale_time_ms=5_000,
    refetch_interval_ms=5_000,
    enable_websocket=True,
)

USE_SPARKLINE_CONFIG = SparklineHookConfig(
    default_window=90,
    normalize=True,
)

USE_METRIC_TREND_CONFIG = MetricTrendHookConfig(
    lookback_hours=24,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def list_metric_cards() -> List[MetricId]:
    """Return all metric card identifiers."""
    return [m.id for m in METRIC_CARDS]


def get_metric_card_info(metric_id: str) -> MetricCardInfo | None:
    """
    Look up metadata for a metric card by ID.

    Args:
        metric_id: One of the MetricId values, e.g. 'sharpe_ratio'.

    Returns:
        MetricCardInfo if known, otherwise None.
    """
    try:
        return _METRIC_CARD_MAP[metric_id]  # type: ignore[index]
    except KeyError:
        return None


def classify_sharpe(value: float) -> SharpeGrade:
    """
    Map a Sharpe ratio to a SharpeGrade, following the prompt's ranges.

    < 0.5  -> F
    0.5-1  -> D/C
    1-1.5  -> B/A
    > 1.5  -> A+
    """
    if value < 0.5:
        return "F"
    if value < 1.0:
        return "D"
    if value < 1.5:
        return "B"
    if value >= 1.5:
        return "A+"
    return "C"


__all__: List[str] = [
    "MetricTrend",
    "MetricColor",
    "SharpeGrade",
    "MetricId",
    "TrendInfo",
    "SparklineData",
    "MetricCardProps",
    "SharpeRatioPayload",
    "DrawdownPayload",
    "WinRatePayload",
    "DriftScorePayload",
    "PnLPayload",
    "ExposurePayload",
    "VaRCVaRPayload",
    "RiskMetricsPayload",
    "SignalQueuePayload",
    "ActivePositionsPayload",
    "UniverseScorePayload",
    "SystemHealthPayload",
    "MetricVisualProfile",
    "MetricRealtimeProfile",
    "MetricCardInfo",
    "METRIC_CARDS",
    "list_metric_cards",
    "get_metric_card_info",
    "MetricDataHookConfig",
    "SparklineHookConfig",
    "MetricTrendHookConfig",
    "USE_METRIC_DATA_CONFIG",
    "USE_SPARKLINE_CONFIG",
    "USE_METRIC_TREND_CONFIG",
    "classify_sharpe",
]
