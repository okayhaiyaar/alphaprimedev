"""
ALPHA-PRIME v2.0 - Risk Monitor (Python Facade)
===============================================

Python-side contract for the ALPHA-PRIME Real-time Risk Management Dashboard.

This module does NOT implement the React 18 + TypeScript UI. Instead it:

- Defines page props and core risk data structures (metrics, alerts,
  stress tests, positions, controls).
- Encodes the 4-quadrant control room layout and mobile priority stack.
- Provides configuration for risk APIs and WebSocket endpoints.

The React implementation should live in a TS/TSX file, e.g.:

  dashboard/pages/5_🛡️_Risk_Monitor.tsx

and mirror the types and IDs defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Page props
# ---------------------------------------------------------------------------

class RiskMonitorProps(TypedDict, total=False):
    portfolioId: str
    realtime: bool
    timezone: str
    userRole: str


# ---------------------------------------------------------------------------
# Core risk types
# ---------------------------------------------------------------------------

AlertSeverity = Literal["CRITICAL", "WARNING", "INFO"]
AlertSource = Literal["portfolio", "model", "system"]


class RiskAlert(TypedDict, total=False):
    id: str
    severity: AlertSeverity
    message: str
    source: AlertSource
    acknowledged: bool
    auto_remediate: bool
    created_at: datetime
    acknowledged_by: str | None
    acknowledged_at: datetime | None


class RiskBreach(TypedDict, total=False):
    id: str
    metric: str               # e.g. "VaR 95", "Concentration AAPL"
    value: float
    limit: float
    severity: AlertSeverity
    breached_at: datetime


class ExposureBreakdown(TypedDict, total=False):
    long: float
    short: float
    net: float


class HoldingConcentration(TypedDict, total=False):
    symbol: str
    pct_portfolio: float
    breached: bool


class SectorExposure(TypedDict, total=False):
    sector: str
    pct_portfolio: float
    breached: bool


class StressScenarioResult(TypedDict, total=False):
    name: str                 # e.g. "2020 Crash"
    loss_pct: float
    limit_pct: float | None
    kind: Literal["historical", "custom", "monte_carlo"]


class MarginalVarItem(TypedDict, total=False):
    symbol: str
    mvar: float
    pct_of_total_var: float


class RiskMetrics(TypedDict, total=False):
    var_95: float
    var_99: float
    cvar_95: float
    expected_shortfall: float
    volatility: float
    volatility_target: float
    beta: float
    skew: float
    tail_ratio: float
    exposure: ExposureBreakdown
    breaches: List[RiskBreach]
    stress_tests: List[StressScenarioResult]
    marginal_var: List[MarginalVarItem]


# ---------------------------------------------------------------------------
# Portfolio exposure & correlation
# ---------------------------------------------------------------------------

class CorrelationCell(TypedDict, total=False):
    symbol_a: str
    symbol_b: str
    correlation: float


class PortfolioExposureSummary(TypedDict, total=False):
    exposure: ExposureBreakdown
    top_holdings: List[HoldingConcentration]
    sectors: List[SectorExposure]
    correlation_matrix: List[CorrelationCell]
    limits_exceeded: int
    limits_total: int


# ---------------------------------------------------------------------------
# Positions table
# ---------------------------------------------------------------------------

class PositionRow(TypedDict, total=False):
    symbol: str
    size: float
    pnl: float
    pct_portfolio: float
    beta: float
    marginal_var: float
    limit: float
    breached: bool


class PositionsTable(TypedDict, total=False):
    rows: List[PositionRow]
    as_of: datetime


# ---------------------------------------------------------------------------
# Stress tests panel model
# ---------------------------------------------------------------------------

class StressTestsPanelData(TypedDict, total=False):
    historical: List[StressScenarioResult]
    custom: List[StressScenarioResult]
    monte_carlo: StressScenarioResult | None


# ---------------------------------------------------------------------------
# Risk controls (manual + auto)
# ---------------------------------------------------------------------------

RiskActionId = Literal[
    "emergency_stop",
    "reduce_exposure_25",
    "increase_limits",
    "apply_auto_rules",
]


class RiskControlAction(TypedDict, total=False):
    id: RiskActionId
    label: str
    description: str
    requires_admin: bool


class AutoRule(TypedDict, total=False):
    id: str
    name: str
    condition: str       # human-readable, e.g. "Max DD > 10%"
    action: str          # e.g. "Reduce positions by 50%"
    enabled: bool


class RiskControlsState(TypedDict, total=False):
    manual_actions: List[RiskControlAction]
    auto_rules: List[AutoRule]


# ---------------------------------------------------------------------------
# Overall risk dashboard state
# ---------------------------------------------------------------------------

class RiskDashboardState(TypedDict, total=False):
    alerts: List[RiskAlert]
    exposure_summary: PortfolioExposureSummary
    metrics: RiskMetrics
    stress_tests_panel: StressTestsPanelData
    positions_table: PositionsTable
    controls: RiskControlsState
    generated_at: datetime


# ---------------------------------------------------------------------------
# Layout sections (4-quadrant desktop + mobile priority)
# ---------------------------------------------------------------------------

SectionId = Literal[
    "risk_alerts_panel",
    "portfolio_exposure",
    "live_risk_metrics",
    "stress_tests",
    "positions_table",
    "risk_controls_panel",
]


@dataclass(frozen=True)
class SectionMeta:
    id: SectionId
    title: str
    description: str
    aria_role: str       # e.g. "main", "region", "complementary"
    quadrant: int        # 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right
    order: int           # ordering hint within layout
    mobile_priority: int # 1=highest priority


SECTIONS: List[SectionMeta] = [
    SectionMeta(
        id="risk_alerts_panel",
        title="Risk Alerts",
        description="Critical, warning, and informational risk alerts with auto-remediation.",
        aria_role="main",
        quadrant=2,
        order=1,
        mobile_priority=1,
    ),
    SectionMeta(
        id="portfolio_exposure",
        title="Portfolio Exposure",
        description="Long/short/net exposure, concentration, sectors, and correlation heatmap.",
        aria_role="main",
        quadrant=1,
        order=1,
        mobile_priority=2,
    ),
    SectionMeta(
        id="live_risk_metrics",
        title="Live Risk Metrics",
        description="VaR, CVaR, volatility, beta, skew, tail ratio, and limit gauges.",
        aria_role="region",
        quadrant=3,
        order=1,
        mobile_priority=3,
    ),
    SectionMeta(
        id="stress_tests",
        title="Stress Tests",
        description="Historical and custom scenarios plus Monte Carlo loss distribution.",
        aria_role="region",
        quadrant=4,
        order=1,
        mobile_priority=4,
    ),
    SectionMeta(
        id="positions_table",
        title="Positions",
        description="Positions table with VaR ranking, bulk reduce and trailing stops.",
        aria_role="region",
        quadrant=1,
        order=2,
        mobile_priority=5,
    ),
    SectionMeta(
        id="risk_controls_panel",
        title="Risk Controls",
        description="Emergency stop, exposure reduction, limits override, and auto rules.",
        aria_role="complementary",
        quadrant=2,
        order=2,
        mobile_priority=6,
    ),
]


# ---------------------------------------------------------------------------
# API & WebSocket configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskApiConfig:
    metrics_endpoint: str
    positions_endpoint: str
    stress_tests_endpoint: str
    action_endpoint: str


RISK_API_CONFIG = RiskApiConfig(
    metrics_endpoint="/api/v2/risk/metrics",
    positions_endpoint="/api/v2/risk/positions",
    stress_tests_endpoint="/api/v2/risk/stress-tests",
    action_endpoint="/api/v2/risk/action",
)


@dataclass(frozen=True)
class RiskRealtimeConfig:
    websocket_metrics: str
    websocket_alerts: str
    update_interval_ms: int


RISK_REALTIME_CONFIG = RiskRealtimeConfig(
    websocket_metrics="/ws/risk/live",
    websocket_alerts="/ws/risk/alerts",
    update_interval_ms=1_000,
)


# ---------------------------------------------------------------------------
# Hook configuration facades (for TS hooks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskMetricsHookConfig:
    refetch_interval_ms: int
    stale_time_ms: int


@dataclass(frozen=True)
class RiskAlertsHookConfig:
    buffer_size: int


@dataclass(frozen=True)
class PositionsHookConfig:
    refetch_interval_ms: int
    virtualized_rows: int


USE_RISK_METRICS_CONFIG = RiskMetricsHookConfig(
    refetch_interval_ms=5_000,
    stale_time_ms=0,
)

USE_RISK_ALERTS_CONFIG = RiskAlertsHookConfig(
    buffer_size=500,
)

USE_POSITIONS_CONFIG = PositionsHookConfig(
    refetch_interval_ms=15_000,
    virtualized_rows=100_000,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_sections() -> List[SectionId]:
    """Return all section identifiers for the Risk Monitor page."""
    return [section.id for section in SECTIONS]


def get_section_meta(section_id: str) -> SectionMeta | None:
    """Return metadata for a given section, if defined."""
    for section in SECTIONS:
        if section.id == section_id:
            return section
    return None


def is_breach_critical(breach: RiskBreach) -> bool:
    """Return True if a breach is considered critical."""
    return breach.get("severity") == "CRITICAL"


__all__: List[str] = [
    "RiskMonitorProps",
    "AlertSeverity",
    "AlertSource",
    "RiskAlert",
    "RiskBreach",
    "ExposureBreakdown",
    "HoldingConcentration",
    "SectorExposure",
    "StressScenarioResult",
    "MarginalVarItem",
    "RiskMetrics",
    "CorrelationCell",
    "PortfolioExposureSummary",
    "PositionRow",
    "PositionsTable",
    "StressTestsPanelData",
    "RiskActionId",
    "RiskControlAction",
    "AutoRule",
    "RiskControlsState",
    "RiskDashboardState",
    "SectionId",
    "SectionMeta",
    "SECTIONS",
    "RiskApiConfig",
    "RISK_API_CONFIG",
    "RiskRealtimeConfig",
    "RISK_REALTIME_CONFIG",
    "RiskMetricsHookConfig",
    "RiskAlertsHookConfig",
    "PositionsHookConfig",
    "USE_RISK_METRICS_CONFIG",
    "USE_RISK_ALERTS_CONFIG",
    "USE_POSITIONS_CONFIG",
    "list_sections",
    "get_section_meta",
    "is_breach_critical",
]
