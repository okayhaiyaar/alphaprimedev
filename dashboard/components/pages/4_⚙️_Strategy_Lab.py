"""
ALPHA-PRIME v2.0 - Strategy Lab (Python Facade)
===============================================

Python-side contract for the ALPHA-PRIME Strategy Lab page.

This module does NOT implement the React 18 + TypeScript UI. Instead it:

- Defines types for strategy blocks, connections, backtest results,
  optimization runs, A/B comparisons, deployment state, and templates.
- Encodes the 3-column lab layout (Builder → Live Testing → Optimize/Deploy).
- Provides configuration for Strategy Lab APIs and WebSocket channels.

The React implementation should live in a TS/TSX file, e.g.:

  dashboard/pages/4_⚙️_Strategy_Lab.tsx

and mirror the types and IDs defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Page props
# ---------------------------------------------------------------------------

StrategyLabMode = Literal["builder", "test", "optimize"]


class StrategyLabProps(TypedDict, total=False):
    strategyId: str          # edit existing strategy
    templateId: str          # load from template
    mode: StrategyLabMode    # initial mode


# ---------------------------------------------------------------------------
# Strategy graph model (React Flow / blocks)
# ---------------------------------------------------------------------------

StrategyBlockType = Literal["data", "feature", "model", "risk", "signal", "execution"]


class BlockValidation(TypedDict, total=False):
    isValid: bool
    messages: List[str]


class StrategyBlock(TypedDict, total=False):
    id: str
    type: StrategyBlockType
    label: str
    params: Dict[str, Any]
    inputs: List[str]
    outputs: List[str]
    validation: BlockValidation


class Connection(TypedDict, total=False):
    id: str
    source: str
    target: str
    sourceHandle: str | None
    targetHandle: str | None


class BacktestPoint(TypedDict):
    timestamp: datetime
    equity: float
    drawdown_pct: float


class BacktestMetrics(TypedDict, total=False):
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int
    cagr_pct: float


class BacktestResult(TypedDict, total=False):
    run_id: str
    started_at: datetime
    completed_at: datetime | None
    points: List[BacktestPoint]
    benchmark_points: List[BacktestPoint]
    metrics: BacktestMetrics
    is_live: bool
    parameters: Dict[str, Any]


# ---------------------------------------------------------------------------
# Optimization & A/B testing
# ---------------------------------------------------------------------------

class ParameterSweepPoint(TypedDict):
    param_name: str
    param_value: float
    metric_name: str         # e.g. "sharpe"
    metric_value: float


class GridSearchCell(TypedDict):
    param_x: float
    param_y: float
    metric_value: float


class WalkForwardConfig(TypedDict, total=False):
    mode: Literal["rolling", "expanding"]
    in_sample_window_days: int
    out_sample_window_days: int
    step_days: int


class OptimizationRun(TypedDict, total=False):
    run_id: str
    started_at: datetime
    completed_at: datetime | None
    sweep_points: List[ParameterSweepPoint]
    grid_cells: List[GridSearchCell]
    walk_forward: WalkForwardConfig
    best_params: Dict[str, Any]


class AbComparisonMetrics(TypedDict, total=False):
    sharpe_a: float
    sharpe_b: float
    delta_sharpe: float
    p_value: float
    regime_matrix_diff: Dict[str, float]


class AbComparison(TypedDict, total=False):
    strategy_a_id: str
    strategy_b_id: str
    equity_a: List[BacktestPoint]
    equity_b: List[BacktestPoint]
    metrics_table: List[Dict[str, Any]]
    metrics: AbComparisonMetrics


# ---------------------------------------------------------------------------
# Deployment & versioning
# ---------------------------------------------------------------------------

DeploymentMode = Literal["paper", "live", "none"]


class VersionInfo(TypedDict, total=False):
    version_id: str
    created_at: datetime
    label: str
    comment: str | None
    is_active: bool


class DeploymentLimits(TypedDict, total=False):
    max_drawdown_pct: float
    max_position_pct: float
    max_leverage: float
    daily_loss_limit_pct: float


class DeploymentState(TypedDict, total=False):
    mode: DeploymentMode
    approved: bool
    approved_by: str | None
    approved_at: datetime | None
    limits: DeploymentLimits
    performance_link: str | None
    history: List[VersionInfo]


# ---------------------------------------------------------------------------
# Strategy state (Zustand-like, minus functions)
# ---------------------------------------------------------------------------

class StrategyState(TypedDict, total=False):
    strategy_id: str | None
    name: str
    tags: List[str]
    blocks: List[StrategyBlock]
    connections: List[Connection]
    params: Dict[str, Any]
    last_backtest: BacktestResult | None
    last_optimization: OptimizationRun | None
    ab_comparison: AbComparison | None
    deployment: DeploymentState
    is_valid: bool


# ---------------------------------------------------------------------------
# Layout sections (3-column lab + mobile tabs)
# ---------------------------------------------------------------------------

SectionId = Literal[
    "builder",
    "live_testing",
    "optimization",
    "ab_comparison",
    "deployment",
    "strategy_library",
    "quick_actions",
]


@dataclass(frozen=True)
class SectionMeta:
    id: SectionId
    title: str
    description: str
    aria_role: str     # "main", "region", "complementary", "banner"
    column: int        # 1=builder, 2=testing/optimize, 3=deploy
    order: int         # order within column


SECTIONS: List[SectionMeta] = [
    SectionMeta(
        id="quick_actions",
        title="Quick Actions",
        description="Toolbar for quick test, optimization, save, deploy, and share.",
        aria_role="banner",
        column=1,
        order=0,
    ),
    SectionMeta(
        id="builder",
        title="Strategy Builder",
        description="Drag-and-drop strategy graph with data, features, models, risk, signals, and execution blocks.",
        aria_role="main",
        column=1,
        order=1,
    ),
    SectionMeta(
        id="live_testing",
        title="Live Testing",
        description="Equity curve, drawdown, metrics, and parameter sliders updating in real time.",
        aria_role="main",
        column=2,
        order=1,
    ),
    SectionMeta(
        id="optimization",
        title="Parameter Optimization",
        description="Single-parameter sweeps, grid search, walk-forward analysis, and optimization history.",
        aria_role="region",
        column=2,
        order=2,
    ),
    SectionMeta(
        id="ab_comparison",
        title="A/B Comparison",
        description="Side-by-side strategy comparison with metrics and regime matrix.",
        aria_role="region",
        column=2,
        order=3,
    ),
    SectionMeta(
        id="deployment",
        title="Deployment",
        description="Paper/live deploy controls, risk limits, version history, and performance links.",
        aria_role="complementary",
        column=3,
        order=1,
    ),
    SectionMeta(
        id="strategy_library",
        title="Strategy Library",
        description="Templates, saved strategies, search, fork, and export actions.",
        aria_role="complementary",
        column=3,
        order=2,
    ),
]


# ---------------------------------------------------------------------------
# Strategy templates library
# ---------------------------------------------------------------------------

class StrategyTemplate(TypedDict, total=False):
    id: str
    name: str
    category: str               # e.g. "Technical", "ML", "Risk", "Multi-asset"
    tags: List[str]
    description: str
    created_at: datetime


STRATEGY_TEMPLATES: List[StrategyTemplate] = [
    StrategyTemplate(
        id="ema-cross",
        name="EMA Crossover",
        category="Technical",
        tags=["trend-following", "ema", "crossover"],
        description="Classic EMA fast/slow crossover for single asset trend following.",
        created_at=datetime(2025, 1, 10),
    ),
    StrategyTemplate(
        id="rsi-divergence",
        name="RSI Divergence",
        category="Technical",
        tags=["rsi", "mean-reversion"],
        description="RSI divergence-based mean reversion strategy.",
        created_at=datetime(2025, 2, 5),
    ),
    StrategyTemplate(
        id="bb-squeeze",
        name="Bollinger Squeeze",
        category="Technical",
        tags=["volatility", "bollinger", "breakout"],
        description="Bollinger Band squeeze breakout strategy.",
        created_at=datetime(2025, 3, 12),
    ),
    StrategyTemplate(
        id="xgboost-classifier",
        name="XGBoost Classifier",
        category="ML",
        tags=["ml", "classification", "features"],
        description="Feature-based XGBoost signal classifier.",
        created_at=datetime(2025, 4, 20),
    ),
    StrategyTemplate(
        id="lstm-returns",
        name="LSTM Returns Forecaster",
        category="ML",
        tags=["lstm", "sequence", "forecast"],
        description="LSTM-based returns forecasting model.",
        created_at=datetime(2025, 5, 15),
    ),
    StrategyTemplate(
        id="risk-parity",
        name="Risk Parity",
        category="Risk",
        tags=["risk", "parity", "portfolio"],
        description="Risk parity allocation across assets.",
        created_at=datetime(2025, 6, 1),
    ),
    StrategyTemplate(
        id="vol-target",
        name="Volatility Targeting",
        category="Risk",
        tags=["volatility", "targeting"],
        description="Volatility targeting overlay strategy.",
        created_at=datetime(2025, 7, 3),
    ),
    StrategyTemplate(
        id="pairs-trading",
        name="Pairs Trading",
        category="Multi-asset",
        tags=["stat-arb", "pairs"],
        description="Cointegration-based pairs trading strategy.",
        created_at=datetime(2025, 8, 18),
    ),
]


# ---------------------------------------------------------------------------
# API + WebSocket configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyLabApiConfig:
    test_endpoint: str
    optimize_endpoint: str
    deploy_endpoint: str
    templates_endpoint: str
    history_endpoint: str


STRATEGY_LAB_API_CONFIG = StrategyLabApiConfig(
    test_endpoint="/api/v2/strategy/test",
    optimize_endpoint="/api/v2/strategy/optimize",
    deploy_endpoint="/api/v2/strategy/deploy",
    templates_endpoint="/api/v2/strategy/templates",
    history_endpoint="/api/v2/strategy/{id}/history",
)


@dataclass(frozen=True)
class StrategyLabRealtimeConfig:
    websocket_endpoint: str
    results_channel_format: str  # e.g. "/ws/strategy/{id}/live"


STRATEGY_LAB_REALTIME_CONFIG = StrategyLabRealtimeConfig(
    websocket_endpoint="/ws/strategy/live",
    results_channel_format="/ws/strategy/{id}/live",
)


# ---------------------------------------------------------------------------
# Hook configuration facades (for TS hooks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuilderHookConfig:
    max_blocks: int
    undo_stack_limit: int
    debounce_ms: int


@dataclass(frozen=True)
class LiveTestingHookConfig:
    debounce_ms: int
    throttle_ms: int
    max_points: int


@dataclass(frozen=True)
class OptimizationHookConfig:
    debounce_ms: int
    max_grid_points: int


USE_BUILDER_CONFIG = BuilderHookConfig(
    max_blocks=200,
    undo_stack_limit=100,
    debounce_ms=150,
)

USE_LIVE_TESTING_CONFIG = LiveTestingHookConfig(
    debounce_ms=250,
    throttle_ms=250,
    max_points=5_000,
)

USE_OPTIMIZATION_CONFIG = OptimizationHookConfig(
    debounce_ms=500,
    max_grid_points=10_000,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_sections() -> List[SectionId]:
    """Return all section identifiers for the Strategy Lab page."""
    return [section.id for section in SECTIONS]


def get_section_meta(section_id: str) -> SectionMeta | None:
    """Return metadata for a given lab section, if defined."""
    for section in SECTIONS:
        if section.id == section_id:
            return section
    return None


def list_templates() -> List[StrategyTemplate]:
    """Return all available strategy templates."""
    return list(STRATEGY_TEMPLATES)


__all__: List[str] = [
    "StrategyLabMode",
    "StrategyLabProps",
    "StrategyBlockType",
    "BlockValidation",
    "StrategyBlock",
    "Connection",
    "BacktestPoint",
    "BacktestMetrics",
    "BacktestResult",
    "ParameterSweepPoint",
    "GridSearchCell",
    "WalkForwardConfig",
    "OptimizationRun",
    "AbComparisonMetrics",
    "AbComparison",
    "DeploymentMode",
    "VersionInfo",
    "DeploymentLimits",
    "DeploymentState",
    "StrategyState",
    "SectionId",
    "SectionMeta",
    "SECTIONS",
    "StrategyTemplate",
    "STRATEGY_TEMPLATES",
    "StrategyLabApiConfig",
    "STRATEGY_LAB_API_CONFIG",
    "StrategyLabRealtimeConfig",
    "STRATEGY_LAB_REALTIME_CONFIG",
    "BuilderHookConfig",
    "LiveTestingHookConfig",
    "OptimizationHookConfig",
    "USE_BUILDER_CONFIG",
    "USE_LIVE_TESTING_CONFIG",
    "USE_OPTIMIZATION_CONFIG",
    "list_sections",
    "get_section_meta",
    "list_templates",
]
