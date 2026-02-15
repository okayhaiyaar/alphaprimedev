"""
ALPHA-PRIME v2.0 - Research & Backtesting Workspace (Python Facade)
===================================================================

Python-side contract for the ALPHA-PRIME Research page.

This module does NOT implement the React 18 + TypeScript UI. Instead it:

- Defines types for notebook cells, pipelines, backtest previews, and
  research results.
- Encodes layout sections (notebook, pipeline builder, live preview,
  results, code editor, mobile tabs).
- Provides configuration for notebook/pipeline APIs and live WebSocket
  streams.

The React implementation should live in a TS/TSX file, e.g.:

  dashboard/pages/3_🔬_Research.tsx

and mirror the types and IDs defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Page props
# ---------------------------------------------------------------------------

class ResearchPageProps(TypedDict, total=False):
    notebookId: str        # existing notebook ID
    pipelineId: str        # existing pipeline ID
    userRole: str          # permissions / RBAC


# ---------------------------------------------------------------------------
# Notebook model
# ---------------------------------------------------------------------------

CellType = Literal["markdown", "python", "sql", "visualization"]


class CellOutput(TypedDict, total=False):
    """
    Generic cell output, to be rendered on the frontend.

    kind:
      - "text": plain text / stdout
      - "html": HTML fragment
      - "plot": Plotly figure JSON
      - "table": tabular data
      - "error": traceback text
    """
    kind: Literal["text", "html", "plot", "table", "error"]
    data: Any
    mimeType: str | None


class NotebookCell(TypedDict, total=False):
    id: str
    type: CellType
    content: str
    outputs: List[CellOutput]
    executed: bool
    executionTime: float          # milliseconds
    lastExecutedAt: datetime | None


class NotebookMetadata(TypedDict, total=False):
    id: str
    title: str
    createdAt: datetime
    updatedAt: datetime
    tags: List[str]
    kernel: str                    # e.g. "python", "sql"
    autosaveIntervalSec: int


class NotebookState(TypedDict, total=False):
    metadata: NotebookMetadata
    cells: List[NotebookCell]
    dirty: bool                    # unsaved changes


# ---------------------------------------------------------------------------
# Pipeline model (React Flow)
# ---------------------------------------------------------------------------

PipelineNodeType = Literal["data", "feature", "model", "backtest", "risk"]


class PipelineNode(TypedDict, total=False):
    id: str
    type: PipelineNodeType
    label: str
    params: Dict[str, Any]
    inputs: List[str]
    outputs: List[str]


class PipelineEdge(TypedDict, total=False):
    id: str
    source: str
    target: str
    label: str | None


class PipelineGraph(TypedDict, total=False):
    id: str
    name: str
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]
    createdAt: datetime
    updatedAt: datetime


# ---------------------------------------------------------------------------
# Live backtest preview
# ---------------------------------------------------------------------------

class BacktestPoint(TypedDict):
    timestamp: datetime
    equity: float
    drawdown_pct: float


class BacktestPreviewMetrics(TypedDict, total=False):
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int
    cagr_pct: float


class BacktestPreview(TypedDict, total=False):
    backtest_id: str | None
    is_running: bool
    benchmark: str | None
    equity_curve: List[BacktestPoint]
    drawdown_curve: List[BacktestPoint]
    metrics: BacktestPreviewMetrics
    updatedAt: datetime | None


# ---------------------------------------------------------------------------
# Results panel data
# ---------------------------------------------------------------------------

class PerformanceSummaryRow(TypedDict):
    metric: str                 # e.g. "Sharpe", "Sortino"
    value: float
    benchmark: float | None
    delta: float | None         # value - benchmark


class FeatureImportanceItem(TypedDict):
    feature: str
    importance: float           # normalized 0-1


class TradeDistributionBin(TypedDict):
    bin_label: str
    count: int
    pct: float


class DrawdownEvent(TypedDict):
    start: datetime
    trough: datetime
    end: datetime | None
    depth_pct: float
    duration_days: int


class ResultStatTests(TypedDict, total=False):
    t_test_p_value: float
    monte_carlo_p_value: float
    bootstrap_p_value: float | None


class ResultsPanelData(TypedDict, total=False):
    performance_summary: List[PerformanceSummaryRow]
    feature_importance: List[FeatureImportanceItem]
    trade_distribution: List[TradeDistributionBin]
    drawdowns: List[DrawdownEvent]
    stat_tests: ResultStatTests


# ---------------------------------------------------------------------------
# Code editor + terminal
# ---------------------------------------------------------------------------

EditorLanguage = Literal["python", "javascript", "sql"]


class EditorState(TypedDict, total=False):
    language: EditorLanguage
    content: str
    lastSavedAt: datetime | None
    cursorLine: int
    cursorColumn: int


class TerminalMessage(TypedDict, total=False):
    timestamp: datetime
    stream: Literal["stdout", "stderr", "system"]
    text: str


class TerminalState(TypedDict, total=False):
    messages: List[TerminalMessage]
    kernel_status: Literal["idle", "busy", "disconnected"]
    lastHeartbeatAt: datetime | None


# ---------------------------------------------------------------------------
# Overall research workspace state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    notebook: NotebookState
    pipeline: PipelineGraph
    backtest_preview: BacktestPreview
    results_panel: ResultsPanelData
    editor: EditorState
    terminal: TerminalState


# ---------------------------------------------------------------------------
# Layout sections (IDE-style panes + mobile tabs)
# ---------------------------------------------------------------------------

SectionId = Literal[
    "notebook",
    "pipeline_builder",
    "live_preview",
    "results_panel",
    "code_editor",
    "mobile_tabs",
]


@dataclass(frozen=True)
class SectionMeta:
    id: SectionId
    title: str
    description: str
    aria_role: str      # e.g. "main", "complementary"
    column: int         # desktop column index (1-4)
    order: int          # order within column


SECTIONS: List[SectionMeta] = [
    SectionMeta(
        id="notebook",
        title="Notebook",
        description="Jupyter-style notebook with markdown, code, and outputs.",
        aria_role="main",
        column=1,
        order=1,
    ),
    SectionMeta(
        id="pipeline_builder",
        title="Pipeline Builder",
        description="React Flow visual pipeline for data, features, model, and backtest.",
        aria_role="region",
        column=1,
        order=2,
    ),
    SectionMeta(
        id="live_preview",
        title="Live Backtest Preview",
        description="Equity curve, drawdown, and key metrics updating as you code.",
        aria_role="main",
        column=2,
        order=1,
    ),
    SectionMeta(
        id="results_panel",
        title="Results Panel",
        description="Performance table, feature importance, distributions, drawdowns, stats.",
        aria_role="region",
        column=2,
        order=2,
    ),
    SectionMeta(
        id="code_editor",
        title="Code Editor",
        description="Monaco editor plus integrated terminal for Python kernel.",
        aria_role="complementary",
        column=3,
        order=1,
    ),
    SectionMeta(
        id="mobile_tabs",
        title="Mobile Tabs",
        description="Tabbed layout for notebook, pipeline, preview, and editor on small screens.",
        aria_role="tablist",
        column=4,
        order=1,
    ),
]


# ---------------------------------------------------------------------------
# API + WebSocket configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotebookApiConfig:
    execute_cell_endpoint: str
    list_notebooks_endpoint: str
    autosave_interval_sec: int


NOTEBOOK_API_CONFIG = NotebookApiConfig(
    execute_cell_endpoint="/api/v2/research/notebook/execute",
    list_notebooks_endpoint="/api/v2/research/notebooks",
    autosave_interval_sec=30,
)


@dataclass(frozen=True)
class PipelineApiConfig:
    run_pipeline_endpoint: str
    get_backtest_endpoint: str


PIPELINE_API_CONFIG = PipelineApiConfig(
    run_pipeline_endpoint="/api/v2/research/pipeline/run",
    get_backtest_endpoint="/api/v2/backtests/{id}",
)


@dataclass(frozen=True)
class ResearchRealtimeConfig:
    websocket_endpoint: str
    channel_notebook: str
    channel_pipeline: str
    channel_backtest: str


RESEARCH_REALTIME_CONFIG = ResearchRealtimeConfig(
    websocket_endpoint="/ws/research/live",
    channel_notebook="notebook",
    channel_pipeline="pipeline",
    channel_backtest="backtest",
)


# ---------------------------------------------------------------------------
# Hook configuration facades (useNotebook, usePipeline, useLiveBacktest)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotebookHookConfig:
    debounce_ms: int
    max_cells_virtualized: int


@dataclass(frozen=True)
class PipelineHookConfig:
    debounce_ms: int
    max_nodes_virtualized: int


@dataclass(frozen=True)
class LiveBacktestHookConfig:
    debounce_ms: int
    max_points: int


USE_NOTEBOOK_CONFIG = NotebookHookConfig(
    debounce_ms=200,
    max_cells_virtualized=500,
)

USE_PIPELINE_CONFIG = PipelineHookConfig(
    debounce_ms=500,
    max_nodes_virtualized=200,
)

USE_LIVE_BACKTEST_CONFIG = LiveBacktestHookConfig(
    debounce_ms=500,
    max_points=5_000,
)


# ---------------------------------------------------------------------------
# Research templates (starter notebooks)
# ---------------------------------------------------------------------------

class ResearchTemplate(TypedDict, total=False):
    id: str
    name: str
    description: str


RESEARCH_TEMPLATES: List[ResearchTemplate] = [
    {
        "id": "ema-crossover-starter",
        "name": "EMA Crossover Starter",
        "description": "Basic EMA crossover strategy with entry/exit rules.",
    },
    {
        "id": "rsi-mean-reversion",
        "name": "RSI Mean Reversion",
        "description": "RSI-based mean reversion notebook with parameter sweep.",
    },
    {
        "id": "ml-feature-pipeline",
        "name": "ML Feature Pipeline",
        "description": "Feature engineering and XGBoost model for signals.",
    },
    {
        "id": "risk-parity-portfolio",
        "name": "Risk Parity Portfolio",
        "description": "Risk parity allocation with backtest and diagnostics.",
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def list_sections() -> List[SectionId]:
    """Return all section identifiers for the Research page."""
    return [section.id for section in SECTIONS]


def get_section_meta(section_id: str) -> SectionMeta | None:
    """Return metadata for a given section, if defined."""
    for section in SECTIONS:
        if section.id == section_id:
            return section
    return None


def list_templates() -> List[ResearchTemplate]:
    """Return all pre-loaded research notebook templates."""
    return list(RESEARCH_TEMPLATES)


__all__: List[str] = [
    "ResearchPageProps",
    "CellType",
    "CellOutput",
    "NotebookCell",
    "NotebookMetadata",
    "NotebookState",
    "PipelineNodeType",
    "PipelineNode",
    "PipelineEdge",
    "PipelineGraph",
    "BacktestPoint",
    "BacktestPreviewMetrics",
    "BacktestPreview",
    "PerformanceSummaryRow",
    "FeatureImportanceItem",
    "TradeDistributionBin",
    "DrawdownEvent",
    "ResultStatTests",
    "ResultsPanelData",
    "EditorLanguage",
    "EditorState",
    "TerminalMessage",
    "TerminalState",
    "ResearchState",
    "SectionId",
    "SectionMeta",
    "SECTIONS",
    "NotebookApiConfig",
    "NOTEBOOK_API_CONFIG",
    "PipelineApiConfig",
    "PIPELINE_API_CONFIG",
    "ResearchRealtimeConfig",
    "RESEARCH_REALTIME_CONFIG",
    "NotebookHookConfig",
    "PipelineHookConfig",
    "LiveBacktestHookConfig",
    "USE_NOTEBOOK_CONFIG",
    "USE_PIPELINE_CONFIG",
    "USE_LIVE_BACKTEST_CONFIG",
    "ResearchTemplate",
    "RESEARCH_TEMPLATES",
    "list_sections",
    "get_section_meta",
    "list_templates",
]
