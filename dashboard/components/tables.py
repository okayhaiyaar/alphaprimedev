"""
ALPHA-PRIME v2.0 - Trading Data Tables (Python Facade)
======================================================

Python-side model of the trading data tables implemented in the
frontend (React + TypeScript). This module is intentionally free
of any React/TS syntax so it can live in `dashboard/components`
without breaking Python tooling.

It encodes:

- Table names, row schemas, and column metadata.
- Feature flags (virtual scroll, realtime, bulk actions).
- UX hints (keyboard navigation, error/empty messages).

The actual interactive implementation (TanStack Table/Virtual,
Framer Motion, shadcn/ui) should live in a separate TS/TSX file,
e.g. `dashboard/components/tables.tsx`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, TypedDict


# ---------------------------------------------------------------------------
# Shared type aliases (Python view of TS types)
# ---------------------------------------------------------------------------

TradingColumnType = Literal[
    "symbol",
    "timestamp",
    "price",
    "pnl",
    "pnl_pct",
    "signal",
    "confidence",
    "rr_ratio",
    "status",
    "sharpe",
    "max_dd",
    "win_rate",
]

ColorMode = Literal["pnl", "signal", "risk"]
FormatMode = Literal["currency", "percentage", "compact_time"]

TableName = Literal[
    "TradesTable",
    "PositionsTable",
    "StrategiesTable",
    "PerformanceTable",
    "SignalsTable",
    "UniverseTable",
    "RiskTable",
]


# ---------------------------------------------------------------------------
# Column metadata and table description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnInfo:
    """
    Lightweight description of a table column.

    Intended to mirror the TS-side `TradingColumnDef<TData>` shape.
    """

    id: str
    title: str
    type: TradingColumnType | None = None
    color_mode: ColorMode | None = None
    format: FormatMode | None = None
    sortable: bool = True
    filterable: bool = True
    resizable: bool = True
    pinnable: bool = True


@dataclass(frozen=True)
class TableUXHints:
    """
    UX-related hints for the frontend implementation.

    These are non-functional flags the TS layer can use when wiring up
    TanStack Table, TanStack Virtual, and keyboard bindings.
    """

    virtual_scroll: bool
    realtime: bool
    infinite_scroll: bool
    sticky_header: bool
    keyboard_mode: Literal["vim_like", "standard"]
    bulk_actions: List[str]
    default_empty_message: str
    default_error_message: str


@dataclass(frozen=True)
class TableInfo:
    """
    High-level table description combining columns and UX features.
    """

    name: TableName
    description: str
    columns: List[ColumnInfo]
    enable_pagination: bool
    enable_sorting: bool
    enable_filters: bool
    enable_bulk_actions: bool
    ux: TableUXHints


# ---------------------------------------------------------------------------
# Row schemas (Python view of TS row interfaces)
# ---------------------------------------------------------------------------


class TradeRow(TypedDict):
    id: str
    timestamp: datetime
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: float
    entry: float
    exit: float | None
    pnl: float
    pnl_pct: float
    duration: str
    signal_confidence: float


class PositionRow(TypedDict):
    id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: float
    entry_avg: float
    current_price: float
    pnl: float
    pnl_pct: float
    unrealized_pnl: float
    exposure_pct: float


class StrategyRow(TypedDict):
    id: str
    name: str
    status: Literal["active", "paused", "archived"]
    sharpe: float
    max_dd: float
    win_rate: float
    active_positions: int
    last_signal: str
    drift_score: float


class PerformanceRow(TypedDict):
    period: str
    total_return: float
    sharpe: float
    sortino: float
    max_dd: float
    win_rate: float
    profit_factor: float
    trades: int


class SignalRow(TypedDict):
    id: str
    timestamp: datetime
    symbol: str
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    strength: float
    target: float
    stop: float
    rr_ratio: float
    status: Literal["pending", "executed", "cancelled"]


class UniverseRow(TypedDict):
    symbol: str
    sector: str
    market_cap: float
    price: float
    volume: float
    beta: float
    pe_ratio: float
    rsi: float
    score: float


class RiskRow(TypedDict):
    risk_source: str
    current: float
    limit: float
    violation_pct: float
    action_required: str
    last_updated: datetime


# ---------------------------------------------------------------------------
# Column definitions per table (match prompt columns)
# ---------------------------------------------------------------------------

TRADES_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("timestamp", "Time", type="timestamp", format="compact_time"),
    ColumnInfo("symbol", "Symbol", type="symbol"),
    ColumnInfo("side", "Side"),
    ColumnInfo("qty", "Qty"),
    ColumnInfo("entry", "Entry", type="price", format="currency"),
    ColumnInfo("exit", "Exit", type="price", format="currency"),
    ColumnInfo("pnl", "PnL", type="pnl", color_mode="pnl", format="currency"),
    ColumnInfo("pnl_pct", "PnL %", type="pnl_pct", color_mode="pnl", format="percentage"),
    ColumnInfo("duration", "Duration"),
    ColumnInfo("signal_confidence", "Conf.", type="confidence", format="percentage"),
]

POSITIONS_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("symbol", "Symbol", type="symbol"),
    ColumnInfo("side", "Side"),
    ColumnInfo("qty", "Qty"),
    ColumnInfo("entry_avg", "Entry", type="price", format="currency"),
    ColumnInfo("current_price", "Price", type="price", format="currency"),
    ColumnInfo("pnl", "PnL", type="pnl", color_mode="pnl", format="currency"),
    ColumnInfo("pnl_pct", "PnL %", type="pnl_pct", color_mode="pnl", format="percentage"),
    ColumnInfo("unrealized_pnl", "Unrealized", type="pnl", color_mode="pnl", format="currency"),
    ColumnInfo("exposure_pct", "Exposure", type="pnl_pct", format="percentage"),
]

STRATEGIES_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("name", "Strategy"),
    ColumnInfo("status", "Status", type="status"),
    ColumnInfo("sharpe", "Sharpe", type="sharpe"),
    ColumnInfo("max_dd", "Max DD", type="max_dd", format="percentage"),
    ColumnInfo("win_rate", "Win %", type="win_rate", format="percentage"),
    ColumnInfo("active_positions", "Positions"),
    ColumnInfo("last_signal", "Last Signal"),
    ColumnInfo("drift_score", "Drift", color_mode="risk"),
]

PERFORMANCE_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("period", "Period"),
    ColumnInfo("total_return", "Return", type="pnl_pct", format="percentage"),
    ColumnInfo("sharpe", "Sharpe", type="sharpe"),
    ColumnInfo("sortino", "Sortino"),
    ColumnInfo("max_dd", "Max DD", type="max_dd", format="percentage"),
    ColumnInfo("win_rate", "Win %", type="win_rate", format="percentage"),
    ColumnInfo("profit_factor", "PF"),
    ColumnInfo("trades", "Trades"),
]

SIGNALS_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("timestamp", "Time", type="timestamp", format="compact_time"),
    ColumnInfo("symbol", "Symbol", type="symbol"),
    ColumnInfo("signal", "Signal", type="signal", color_mode="signal"),
    ColumnInfo("confidence", "Conf.", type="confidence", format="percentage"),
    ColumnInfo("strength", "Strength", format="percentage"),
    ColumnInfo("target", "Target", type="price", format="currency"),
    ColumnInfo("stop", "Stop", type="price", format="currency"),
    ColumnInfo("rr_ratio", "R:R", type="rr_ratio"),
    ColumnInfo("status", "Status", type="status"),
]

UNIVERSE_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("symbol", "Symbol", type="symbol"),
    ColumnInfo("sector", "Sector"),
    ColumnInfo("market_cap", "Mkt Cap"),
    ColumnInfo("price", "Price", type="price", format="currency"),
    ColumnInfo("volume", "Volume"),
    ColumnInfo("beta", "Beta"),
    ColumnInfo("pe_ratio", "P/E"),
    ColumnInfo("rsi", "RSI"),
    ColumnInfo("score", "Score", color_mode="risk"),
]

RISK_COLUMNS: List[ColumnInfo] = [
    ColumnInfo("risk_source", "Risk Source"),
    ColumnInfo("current", "Current", format="percentage"),
    ColumnInfo("limit", "Limit", format="percentage"),
    ColumnInfo(
        "violation_pct",
        "Violation",
        type="pnl_pct",
        color_mode="risk",
        format="percentage",
    ),
    ColumnInfo("action_required", "Action"),
    ColumnInfo("last_updated", "Updated", type="timestamp", format="compact_time"),
]


# ---------------------------------------------------------------------------
# UX hints per table (virtual scroll, realtime, bulk actions, keyboard)
# ---------------------------------------------------------------------------

TRADES_UX = TableUXHints(
    virtual_scroll=True,
    infinite_scroll=True,
    realtime=True,
    sticky_header=True,
    keyboard_mode="vim_like",
    bulk_actions=["close_trades", "archive_trades"],
    default_empty_message="No trades yet. Check back later.",
    default_error_message="Unable to load trades. Please retry.",
)

POSITIONS_UX = TableUXHints(
    virtual_scroll=True,
    infinite_scroll=True,
    realtime=True,
    sticky_header=True,
    keyboard_mode="vim_like",
    bulk_actions=["close_positions", "adjust_positions", "set_trailing_stop"],
    default_empty_message="No open positions.",
    default_error_message="Unable to load positions.",
)

STRATEGIES_UX = TableUXHints(
    virtual_scroll=True,
    infinite_scroll=False,
    realtime=False,
    sticky_header=True,
    keyboard_mode="standard",
    bulk_actions=["toggle_active", "archive_strategies"],
    default_empty_message="No strategies configured.",
    default_error_message="Unable to load strategies.",
)

PERFORMANCE_UX = TableUXHints(
    virtual_scroll=False,
    infinite_scroll=False,
    realtime=False,
    sticky_header=True,
    keyboard_mode="standard",
    bulk_actions=[],
    default_empty_message="No performance data.",
    default_error_message="Unable to load performance.",
)

SIGNALS_UX = TableUXHints(
    virtual_scroll=True,
    infinite_scroll=True,
    realtime=True,
    sticky_header=True,
    keyboard_mode="vim_like",
    bulk_actions=["execute_signals", "cancel_signals"],
    default_empty_message="No signals at the moment.",
    default_error_message="Unable to load signals.",
)

UNIVERSE_UX = TableUXHints(
    virtual_scroll=True,
    infinite_scroll=True,
    realtime=False,
    sticky_header=True,
    keyboard_mode="standard",
    bulk_actions=["add_to_watchlist", "remove_from_watchlist"],
    default_empty_message="Universe is empty.",
    default_error_message="Unable to load universe.",
)

RISK_UX = TableUXHints(
    virtual_scroll=False,
    infinite_scroll=False,
    realtime=False,
    sticky_header=True,
    keyboard_mode="standard",
    bulk_actions=["acknowledge_risk", "trigger_auto_remediation"],
    default_empty_message="No risk items.",
    default_error_message="Unable to load risk overview.",
)


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------

TABLES: List[TableInfo] = [
    TableInfo(
        name="TradesTable",
        description="Executed trades with PnL and signal confidence.",
        columns=TRADES_COLUMNS,
        enable_pagination=True,
        enable_sorting=True,
        enable_filters=True,
        enable_bulk_actions=True,
        ux=TRADES_UX,
    ),
    TableInfo(
        name="PositionsTable",
        description="Open positions with unrealized PnL and exposure.",
        columns=POSITIONS_COLUMNS,
        enable_pagination=True,
        enable_sorting=True,
        enable_filters=True,
        enable_bulk_actions=True,
        ux=POSITIONS_UX,
    ),
    TableInfo(
        name="StrategiesTable",
        description="Strategy-level performance and drift overview.",
        columns=STRATEGIES_COLUMNS,
        enable_pagination=True,
        enable_sorting=True,
        enable_filters=True,
        enable_bulk_actions=True,
        ux=STRATEGIES_UX,
    ),
    TableInfo(
        name="PerformanceTable",
        description="Aggregated performance metrics by period.",
        columns=PERFORMANCE_COLUMNS,
        enable_pagination=True,
        enable_sorting=True,
        enable_filters=True,
        enable_bulk_actions=False,
        ux=PERFORMANCE_UX,
    ),
    TableInfo(
        name="SignalsTable",
        description="Live strategy signals with risk parameters.",
        columns=SIGNALS_COLUMNS,
        enable_pagination=True,
        enable_sorting=True,
        enable_filters=True,
        enable_bulk_actions=True,
        ux=SIGNALS_UX,
    ),
    TableInfo(
        name="UniverseTable",
        description="Universe-level fundamentals and technical scores.",
        columns=UNIVERSE_COLUMNS,
        enable_pagination=True,
        enable_sorting=True,
        enable_filters=True,
        enable_bulk_actions=True,
        ux=UNIVERSE_UX,
    ),
    TableInfo(
        name="RiskTable",
        description="Risk limits, utilisation, and remediation status.",
        columns=RISK_COLUMNS,
        enable_pagination=False,
        enable_sorting=False,
        enable_filters=False,
        enable_bulk_actions=True,
        ux=RISK_UX,
    ),
]

_TABLE_MAP: Dict[TableName, TableInfo] = {t.name: t for t in TABLES}


def list_tables() -> List[TableName]:
    """
    Return all registered table names.

    This is useful for APIs that expose supported table IDs to the frontend.
    """
    return [t.name for t in TABLES]


def get_table_info(name: str) -> TableInfo | None:
    """
    Look up a table description by name.

    Args:
        name: Table name (e.g. 'TradesTable').

    Returns:
        TableInfo if the name is known, otherwise None.
    """
    try:
        return _TABLE_MAP[name]  # type: ignore[index]
    except KeyError:
        return None


__all__: List[str] = [
    "TradingColumnType",
    "ColorMode",
    "FormatMode",
    "TableName",
    "ColumnInfo",
    "TableUXHints",
    "TableInfo",
    "TradeRow",
    "PositionRow",
    "StrategyRow",
    "PerformanceRow",
    "SignalRow",
    "UniverseRow",
    "RiskRow",
    "TRADES_COLUMNS",
    "POSITIONS_COLUMNS",
    "STRATEGIES_COLUMNS",
    "PERFORMANCE_COLUMNS",
    "SIGNALS_COLUMNS",
    "UNIVERSE_COLUMNS",
    "RISK_COLUMNS",
    "TABLES",
    "list_tables",
    "get_table_info",
]
