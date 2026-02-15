"""
ALPHA-PRIME v2.0 Dashboard Components
=====================================

Python-side registry for the ALPHA-PRIME dashboard frontend component
library. This file mirrors the TypeScript `dashboard/components` barrel
in a lightweight, Python-friendly way so that backend code can:

- Refer to stable component names (for layouts, schemas, config).
- Inspect available components and their metadata.
- Build layouts like strategy_monitoring / risk_center in a type-safe way.

NOTE:
The actual React/TypeScript components live in the frontend codebase
(e.g. `dashboard/components/index.ts`). This Python module does NOT
implement UI components; it only provides a stable registry and helpers
for backend usage (FastAPI routes, templates, config schemas, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Core type aliases (Python view of TS types)
# ---------------------------------------------------------------------------

ComponentName = Literal[
    # UI primitives
    "Button",
    "Input",
    "Card",
    "Badge",
    # Cards
    "StrategyCard",
    "RiskCard",
    "PerformanceCard",
    "SignalCard",
    # Charts
    "CandlestickChart",
    "EquityCurveChart",
    "HeatmapChart",
    "PerformanceScatter",
    # Tables
    "TradesTable",
    "PositionsTable",
    "StrategiesTable",
    "PerformanceTable",
    # Modals / forms
    "StrategyConfigModal",
    "TradeConfirmationModal",
    "RiskSettingsForm",
    # Layout
    "DashboardGrid",
    "SidebarLayout",
    "HeaderBar",
]

ComponentCategory = Literal[
    "ui",
    "cards",
    "charts",
    "tables",
    "modals",
    "forms",
    "layout",
]

ComponentSize = Literal["xs", "sm", "md", "lg", "xl"]


@dataclass(frozen=True)
class ComponentInfo:
    """
    Metadata for a single frontend component.

    This mirrors the TS-side `ComponentInfo` interface and is safe to use
    in Python for config, layout definitions, and documentation.
    """

    name: ComponentName
    category: ComponentCategory
    size: ComponentSize
    lazy: bool
    bundle_size: str | None = None


class LayoutRow(TypedDict):
    """
    Layout row description.

    A row is represented as a list of component names (one or many per row).
    """

    components: List[ComponentName]


# ---------------------------------------------------------------------------
# Component metadata registry
# ---------------------------------------------------------------------------

# Bundle-size hints (for docs / logging only).
_COMPONENT_SIZES: Dict[ComponentName, str] = {
    "Button": "1.2kb",
    "Input": "1.4kb",
    "Card": "1.8kb",
    "Badge": "0.9kb",
    "StrategyCard": "8.4kb",
    "RiskCard": "7.9kb",
    "PerformanceCard": "7.6kb",
    "SignalCard": "6.2kb",
    "CandlestickChart": "245kb (lazy)",
    "EquityCurveChart": "210kb (lazy)",
    "HeatmapChart": "230kb (lazy)",
    "PerformanceScatter": "190kb (lazy)",
    "TradesTable": "189kb (lazy)",
    "PositionsTable": "175kb (lazy)",
    "StrategiesTable": "160kb (lazy)",
    "PerformanceTable": "165kb (lazy)",
    "StrategyConfigModal": "22kb (lazy)",
    "TradeConfirmationModal": "18kb (lazy)",
    "RiskSettingsForm": "14kb (lazy)",
    "DashboardGrid": "5.5kb",
    "SidebarLayout": "6.0kb",
    "HeaderBar": "4.2kb",
}


# Canonical component list with metadata.
COMPONENT_LIST: List[ComponentInfo] = [
    # UI
    ComponentInfo("Button", "ui", "sm", False, _COMPONENT_SIZES["Button"]),
    ComponentInfo("Input", "ui", "sm", False, _COMPONENT_SIZES["Input"]),
    ComponentInfo("Card", "ui", "md", False, _COMPONENT_SIZES["Card"]),
    ComponentInfo("Badge", "ui", "xs", False, _COMPONENT_SIZES["Badge"]),
    # Cards
    ComponentInfo(
        "StrategyCard", "cards", "md", False, _COMPONENT_SIZES["StrategyCard"]
    ),
    ComponentInfo("RiskCard", "cards", "md", False, _COMPONENT_SIZES["RiskCard"]),
    ComponentInfo(
        "PerformanceCard",
        "cards",
        "md",
        False,
        _COMPONENT_SIZES["PerformanceCard"],
    ),
    ComponentInfo("SignalCard", "cards", "sm", False, _COMPONENT_SIZES["SignalCard"]),
    # Charts
    ComponentInfo(
        "CandlestickChart",
        "charts",
        "xl",
        True,
        _COMPONENT_SIZES["CandlestickChart"],
    ),
    ComponentInfo(
        "EquityCurveChart",
        "charts",
        "lg",
        True,
        _COMPONENT_SIZES["EquityCurveChart"],
    ),
    ComponentInfo(
        "HeatmapChart", "charts", "lg", True, _COMPONENT_SIZES["HeatmapChart"]
    ),
    ComponentInfo(
        "PerformanceScatter",
        "charts",
        "lg",
        True,
        _COMPONENT_SIZES["PerformanceScatter"],
    ),
    # Tables
    ComponentInfo("TradesTable", "tables", "xl", True, _COMPONENT_SIZES["TradesTable"]),
    ComponentInfo(
        "PositionsTable",
        "tables",
        "xl",
        True,
        _COMPONENT_SIZES["PositionsTable"],
    ),
    ComponentInfo(
        "StrategiesTable",
        "tables",
        "xl",
        True,
        _COMPONENT_SIZES["StrategiesTable"],
    ),
    ComponentInfo(
        "PerformanceTable",
        "tables",
        "xl",
        True,
        _COMPONENT_SIZES["PerformanceTable"],
    ),
    # Modals / forms
    ComponentInfo(
        "StrategyConfigModal",
        "modals",
        "md",
        True,
        _COMPONENT_SIZES["StrategyConfigModal"],
    ),
    ComponentInfo(
        "TradeConfirmationModal",
        "modals",
        "md",
        True,
        _COMPONENT_SIZES["TradeConfirmationModal"],
    ),
    ComponentInfo(
        "RiskSettingsForm",
        "forms",
        "md",
        True,
        _COMPONENT_SIZES["RiskSettingsForm"],
    ),
    # Layout
    ComponentInfo(
        "DashboardGrid", "layout", "xl", False, _COMPONENT_SIZES["DashboardGrid"]
    ),
    ComponentInfo(
        "SidebarLayout", "layout", "lg", False, _COMPONENT_SIZES["SidebarLayout"]
    ),
    ComponentInfo("HeaderBar", "layout", "md", False, _COMPONENT_SIZES["HeaderBar"]),
]


# Quick lookup by name.
_COMPONENT_MAP: Dict[ComponentName, ComponentInfo] = {
    c.name: c for c in COMPONENT_LIST
}


def get_component_info(name: str) -> ComponentInfo | None:
    """
    Look up metadata for a component by name.

    Args:
        name: Component name exactly as exported in the frontend
              (e.g. 'StrategyCard', 'CandlestickChart').

    Returns:
        ComponentInfo instance if the component is known; otherwise None.
    """
    try:
        return _COMPONENT_MAP[name]  # type: ignore[index]
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Pre-built dashboard layouts (Python view)
# ---------------------------------------------------------------------------

DASHBOARD_LAYOUTS: Dict[str, List[LayoutRow]] = {
    "strategy_monitoring": [
        {"components": ["HeaderBar"]},
        {"components": ["StrategyCard", "PerformanceCard"]},
        {"components": ["RiskCard", "SignalCard"]},
        {"components": ["EquityCurveChart"]},
    ],
    "risk_center": [
        {"components": ["RiskCard"]},
        {"components": ["HeatmapChart", "PositionsTable"]},
        {"components": ["PerformanceScatter"]},
    ],
}


def create_dashboard_grid(layout_name: str) -> List[LayoutRow]:
    """
    Return a normalised grid layout for the given layout name.

    This is intended for backend layout-aware APIs that need to know
    which components the frontend will render.

    Args:
        layout_name: One of the keys in `DASHBOARD_LAYOUTS`.

    Raises:
        KeyError: If the layout name is not registered.
    """
    if layout_name not in DASHBOARD_LAYOUTS:
        raise KeyError(f"Unknown dashboard layout: {layout_name!r}")
    # Return a deep copy so callers cannot mutate the registry in-place.
    return [
        {"components": list(row["components"])}
        for row in DASHBOARD_LAYOUTS[layout_name]
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "ComponentName",
    "ComponentCategory",
    "ComponentSize",
    "ComponentInfo",
    "LayoutRow",
    "COMPONENT_LIST",
    "DASHBOARD_LAYOUTS",
    "get_component_info",
    "create_dashboard_grid",
]
