"""
ALPHA-PRIME v2.0 - Financial Visualization Components (Python Facade)
=====================================================================

Python-side model of the financial chart suite implemented on the
frontend (React 18 + TypeScript).

The real charts (Recharts, TradingView Lightweight Charts, Framer Motion,
Tailwind/shadcn, TanStack Query, React Aria) live in the TS/TSX layer,
e.g. `dashboard/components/charts.tsx`.

This module provides:
- Stable chart component names.
- Data/props types used by the backend (for schemas and config).
- Metadata about performance, theming, responsiveness, and interactions.

No React/TS code is included here so that Python tooling (Pylance, mypy)
remains error free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, TypedDict


# ---------------------------------------------------------------------------
# Core data types (Python view of TS interfaces)
# ---------------------------------------------------------------------------

Theme = Literal["light", "dark", "auto"]

Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d", "1w"]


@dataclass(frozen=True)
class OHLCVBar:
    """Python equivalent of the TS OHLCVBar interface."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    change: float | None = None


class ChartProps(TypedDict, total=False):
    """
    Universal chart props as seen from backend side.

    The frontend TS components can mirror this shape.
    """

    data: List[Dict[str, Any]]
    height: int | str
    theme: Theme
    loading: bool
    error: str | None
    className: str
    realtime: bool


# ---------------------------------------------------------------------------
# Component names & kinds
# ---------------------------------------------------------------------------

ChartComponentName = Literal[
    "CandlestickChart",
    "EquityCurveChart",
    "PerformanceScatter",
    "CorrelationHeatmap",
    "DrawdownWaterfall",
    "SignalTimeline",
    "RiskReturnScatter",
    "DriftHeatmap",
    "RollingMetrics",
]

ChartKind = Literal[
    "candlestick",
    "line",
    "area",
    "scatter",
    "heatmap",
    "waterfall",
    "timeline",
    "composite",
]


# ---------------------------------------------------------------------------
# Metadata classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceProfile:
    """Performance-related expectations and hints."""

    max_points: int
    target_initial_render_ms: int
    supports_realtime: bool
    virtual_axes: bool
    canvas_fallback: bool


@dataclass(frozen=True)
class InteractionProfile:
    """Interaction and accessibility capabilities."""

    crosshair: bool
    zoom_pan: bool
    keyboard_pan_zoom: bool
    touch_gestures: bool
    tradingview_link: bool
    has_entry_exit_markers: bool
    has_indicator_library: bool
    aria_live_region: bool


@dataclass(frozen=True)
class ThemingProfile:
    """Theming and responsiveness information."""

    theme_modes: List[Theme]
    uses_css_variables: bool
    mobile_first: bool
    breakpoints: List[str]
    honors_reduced_motion: bool
    colorblind_friendly: bool


@dataclass(frozen=True)
class ChartComponentInfo:
    """
    High-level description of a chart component.

    This is used purely for backend config / OpenAPI docs. The frontend
    uses an equivalent TS definition for implementation.
    """

    name: ChartComponentName
    kind: ChartKind
    description: str
    usage_weight_pct: int
    default_height: int
    props_schema: Mapping[str, str]
    perf: PerformanceProfile
    interactions: InteractionProfile
    theming: ThemingProfile


# ---------------------------------------------------------------------------
# Chart registry (all 9 required components)
# ---------------------------------------------------------------------------

CHARTS: List[ChartComponentInfo] = [
    ChartComponentInfo(
        name="CandlestickChart",
        kind="candlestick",
        description=(
            "Flagship OHLCV candlestick view with TradingView Lightweight "
            "Charts, SMA/EMA/Bollinger overlays, volume profile, crosshair "
            "and OHLC tooltip, zoom/pan and indicator library."
        ),
        usage_weight_pct=40,
        default_height=400,
        props_schema={
            "symbol": "string",
            "timeframe": "Timeframe",
            "overlays": "Overlay[]",
            "indicators": "Indicator[]",
            "realtime": "boolean",
        },
        perf=PerformanceProfile(
            max_points=10_000,
            target_initial_render_ms=100,
            supports_realtime=True,
            virtual_axes=True,
            canvas_fallback=True,
        ),
        interactions=InteractionProfile(
            crosshair=True,
            zoom_pan=True,
            keyboard_pan_zoom=True,
            touch_gestures=True,
            tradingview_link=True,
            has_entry_exit_markers=True,
            has_indicator_library=True,
            aria_live_region=True,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="EquityCurveChart",
        kind="composite",
        description=(
            "Cumulative PnL vs benchmark with underwater drawdown subchart "
            "and rolling Sharpe indicator."
        ),
        usage_weight_pct=30,
        default_height=320,
        props_schema={
            "strategySeries": "EquityCurvePoint[]",
            "benchmarkSeries": "EquityCurvePoint[]",
            "showBenchmark": "boolean",
        },
        perf=PerformanceProfile(
            max_points=5_000,
            target_initial_render_ms=80,
            supports_realtime=False,
            virtual_axes=True,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=True,
            zoom_pan=True,
            keyboard_pan_zoom=True,
            touch_gestures=True,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="PerformanceScatter",
        kind="scatter",
        description=(
            "Sharpe vs Sortino scatter of strategies with regime colouring "
            "and efficient frontier overlay."
        ),
        usage_weight_pct=15,
        default_height=300,
        props_schema={"points": "PerformancePoint[]"},
        perf=PerformanceProfile(
            max_points=1_000,
            target_initial_render_ms=60,
            supports_realtime=False,
            virtual_axes=False,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=False,
            zoom_pan=False,
            keyboard_pan_zoom=False,
            touch_gestures=False,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="CorrelationHeatmap",
        kind="heatmap",
        description=(
            "Feature/strategy correlation matrix with clickable drill-down "
            "and regime-specific correlations."
        ),
        usage_weight_pct=5,
        default_height=320,
        props_schema={"cells": "CorrelationCell[]"},
        perf=PerformanceProfile(
            max_points=2_500,
            target_initial_render_ms=80,
            supports_realtime=False,
            virtual_axes=False,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=False,
            zoom_pan=False,
            keyboard_pan_zoom=False,
            touch_gestures=True,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="DrawdownWaterfall",
        kind="waterfall",
        description=(
            "Peak-to-trough-to-recovery drawdown timeline with max DD "
            "annotation, recovery factor, and time-under-water metrics."
        ),
        usage_weight_pct=5,
        default_height=280,
        props_schema={"points": "DrawdownWaterfallPoint[]"},
        perf=PerformanceProfile(
            max_points=2_000,
            target_initial_render_ms=80,
            supports_realtime=False,
            virtual_axes=True,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=True,
            zoom_pan=True,
            keyboard_pan_zoom=True,
            touch_gestures=True,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="SignalTimeline",
        kind="timeline",
        description=(
            "Signal history timeline with buy/sell markers, win/loss "
            "outcomes, confidence bands, and drawdown context shading."
        ),
        usage_weight_pct=2,
        default_height=260,
        props_schema={"events": "SignalEvent[]"},
        perf=PerformanceProfile(
            max_points=5_000,
            target_initial_render_ms=80,
            supports_realtime=True,
            virtual_axes=True,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=True,
            zoom_pan=True,
            keyboard_pan_zoom=True,
            touch_gestures=True,
            tradingview_link=False,
            has_entry_exit_markers=True,
            has_indicator_library=False,
            aria_live_region=True,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="RiskReturnScatter",
        kind="scatter",
        description=(
            "Strategy universe scatter (volatility vs return) with risk "
            "parity line and Sharpe ratio contours."
        ),
        usage_weight_pct=1,
        default_height=300,
        props_schema={"points": "RiskReturnPoint[]"},
        perf=PerformanceProfile(
            max_points=1_000,
            target_initial_render_ms=60,
            supports_realtime=False,
            virtual_axes=False,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=False,
            zoom_pan=False,
            keyboard_pan_zoom=False,
            touch_gestures=False,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="DriftHeatmap",
        kind="heatmap",
        description=(
            "PSI heatmap by feature and time, with error-ratio surface and "
            "alert thresholds for model drift."
        ),
        usage_weight_pct=1,
        default_height=280,
        props_schema={"cells": "DriftCell[]"},
        perf=PerformanceProfile(
            max_points=2_500,
            target_initial_render_ms=80,
            supports_realtime=False,
            virtual_axes=False,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=False,
            zoom_pan=False,
            keyboard_pan_zoom=False,
            touch_gestures=True,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
    ChartComponentInfo(
        name="RollingMetrics",
        kind="line",
        description=(
            "Rolling Sharpe/Sortino metrics (e.g. 252d window) with regime "
            "change markers and threshold breaches."
        ),
        usage_weight_pct=1,
        default_height=260,
        props_schema={"points": "RollingMetricPoint[]"},
        perf=PerformanceProfile(
            max_points=5_000,
            target_initial_render_ms=80,
            supports_realtime=False,
            virtual_axes=True,
            canvas_fallback=False,
        ),
        interactions=InteractionProfile(
            crosshair=True,
            zoom_pan=True,
            keyboard_pan_zoom=True,
            touch_gestures=True,
            tradingview_link=False,
            has_entry_exit_markers=False,
            has_indicator_library=False,
            aria_live_region=False,
        ),
        theming=ThemingProfile(
            theme_modes=["light", "dark", "auto"],
            uses_css_variables=True,
            mobile_first=True,
            breakpoints=["xs", "sm", "md", "lg", "xl"],
            honors_reduced_motion=True,
            colorblind_friendly=True,
        ),
    ),
]

_CHART_MAP: Dict[ChartComponentName, ChartComponentInfo] = {
    c.name: c for c in CHARTS
}


# ---------------------------------------------------------------------------
# Hook configuration facades (not actual React hooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OHLCVDataHookConfig:
    """Configuration for a frontend `useOHLCVData` hook."""

    default_timeframe: Timeframe
    websocket_channel: str
    polling_interval_ms: int


@dataclass(frozen=True)
class ThemeHookConfig:
    """Configuration for a frontend `useChartTheme` hook."""

    uses_prefers_color_scheme: bool
    css_variable_prefix: str


@dataclass(frozen=True)
class DimensionsHookConfig:
    """Configuration for a frontend `useChartDimensions` hook."""

    observes_resize: bool
    uses_container_queries: bool


USE_OHLCV_DATA_CONFIG = OHLCVDataHookConfig(
    default_timeframe="1d",
    websocket_channel="ohlcv_stream",
    polling_interval_ms=5_000,
)

USE_CHART_THEME_CONFIG = ThemeHookConfig(
    uses_prefers_color_scheme=True,
    css_variable_prefix="--chart-",
)

USE_CHART_DIMENSIONS_CONFIG = DimensionsHookConfig(
    observes_resize=True,
    uses_container_queries=True,
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def list_chart_components() -> List[ChartComponentName]:
    """Return all known chart component names."""
    return [c.name for c in CHARTS]


def get_chart_info(name: str) -> ChartComponentInfo | None:
    """
    Look up chart metadata by component name.

    Args:
        name: Component name, e.g. 'CandlestickChart'.

    Returns:
        ChartComponentInfo if known; otherwise None.
    """
    try:
        return _CHART_MAP[name]  # type: ignore[index]
    except KeyError:
        return None


__all__: List[str] = [
    "Theme",
    "Timeframe",
    "OHLCVBar",
    "ChartProps",
    "ChartComponentName",
    "ChartKind",
    "PerformanceProfile",
    "InteractionProfile",
    "ThemingProfile",
    "ChartComponentInfo",
    "CHARTS",
    "list_chart_components",
    "get_chart_info",
    "OHLCVDataHookConfig",
    "ThemeHookConfig",
    "DimensionsHookConfig",
    "USE_OHLCV_DATA_CONFIG",
    "USE_CHART_THEME_CONFIG",
    "USE_CHART_DIMENSIONS_CONFIG",
]
