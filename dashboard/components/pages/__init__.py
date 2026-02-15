"""
ALPHA-PRIME v2.0 - Dashboard Pages Registry (Python Facade)
===========================================================

Python-side registry for ALPHA-PRIME dashboard pages.

This module does NOT implement React pages or routing. Instead, it:

- Defines stable page IDs and metadata for all dashboard pages.
- Encodes categories, permissions, layouts, and performance budgets.
- Provides helpers for route generation and role-based access.
- Mirrors the intended React Router v6 + FastAPI architecture in a
  backend-friendly way.

The corresponding React/TypeScript implementation can mirror these
types and structures in `dashboard/pages/core.tsx`, `analytics.tsx`,
`operational.tsx`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict


# ---------------------------------------------------------------------------
# Type definitions (Python view of TS types)
# ---------------------------------------------------------------------------

PageId = Literal[
    "overview",
    "strategies",
    "risk",
    "signals",
    "performance",
    "drift",
    "universe",
    "trades",
    "settings",
]

PageCategory = Literal["core", "analytics", "operational", "admin"]

LayoutType = Literal["grid", "sidebar", "full", "split"]

UserRole = Literal["viewer", "trader", "admin"]


class PageMetrics(TypedDict, total=False):
    loadTimeMs: int              # LCP target
    dataSources: List[str]       # e.g. ["redis", "postgres"]
    bundleSizeKb: int            # gzipped bundle budget
    realtimeCharts: int          # number of live charts


@dataclass(frozen=True)
class PageMeta:
    """
    Metadata for a dashboard page.

    Mirrors the TS-side `PageMeta` interface, minus the Lucide icon type
    (here represented as a simple string identifier).
    """

    id: PageId
    name: str
    path: str
    description: str
    category: PageCategory
    permissions: List[str]
    icon: str                    # Lucide icon name in the frontend
    defaultLayout: LayoutType
    metrics: PageMetrics
    featured: bool


# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------

PAGES: List[PageMeta] = [
    # Core pages (always bundled)
    PageMeta(
        id="overview",
        name="Overview",
        path="/dashboard/overview",
        description="Executive summary of strategy health, PnL, and risk.",
        category="core",
        permissions=["viewer", "trader", "admin"],
        icon="LayoutDashboard",
        defaultLayout="grid",
        metrics={
            "loadTimeMs": 250,
            "dataSources": ["redis", "postgres"],
            "bundleSizeKb": 45,
            "realtimeCharts": 3,
        },
        featured=True,
    ),
    PageMeta(
        id="strategies",
        name="Strategies",
        path="/dashboard/strategies",
        description="Strategy configuration, allocations, and status.",
        category="core",
        permissions=["viewer", "trader", "admin"],
        icon="TrendingUp",
        defaultLayout="sidebar",
        metrics={
            "loadTimeMs": 300,
            "dataSources": ["postgres"],
            "bundleSizeKb": 40,
            "realtimeCharts": 1,
        },
        featured=True,
    ),
    PageMeta(
        id="risk",
        name="Risk",
        path="/dashboard/risk",
        description="Real-time risk monitoring, limits, and breaches.",
        category="core",
        permissions=["viewer", "trader", "admin"],
        icon="ShieldAlert",
        defaultLayout="grid",
        metrics={
            "loadTimeMs": 300,
            "dataSources": ["redis", "timeseries"],
            "bundleSizeKb": 42,
            "realtimeCharts": 3,
        },
        featured=True,
    ),
    PageMeta(
        id="signals",
        name="Signals",
        path="/dashboard/signals",
        description="Live trade signals, executions, and signal queue.",
        category="core",
        permissions=["trader", "admin"],
        icon="Flashlight",
        defaultLayout="split",
        metrics={
            "loadTimeMs": 280,
            "dataSources": ["redis", "kafka"],
            "bundleSizeKb": 38,
            "realtimeCharts": 2,
        },
        featured=True,
    ),
    # Analytics pages (lazy recommended)
    PageMeta(
        id="performance",
        name="Performance",
        path="/dashboard/performance",
        description="Backtest analytics, equity curves, and attribution.",
        category="analytics",
        permissions=["viewer", "trader", "admin"],
        icon="LineChart",
        defaultLayout="full",
        metrics={
            "loadTimeMs": 400,
            "dataSources": ["postgres"],
            "bundleSizeKb": 55,
            "realtimeCharts": 1,
        },
        featured=False,
    ),
    PageMeta(
        id="drift",
        name="Drift",
        path="/dashboard/drift",
        description="Model and data drift monitoring with PSI and alerts.",
        category="analytics",
        permissions=["admin"],
        icon="Activity",
        defaultLayout="grid",
        metrics={
            "loadTimeMs": 450,
            "dataSources": ["postgres", "object_store"],
            "bundleSizeKb": 52,
            "realtimeCharts": 1,
        },
        featured=False,
    ),
    PageMeta(
        id="universe",
        name="Universe",
        path="/dashboard/universe",
        description="Stock screening, factor scores, and coverage.",
        category="analytics",
        permissions=["viewer", "trader", "admin"],
        icon="Globe2",
        defaultLayout="sidebar",
        metrics={
            "loadTimeMs": 350,
            "dataSources": ["postgres"],
            "bundleSizeKb": 48,
            "realtimeCharts": 0,
        },
        featured=False,
    ),
    # Operational pages (lazy)
    PageMeta(
        id="trades",
        name="Trades",
        path="/dashboard/trades",
        description="Trade history, execution quality, and fills.",
        category="operational",
        permissions=["trader", "admin"],
        icon="ListChecks",
        defaultLayout="full",
        metrics={
            "loadTimeMs": 380,
            "dataSources": ["postgres", "timeseries"],
            "bundleSizeKb": 50,
            "realtimeCharts": 1,
        },
        featured=False,
    ),
    PageMeta(
        id="settings",
        name="Settings",
        path="/dashboard/settings",
        description="System, account, and strategy configuration.",
        category="operational",
        permissions=["admin"],
        icon="Settings",
        defaultLayout="sidebar",
        metrics={
            "loadTimeMs": 300,
            "dataSources": ["postgres"],
            "bundleSizeKb": 30,
            "realtimeCharts": 0,
        },
        featured=False,
    ),
]

_PAGE_MAP: Dict[PageId, PageMeta] = {p.id: p for p in PAGES}


# ---------------------------------------------------------------------------
# Core exports (Python-side view of TS exports)
# ---------------------------------------------------------------------------

CORE_PAGES: List[PageId] = ["overview", "strategies", "risk", "signals"]
ANALYTICS_PAGES: List[PageId] = ["performance", "drift", "universe"]
OPERATIONAL_PAGES: List[PageId] = ["trades", "settings"]


# These strings mirror TS module paths; used for FastAPI / config only.
CORE_MODULE = "dashboard.pages.core"
ANALYTICS_MODULE = "dashboard.pages.analytics"
OPERATIONAL_MODULE = "dashboard.pages.operational"


# ---------------------------------------------------------------------------
# Lazy loading descriptors (for TS layer)
# ---------------------------------------------------------------------------

class LazyPageDescriptor(TypedDict):
    pageId: PageId
    module: str
    export: str


LAZY_PAGES: Dict[PageId, LazyPageDescriptor] = {
    "performance": {
        "pageId": "performance",
        "module": "dashboard.pages.performance",
        "export": "PerformancePage",
    },
    "drift": {
        "pageId": "drift",
        "module": "dashboard.pages.drift",
        "export": "DriftPage",
    },
    "universe": {
        "pageId": "universe",
        "module": "dashboard.pages.universe",
        "export": "UniversePage",
    },
    "trades": {
        "pageId": "trades",
        "module": "dashboard.pages.trades",
        "export": "TradesPage",
    },
    "settings": {
        "pageId": "settings",
        "module": "dashboard.pages.settings",
        "export": "SettingsPage",
    },
}


# ---------------------------------------------------------------------------
# Route generation (backend-friendly shape)
# ---------------------------------------------------------------------------

class RouteConfig(TypedDict):
    path: str
    pageId: PageId
    meta: PageMeta


ROUTES: List[RouteConfig] = [
    {"path": page.path, "pageId": page.id, "meta": page} for page in PAGES
]


# ---------------------------------------------------------------------------
# Layout templates (executive / trader / quant / admin)
# ---------------------------------------------------------------------------

LAYOUT_TEMPLATES: Dict[str, List[str]] = {
    "executive": ["overview", "strategies", "risk"],
    "trader": ["signals", "trades"],
    "quant": ["performance", "drift", "universe"],
    "admin": ["overview", "risk", "settings", "drift"],
}


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------

def get_page_by_id(page_id: str) -> PageMeta | None:
    """Return PageMeta for a given page ID, or None if unknown."""
    try:
        return _PAGE_MAP[page_id]  # type: ignore[index]
    except KeyError:
        return None


def get_featured_pages() -> List[PageMeta]:
    """Return all pages flagged as featured (for homepage grid)."""
    return [p for p in PAGES if p.featured]


def get_pages_by_category(category: PageCategory) -> List[PageMeta]:
    """Return all pages matching a given category."""
    return [p for p in PAGES if p.category == category]


def get_accessible_pages(user_role: UserRole) -> List[PageMeta]:
    """
    Return pages accessible to a given user role.

    - viewer: overview, strategies, risk, performance, universe
    - trader: viewer pages + signals, trades
    - admin: all pages (including drift, settings)
    """
    pages: List[PageMeta] = []

    for page in PAGES:
        if user_role in page.permissions:
            pages.append(page)

    return pages


# ---------------------------------------------------------------------------
# Type helpers (backend view of PageComponentProps)
# ---------------------------------------------------------------------------

class PageComponentProps(TypedDict, total=False):
    pageMeta: PageMeta
    userRole: str
    realtime: bool


__all__: List[str] = [
    "PageId",
    "PageCategory",
    "LayoutType",
    "UserRole",
    "PageMetrics",
    "PageMeta",
    "PAGES",
    "CORE_PAGES",
    "ANALYTICS_PAGES",
    "OPERATIONAL_PAGES",
    "LAZY_PAGES",
    "RouteConfig",
    "ROUTES",
    "LAYOUT_TEMPLATES",
    "PageComponentProps",
    "get_page_by_id",
    "get_featured_pages",
    "get_pages_by_category",
    "get_accessible_pages",
]
