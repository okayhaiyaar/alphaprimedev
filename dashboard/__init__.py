"""
ALPHA-PRIME v2.0 - Dashboard Public API
=======================================

This module defines the **official public API surface** for the
`dashboard` package used in ALPHA-PRIME v2.0.

The dashboard provides strategy, risk, drift, and performance monitoring
via FastAPI/Starlette or similar ASGI frontends. It exposes a clean set
of helpers to create the dashboard app, register pages/widgets, and
query status/metrics, while **hiding internal layout and wiring**.

Internal submodules such as `dashboard.app`, `dashboard.routes`,
`dashboard.views`, `dashboard.widgets`, `dashboard.schemas`,
`dashboard.state`, and `dashboard.dependencies` are considered
**implementation details** and may change without notice. Consumers
should import from `dashboard` (this module) rather than from those
internal modules directly.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from config import get_settings as _get_global_settings
from config import get_logger

__version__ = "2.0.0"
PACKAGE_NAME = "ALPHA-PRIME Dashboard"

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy import infrastructure
# ---------------------------------------------------------------------------

_lazy_modules: Dict[str, ModuleType] = {}
_lazy_factories: Dict[str, Callable[[], ModuleType]] = {}


def _lazy_import(module_path: str) -> ModuleType:
    """
    Import a module lazily, caching the result.

    The import is deferred until this function is first called, and any
    ImportError is wrapped into a developer-friendly RuntimeError that
    explains which optional dashboard component is missing.
    """
    if module_path in _lazy_modules:
        return _lazy_modules[module_path]

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Optional dashboard module '{module_path}' is not available. "
            f"Ensure the corresponding package or extra is installed and "
            f"that your PYTHONPATH is correctly configured."
        ) from exc

    _lazy_modules[module_path] = module
    return module


# ---------------------------------------------------------------------------
# Settings facade
# ---------------------------------------------------------------------------


def get_dashboard_settings() -> Any:
    """
    Return the global dashboard-related settings object.

    This is a thin facade over the project's central configuration
    (typically `config.get_settings()` or `dashboard.state.get_settings()`).
    """
    return _get_global_settings()


# ---------------------------------------------------------------------------
# App/bootstrap helpers
# ---------------------------------------------------------------------------


def create_dashboard_app(
    env: str | None = None,
    enable_auth: bool = True,
    enable_docs: bool = True,
) -> Any:
    """
    Create and return a fully configured ASGI dashboard application.

    This is the main entrypoint for running the ALPHA-PRIME dashboard and is
    typically used by ASGI servers such as `uvicorn` or `hypercorn`.

    Parameters
    ----------
    env:
        Optional environment label (e.g. 'dev', 'prod'); forwarded to
        `dashboard.app.create_app`.
    enable_auth:
        Whether to enable authentication middleware.
    enable_docs:
        Whether to expose interactive API docs (e.g. Swagger/Redoc).

    Returns
    -------
    Any
        An ASGI app instance (e.g. FastAPI or Starlette application).
    """
    app_mod = _lazy_import("dashboard.app")
    if not hasattr(app_mod, "create_app"):
        raise RuntimeError("Module 'dashboard.app' is missing 'create_app' factory.")
    return app_mod.create_app(env=env, enable_auth=enable_auth, enable_docs=enable_docs)


def register_builtin_routes(app: Any) -> None:
    """
    Register all built-in dashboard routes on the given ASGI app.

    This typically includes health checks, metrics endpoints, UI pages,
    websocket feeds, and other standard dashboard routes.
    """
    routes_mod = _lazy_import("dashboard.routes")
    if not hasattr(routes_mod, "register_builtin_routes"):
        raise RuntimeError(
            "Module 'dashboard.routes' is missing 'register_builtin_routes(app)'"
        )
    routes_mod.register_builtin_routes(app)


# ---------------------------------------------------------------------------
# Page/view registration facade
# ---------------------------------------------------------------------------


def register_page(*args: Any, **kwargs: Any) -> Any:
    """
    Register a dashboard page/view with the internal registry.

    This is a lightweight proxy that delegates to the actual registration
    logic in `dashboard.views` or `dashboard.state`, depending on how the
    project is structured.
    """
    try:
        mod = _lazy_import("dashboard.views")
        if hasattr(mod, "register_page"):
            return mod.register_page(*args, **kwargs)
    except RuntimeError:
        # Fallback to dashboard.state if views not present
        state_mod = _lazy_import("dashboard.state")
        if hasattr(state_mod, "register_page"):
            return state_mod.register_page(*args, **kwargs)
        raise
    raise RuntimeError("No 'register_page' implementation found in dashboard.views/state.")


def get_registered_pages() -> Sequence[Any]:
    """
    Return a sequence of all registered dashboard pages.

    The concrete page type depends on the implementation in `dashboard.views`
    or `dashboard.state` (usually a Pydantic model or dataclass).
    """
    for module_name in ("dashboard.views", "dashboard.state"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover - missing module
            continue
        getter = getattr(mod, "get_registered_pages", None)
        if callable(getter):
            return getter()
    raise RuntimeError("No 'get_registered_pages' implementation found in dashboard.*.")


def get_page_by_id(page_id: str) -> Any:
    """
    Retrieve a registered page definition by its identifier.

    Parameters
    ----------
    page_id:
        Unique page identifier string.
    """
    for module_name in ("dashboard.views", "dashboard.state"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover - missing module
            continue
        getter = getattr(mod, "get_page_by_id", None)
        if callable(getter):
            return getter(page_id)
    raise RuntimeError("No 'get_page_by_id' implementation found in dashboard.*.")


# ---------------------------------------------------------------------------
# Widgets & tiles facade
# ---------------------------------------------------------------------------


def register_widget(*args: Any, **kwargs: Any) -> Any:
    """
    Register a dashboard widget/tile with the internal registry.

    Delegates to `dashboard.widgets.register_widget` or
    `dashboard.state.register_widget`.
    """
    try:
        mod = _lazy_import("dashboard.widgets")
        if hasattr(mod, "register_widget"):
            return mod.register_widget(*args, **kwargs)
    except RuntimeError:
        state_mod = _lazy_import("dashboard.state")
        if hasattr(state_mod, "register_widget"):
            return state_mod.register_widget(*args, **kwargs)
        raise
    raise RuntimeError("No 'register_widget' implementation found in dashboard.widgets/state.")


def get_registered_widgets() -> Sequence[Any]:
    """
    Return a sequence of all registered widgets/tiles.
    """
    for module_name in ("dashboard.widgets", "dashboard.state"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover
            continue
        getter = getattr(mod, "get_registered_widgets", None)
        if callable(getter):
            return getter()
    raise RuntimeError("No 'get_registered_widgets' implementation found in dashboard.*.")


def get_widget_by_id(widget_id: str) -> Any:
    """
    Retrieve a registered widget by its identifier.

    Parameters
    ----------
    widget_id:
        Unique widget identifier string.
    """
    for module_name in ("dashboard.widgets", "dashboard.state"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover
            continue
        getter = getattr(mod, "get_widget_by_id", None)
        if callable(getter):
            return getter(widget_id)
    raise RuntimeError("No 'get_widget_by_id' implementation found in dashboard.*.")


# ---------------------------------------------------------------------------
# Metrics & health facade
# ---------------------------------------------------------------------------


def get_dashboard_status() -> Any:
    """
    Return a high-level status object for the dashboard.

    This may include uptime, last refresh, error counts, and other
    operational health indicators.
    """
    for module_name in ("dashboard.state", "dashboard.metrics"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover
            continue
        fn = getattr(mod, "get_dashboard_status", None)
        if callable(fn):
            return fn()
    raise RuntimeError("No 'get_dashboard_status' implementation found in dashboard.*.")


def get_active_strategies_summary() -> Any:
    """
    Return a summary of active strategies currently displayed/tracked
    on the dashboard (PnL, risk, health flags, etc.).
    """
    for module_name in ("dashboard.state", "dashboard.metrics"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover
            continue
        fn = getattr(mod, "get_active_strategies_summary", None)
        if callable(fn):
            return fn()
    raise RuntimeError(
        "No 'get_active_strategies_summary' implementation found in dashboard.*."
    )


def get_risk_overview() -> Any:
    """
    Return an aggregate risk overview object.

    Typically includes portfolio exposures, drawdowns, VaR metrics,
    and any outstanding risk alerts.
    """
    for module_name in ("dashboard.state", "dashboard.metrics"):
        try:
            mod = _lazy_import(module_name)
        except RuntimeError:  # pragma: no cover
            continue
        fn = getattr(mod, "get_risk_overview", None)
        if callable(fn):
            return fn()
    raise RuntimeError("No 'get_risk_overview' implementation found in dashboard.*.")


# ---------------------------------------------------------------------------
# Lazy class re-exports (widgets & schemas)
# ---------------------------------------------------------------------------

# We bind these names on first access, so that `from dashboard import MetricCard`
# works while still deferring imports until used.


def _load_widget_class(name: str) -> type:
    widgets_mod = _lazy_import("dashboard.widgets")
    if not hasattr(widgets_mod, name):
        raise RuntimeError(f"'dashboard.widgets' does not define '{name}'.")
    cls = getattr(widgets_mod, name)
    globals()[name] = cls  # cache at module level
    return cls


def _load_schema_class(name: str) -> type:
    schemas_mod = _lazy_import("dashboard.schemas")
    if not hasattr(schemas_mod, name):
        raise RuntimeError(f"'dashboard.schemas' does not define '{name}'.")
    cls = getattr(schemas_mod, name)
    globals()[name] = cls  # cache at module level
    return cls


class _WidgetSchemaProxy:
    """Descriptor-style proxy for lazily loaded widget/schema classes."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple proxy
        if item in ("MetricCard", "TimeSeriesChart", "TableWidget", "HeatmapWidget"):
            return _load_widget_class(item)
        if item in ("DashboardStatus", "StrategySummary", "RiskOverview"):
            return _load_schema_class(item)
        raise AttributeError(item)


# Allow attribute-style access for advanced use if desired:
proxy = _WidgetSchemaProxy()


def __getattr__(name: str) -> Any:  # pragma: no cover - Python-level hook
    """
    Support `from dashboard import MetricCard` with lazy imports.

    This hook is invoked only if the attribute is not found in globals().
    """
    if name in ("MetricCard", "TimeSeriesChart", "TableWidget", "HeatmapWidget"):
        return _load_widget_class(name)
    if name in ("DashboardStatus", "StrategySummary", "RiskOverview"):
        return _load_schema_class(name)
    raise AttributeError(f"module 'dashboard' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Developer helpers
# ---------------------------------------------------------------------------


def list_public_api() -> List[str]:
    """
    Return a sorted list of symbols that constitute the public API.

    This is a convenience helper for introspection and documentation
    generation; it does not trigger any heavy imports.
    """
    return sorted(__all__)


def print_quickstart() -> None:
    """
    Print a short quickstart guide for using the dashboard package.

    This function is side-effect free beyond printing to stdout and does
    not import any heavy dependencies.
    """
    guide = f"""
{PACKAGE_NAME} Quickstart
=========================

1. Create an ASGI app:

   from dashboard import create_dashboard_app

   app = create_dashboard_app(env="dev", enable_auth=True, enable_docs=True)

2. Register builtin routes:

   from dashboard import register_builtin_routes

   register_builtin_routes(app)

3. Register a page and widget:

   from dashboard import register_page, register_widget, MetricCard

   register_page(id="overview", title="Strategy Overview")
   register_widget(MetricCard(id="pnl_card", title="PnL", value=0.0))

4. Run with uvicorn:

   uvicorn dashboard_app:app --reload --port 8000

Only the symbols listed in `dashboard.__all__` are considered stable.
"""
    print(guide.strip())


# ---------------------------------------------------------------------------
# Explicit public API
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Core app/bootstrap
    "create_dashboard_app",
    "get_dashboard_settings",
    "register_builtin_routes",
    # Pages
    "register_page",
    "get_registered_pages",
    "get_page_by_id",
    # Widgets
    "register_widget",
    "get_registered_widgets",
    "get_widget_by_id",
    # Metrics/health
    "get_dashboard_status",
    "get_active_strategies_summary",
    "get_risk_overview",
    # Widget classes (lazy)
    "MetricCard",
    "TimeSeriesChart",
    "TableWidget",
    "HeatmapWidget",
    # Schema classes (lazy)
    "DashboardStatus",
    "StrategySummary",
    "RiskOverview",
    # Metadata & helpers
    "__version__",
    "PACKAGE_NAME",
    "list_public_api",
    "print_quickstart",
]
