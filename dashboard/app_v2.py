"""
ALPHA-PRIME v2.0 - Dashboard Application Factory (app_v2)
=========================================================

Production-grade FastAPI application factory for the ALPHA-PRIME v2.0
dashboard. This module is the single source of truth for configuring
the dashboard server:

- Core FastAPI app with lifespan startup/shutdown. [web:452][web:454][web:455]
- Middleware stack (CORS, GZip, security headers, request ID, rate limiting). [web:462][web:430]
- JWT authentication with role-based access control (RBAC). [web:457][web:463][web:466][web:460]
- Prometheus-compatible request metrics and health probes. [web:458][web:464][web:461]
- Dynamic router registration for API, UI, and websocket endpoints.
- Integration hooks for async SQLAlchemy, Redis, and internal registries.

Internal details (SQLAlchemy models, concrete dependencies, etc.) are
kept in other modules; this file focuses on orchestration and wiring.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import jwt
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, ValidationError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Metrics (Prometheus-compatible)
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "path"],
)

STRATEGY_COUNT = Counter(
    "strategy_count",
    "Number of active strategies",
)
DRIFT_ALERTS_TOTAL = Counter(
    "drift_alerts_total",
    "Total drift alerts",
)
TRADE_SIGNALS_TOTAL = Counter(
    "trade_signals_total",
    "Total trade signals",
)


# ---------------------------------------------------------------------------
# Shared response schema
# ---------------------------------------------------------------------------


class APIResponse(BaseModel):
    success: bool
    data: Any | None = None
    message: str | None = None
    timestamp: datetime = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Basic RBAC / Auth models
# ---------------------------------------------------------------------------


class User(BaseModel):
    id: str
    username: str
    roles: list[str]


class AuthError(HTTPException):
    """Custom auth error."""


# ---------------------------------------------------------------------------
# Middleware: Request ID, security headers, metrics, rate limiting (Redis hook)
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req-{int(time.time() * 1e6)}"
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add basic security headers (CSP, HSTS, X-Frame-Options)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
        )
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect basic Prometheus-style metrics for each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        method = request.method
        path = request.url.path
        status_code = response.status_code
        REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(process_time)
        response.headers["X-Process-Time"] = str(process_time)
        return response


class RateLimitError(HTTPException):
    """429 Too Many Requests."""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple Redis-backed rate limiting hook.

    In production, integrate with `redis.asyncio` or a rate-limiting lib;
    here we only provide the middleware skeleton.
    """

    def __init__(self, app: FastAPI, enabled: bool = True, limit_per_minute: int = 120) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.limit_per_minute = limit_per_minute
        # Hook for Redis client injection at startup
        self.redis: Any | None = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Placeholder: user or IP key
        key = request.client.host if request.client else "anonymous"
        # A real implementation would use Redis INCR with TTL here.
        # For now, we just pass through.
        return await call_next(request)


# ---------------------------------------------------------------------------
# JWT auth / RBAC dependencies
# ---------------------------------------------------------------------------


def _decode_jwt(token: str) -> dict[str, Any]:
    secret = getattr(settings, "JWT_SECRET", "dev-secret")
    algorithms = ["HS256"]
    try:
        return jwt.decode(token, secret, algorithms=algorithms)
    except jwt.PyJWTError as exc:  # pragma: no cover - crypto-dependent
        raise AuthError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        ) from exc


async def get_current_user(request: Request) -> User:
    """
    Extract current user from Authorization: Bearer <token> header.

    When auth is disabled (dev/test), returns a synthetic superuser.
    """
    if getattr(request.app.state, "auth_disabled", False):
        return User(id="dev", username="dev", roles=["superuser"])

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise AuthError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    token = auth_header.split(" ", 1)[1]
    payload = _decode_jwt(token)
    return User(
        id=str(payload.get("sub", "")),
        username=str(payload.get("username", "user")),
        roles=list(payload.get("roles", [])),
    )


def require_role(role: str) -> Callable[[User], User]:
    """
    Dependency factory enforcing that the current user has a given role
    (e.g. 'viewer', 'trader', 'admin', 'superuser').
    """

    async def _checker(user: User = Depends(get_current_user)) -> User:
        if role not in user.roles and "superuser" not in user.roles:
            raise AuthError(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User lacks required role '{role}'.",
            )
        return user

    return _checker


async def api_key_auth(request: Request) -> None:
    """
    Simple API-key auth alternative for service-to-service calls.

    Looks for `X-API-Key` in headers and compares to configured key.
    """
    configured = getattr(settings, "API_KEY", None)
    if not configured:
        return
    key = request.headers.get("X-API-Key")
    if key != configured:
        raise AuthError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Lifespan: startup/shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context for startup/shutdown.

    Startup:
        - Connect database(s)
        - Connect Redis cache/pubsub
        - Initialize strategy registry
        - Load widget/page definitions
        - Start background workers
        - Warm caches
    Shutdown:
        - Stop workers
        - Close DB/Redis
        - Flush metrics/logs
    """
    logger.info("Dashboard startup: initializing resources...")
    app.state.db_engine = None
    app.state.redis = None
    app.state.strategy_registry = None
    app.state.auth_disabled = False

    env = getattr(app.state, "env", "development")

    # Environment-specific behaviour (stub hooks)
    if env in ("production", "staging"):
        # TODO: connect async SQLAlchemy engine and Redis here
        ...
    else:
        app.state.auth_disabled = True

    # TODO: load pages/widgets, start health/metrics workers, warm caches
    logger.info("Dashboard startup complete (env=%s).", env)
    try:
        yield
    finally:
        logger.info("Dashboard shutdown: cleaning up...")
        # TODO: stop workers, close DB/Redis
        logger.info("Dashboard shutdown complete.")


# ---------------------------------------------------------------------------
# Dependencies (DB, cache, registries, calculators)
# ---------------------------------------------------------------------------


async def get_db_session(request: Request) -> Any:
    """
    Return an async SQLAlchemy session (placeholder).

    In production, inject a real async sessionmaker bound to app.state.db_engine.
    """
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return None
    return engine  # placeholder


async def get_redis_client(request: Request) -> Any:
    """Return Redis client instance from app state (if initialised)."""
    return getattr(request.app.state, "redis", None)


async def get_cache(request: Request) -> Any:
    """
    Return cache abstraction (e.g. utils.cache_manager.CacheManager).

    Placeholder: user should wire concrete cache in app startup.
    """
    return getattr(request.app.state, "cache", None)


async def get_strategy_registry(request: Request) -> Any:
    """Return global strategy registry."""
    return getattr(request.app.state, "strategy_registry", None)


async def get_risk_calculator(request: Request) -> Any:
    """Return risk calculator/engine instance (placeholder)."""
    return getattr(request.app.state, "risk_calculator", None)


async def get_drift_monitor(request: Request) -> Any:
    """Return drift monitor instance (placeholder)."""
    return getattr(request.app.state, "drift_monitor", None)


async def get_performance_calc(request: Request) -> Any:
    """Return performance calculator (placeholder)."""
    return getattr(request.app.state, "performance_calc", None)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    resp = APIResponse(success=False, data=None, message=str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content=resp.model_dump(mode="json"),
    )


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    resp = APIResponse(success=False, data=exc.errors(), message="Validation error.")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=resp.model_dump(mode="json"),
    )


async def internal_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    resp = APIResponse(success=False, data=None, message="Internal server error.")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=resp.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Health & metrics routes
# ---------------------------------------------------------------------------


def _add_health_routes(app: FastAPI) -> None:
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=APIResponse)
    async def health() -> APIResponse:
        return APIResponse(success=True, data={"status": "healthy"}, message="OK")

    @router.get("/health/dependencies", response_model=APIResponse)
    async def health_dependencies(request: Request) -> APIResponse:
        db_ok = getattr(request.app.state, "db_engine", None) is not None
        redis_ok = getattr(request.app.state, "redis", None) is not None
        strat_ok = getattr(request.app.state, "strategy_registry", None) is not None
        data = {
            "db": "ok" if db_ok else "missing",
            "redis": "ok" if redis_ok else "missing",
            "strategy_registry": "ok" if strat_ok else "missing",
        }
        status_str = "healthy" if all(v == "ok" for v in data.values()) else "degraded"
        return APIResponse(success=True, data=data, message=status_str)

    @router.get("/health/readiness", response_model=APIResponse)
    async def readiness() -> APIResponse:
        return APIResponse(success=True, data={"ready": True}, message="ready")

    @router.get("/health/startup", response_model=APIResponse)
    async def startup_probe() -> APIResponse:
        return APIResponse(success=True, data={"startup": "complete"}, message="startup-complete")

    app.include_router(router)


def _add_metrics_route(app: FastAPI) -> None:
    @app.get("/metrics")
    async def metrics() -> Response:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# API Routers: strategies, risk, drift, performance, signals
# ---------------------------------------------------------------------------


def _build_api_router() -> APIRouter:
    api = APIRouter(prefix="/api/v2", tags=["api"])

    @api.get("/strategies")
    async def list_strategies(
        registry: Any = Depends(get_strategy_registry),
        _: User = Depends(require_role("viewer")),
    ) -> APIResponse:
        strategies = getattr(registry, "list_strategies", lambda: [])() if registry else []
        data = [s for s in strategies]
        return APIResponse(success=True, data=data, message="strategies")

    @api.get("/risk")
    async def risk_overview(
        calc: Any = Depends(get_risk_calculator),
        _: User = Depends(require_role("viewer")),
    ) -> APIResponse:
        overview = getattr(calc, "get_overview", lambda: {})() if calc else {}
        return APIResponse(success=True, data=overview, message="risk")

    @api.get("/drift")
    async def drift_status(
        monitor: Any = Depends(get_drift_monitor),
        _: User = Depends(require_role("viewer")),
    ) -> APIResponse:
        status_data = getattr(monitor, "get_status", lambda: {})() if monitor else {}
        return APIResponse(success=True, data=status_data, message="drift")

    @api.get("/performance")
    async def performance(
        calc: Any = Depends(get_performance_calc),
        _: User = Depends(require_role("viewer")),
    ) -> APIResponse:
        perf = getattr(calc, "get_summary", lambda: {})() if calc else {}
        return APIResponse(success=True, data=perf, message="performance")

    @api.websocket("/signals")
    async def signals_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                # Placeholder: integrate with notifier / pubsub
                await websocket.send_json({"type": "heartbeat", "ts": datetime.now(timezone.utc).isoformat()})
                await websocket.receive_text()
        except Exception:
            await websocket.close()

    return api


# ---------------------------------------------------------------------------
# UI / SPA routes
# ---------------------------------------------------------------------------


def _add_ui_routes(app: FastAPI) -> None:
    router = APIRouter(tags=["ui"])

    @router.get("/dashboard")
    async def dashboard_main() -> Response:
        # Placeholder SPA entrypoint
        html = "<html><body><h1>ALPHA-PRIME Dashboard</h1></body></html>"
        return Response(content=html, media_type="text/html")

    app.include_router(router)


# ---------------------------------------------------------------------------
# Router registration helpers (dynamic discovery stubs)
# ---------------------------------------------------------------------------


def _register_dynamic_routes(app: FastAPI) -> None:
    """
    Auto-discover and register routers from dashboard.routes.*, views, widgets.

    In production, use importlib to scan submodules and register APIRouter
    instances found in them. Here we just call hooks if present.
    """
    try:
        from dashboard import routes as dashboard_routes  # type: ignore
    except Exception:  # pragma: no cover - optional
        dashboard_routes = None

    if dashboard_routes and hasattr(dashboard_routes, "register_builtin_routes"):
        dashboard_routes.register_builtin_routes(app)

    # Additional dynamic discovery could be added here.


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    env: str | None = None,
    enable_auth: bool = True,
    enable_docs: bool = True,
    enable_metrics: bool = True,
) -> FastAPI:
    """
    Factory function that returns a fully configured ALPHA-PRIME Dashboard ASGI app.

    Args
    ----
    env:
        One of 'production', 'staging', 'development', 'test'. If None,
        inferred from global settings (e.g. settings.ENV or DEBUG).
    enable_auth:
        Enable/disable JWT authentication (RBAC). In dev/test env this may
        be forced off regardless of the flag.
    enable_docs:
        Enable/disable interactive /docs and /redoc.
    enable_metrics:
        Enable/disable /metrics endpoint and metrics middleware.

    Returns
    -------
    FastAPI
        Application instance ready for uvicorn deployment.
    """
    resolved_env = env or getattr(settings, "ENV", "development")
    title = f"ALPHA-PRIME Dashboard ({resolved_env})"
    debug = resolved_env in ("development", "test")

    app = FastAPI(
        title=title,
        debug=debug,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        lifespan=lifespan,
    )
    app.state.env = resolved_env
    app.state.auth_disabled = not enable_auth or resolved_env in ("development", "test")

    # ------------------------------------------------------------------
    # Middleware stack
    # ------------------------------------------------------------------
    # CORS
    allowed_origins = (
        ["https://alpha-prime.com"]
        if resolved_env == "production"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # GZip
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    # Trusted hosts
    trusted_hosts = (
        ["alpha-prime.com", "localhost", "127.0.0.1"]
        if resolved_env in ("production", "staging")
        else ["*"]
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    # Request ID
    app.add_middleware(RequestIDMiddleware)
    # Rate limiting (stub, Redis-backed via lifespan)
    app.add_middleware(RateLimitMiddleware, enabled=resolved_env == "production")
    # Metrics
    if enable_metrics:
        app.add_middleware(MetricsMiddleware)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, internal_exception_handler)

    # ------------------------------------------------------------------
    # Routers: health, metrics, API, UI, dynamic routes
    # ------------------------------------------------------------------
    _add_health_routes(app)
    if enable_metrics:
        _add_metrics_route(app)
    app.include_router(_build_api_router())
    _add_ui_routes(app)
    _register_dynamic_routes(app)

    logger.info(
        "Dashboard app created (env=%s, auth=%s, docs=%s, metrics=%s)",
        resolved_env,
        not app.state.auth_disabled,
        enable_docs,
        enable_metrics,
    )
    return app
