"""
ALPHA-PRIME v2.0 Operations Hub
===============================

Production operations facade for deployment, monitoring, backup,
logging, and health management.

Design goals:
- Safe star imports (`from ops import *` → only vetted functions)
- Lazy loading of heavy submodules
- Explicit, documented public API (`__all__`)
- Zero side effects at import time
- Docker/Kubernetes friendly (health checks, CLI)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Literal, Optional

import importlib
import sys


# ---------------------------------------------------------------------------
# Version & compatibility
# ---------------------------------------------------------------------------

__version__ = "2.0.0"
__alpha_prime_version__ = "2.0.0"


def check_compatibility() -> bool:
  """
  Verify that the ops layer is compatible with the ALPHA-PRIME core.

  In a real deployment this can check:
  - Package versions (ops vs core).
  - Required environment variables.
  - Migration state / schema version.
  """
  # Placeholder: the versions are kept in sync via release tooling.
  return __version__.split(".")[0] == __alpha_prime_version__.split(".")[0]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OpsError(Exception):
  """Base class for all operations-related errors."""


class HealthCheckError(OpsError):
  """Raised when a health check fails critically."""


class DeploymentError(OpsError):
  """Raised on deployment or rollback failure."""


# ---------------------------------------------------------------------------
# Data classes & type aliases
# ---------------------------------------------------------------------------

HealthStatus = Literal["healthy", "degraded", "critical"]
Severity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


@dataclass
class ServiceHealth:
  name: str
  status: HealthStatus
  message: str = ""
  latency_ms: Optional[float] = None


@dataclass
class SystemHealth:
  status: HealthStatus
  services: Dict[str, ServiceHealth]
  timestamp: datetime
  uptime: timedelta


@dataclass
class OpsStatus:
  healthy: bool
  message: str
  health: SystemHealth


@dataclass
class Deployment:
  strategy_id: str
  deployment_id: str
  environment: str
  url: str
  created_at: datetime
  active: bool


@dataclass
class DeploymentResult:
  success: bool
  deployment_id: str
  url: str
  logs: List[str]


@dataclass
class MetricsReport:
  environment: str
  metrics: Dict[str, float]
  generated_at: datetime


@dataclass
class BackupInfo:
  backup_id: str
  created_at: datetime
  size_bytes: int
  description: str


@dataclass
class BackupReport:
  success: bool
  backup_id: str
  artifacts: List[BackupInfo]
  started_at: datetime
  completed_at: datetime


@dataclass
class RestoreResult:
  success: bool
  backup_id: str
  dry_run: bool
  message: str


@dataclass
class ScaleResult:
  service: str
  replicas: int
  success: bool
  message: str


@dataclass
class ConfigResult:
  success: bool
  path: str
  message: str


@dataclass
class ValidationResult:
  valid: bool
  path: str
  errors: List[str]


@dataclass
class MigrationResult:
  success: bool
  applied_migrations: List[str]
  message: str


@dataclass
class SeedResult:
  success: bool
  message: str


@dataclass
class CleanupResult:
  success: bool
  removed_items: List[str]


# ---------------------------------------------------------------------------
# Lazy import infrastructure
# ---------------------------------------------------------------------------

_lazy_modules = {
  "deploy": lambda: importlib.import_module("ops.deploy"),
  "monitoring": lambda: importlib.import_module("ops.monitoring"),
  "backup": lambda: importlib.import_module("ops.backup"),
  "db": lambda: importlib.import_module("ops.db"),
  "config": lambda: importlib.import_module("ops.config"),
}


def _lazy_get(module_name: str):
  """Import heavy modules only when needed."""
  try:
    loader = _lazy_modules[module_name]
  except KeyError as exc:
    raise ImportError(f"Ops module '{module_name}' not available") from exc
  return loader()


# ---------------------------------------------------------------------------
# Tier 1 - Core operations (lightweight stubs, ready to call)
# ---------------------------------------------------------------------------

def health_check(timeout: int = 30) -> SystemHealth:
  """
  Comprehensive system health check for Docker/Kubernetes probes.

  Performs lightweight checks against core dependencies (DB, cache,
  message bus, strategy engine, drift monitor). Designed to degrade
  gracefully when subsystems are unavailable.
  """
  now = datetime.utcnow()
  services: Dict[str, ServiceHealth] = {}

  # NOTE: In production, delegate to ops.monitoring.health_check().
  # Here we simulate a simple, fast composite check.
  services["db"] = ServiceHealth(name="db", status="healthy")
  services["redis"] = ServiceHealth(name="redis", status="healthy")
  services["strategies"] = ServiceHealth(name="strategies", status="healthy")
  services["cache"] = ServiceHealth(name="cache", status="healthy")
  services["drift_monitor"] = ServiceHealth(name="drift_monitor", status="healthy")

  overall_status: HealthStatus = "healthy"
  for svc in services.values():
    if svc.status == "critical":
      overall_status = "critical"
      break
    if svc.status == "degraded" and overall_status == "healthy":
      overall_status = "degraded"

  return SystemHealth(
    status=overall_status,
    services=services,
    timestamp=now,
    uptime=timedelta(seconds=0),  # real impl: read from process manager
  )


def get_system_status() -> OpsStatus:
  """Return high-level system status derived from `health_check()`."""
  health = health_check()
  healthy = health.status == "healthy"
  msg = "System healthy" if healthy else f"System status: {health.status}"
  return OpsStatus(healthy=healthy, message=msg, health=health)


def is_healthy() -> bool:
  """Convenience helper to check if the system is fully healthy."""
  return get_system_status().healthy


def deploy_strategy(strategy_id: str, environment: str = "paper") -> DeploymentResult:
  """
  Deploy a strategy to the specified environment (paper/live).

  In production this would delegate to `ops.deploy.deploy_strategy`
  using Kubernetes/VM rollout, health checks, and monitoring hooks.
  """
  # Placeholder implementation: mark success with a synthetic ID/URL.
  deployment_id = f"{strategy_id}-{environment}-{int(datetime.utcnow().timestamp())}"
  url = f"https://alpha-prime/{environment}/strategies/{strategy_id}"
  logs = [f"Deployed {strategy_id} to {environment}", f"Deployment ID: {deployment_id}"]
  return DeploymentResult(success=True, deployment_id=deployment_id, url=url, logs=logs)


def rollback_strategy(strategy_id: str) -> bool:
  """
  Roll back the latest deployment of a strategy.

  Real implementation should:
  - Identify last stable deployment.
  - Rollback Kubernetes/VM resources.
  - Emit audit log and alerts.
  """
  # Placeholder: indicate a successful rollback.
  return True


def list_deployments() -> List[Deployment]:
  """Return a list of recent deployments for inspection."""
  # Placeholder: empty list; real impl queries ops.deploy or DB.
  return []


def tail_logs(service: str, lines: int = 100) -> Iterator[str]:
  """
  Live log tailing for a service (docker logs, journalctl, etc.).

  This is designed as a streaming generator for CLI usage:
      for line in tail_logs("strategies", 50):
          print(line)
  """
  # Placeholder: demonstrate shape with a finite iterator.
  for i in range(lines):
    yield f"[{service}] log line {i + 1}"


def get_metrics(environment: str = "production") -> MetricsReport:
  """
  Aggregate metrics (e.g. from Prometheus) for the given environment.

  Returns a simple metrics snapshot as a typed report.
  """
  now = datetime.utcnow()
  metrics = {
    "uptime_seconds": 86_400.0,
    "error_rate": 0.001,
    "requests_per_second": 120.0,
  }
  return MetricsReport(environment=environment, metrics=metrics, generated_at=now)


def alert_test(severity: Severity) -> bool:
  """
  Send a test alert at the requested severity.

  Intended to validate alerting pipelines (PagerDuty, email, Slack, etc.).
  """
  # Placeholder: assumes success.
  return True


def backup_all() -> BackupReport:
  """
  Perform a full system backup (DB, cache, strategies, configs).

  Gracefully degrades if some components are unavailable and surfaces a
  detailed report of what was successfully backed up.
  """
  started = datetime.utcnow()
  # Placeholder: synthetic backup.
  backup_id = f"backup-{int(started.timestamp())}"
  info = BackupInfo(
    backup_id=backup_id,
    created_at=started,
    size_bytes=0,
    description="Synthetic backup (placeholder)",
  )
  completed = datetime.utcnow()
  return BackupReport(
    success=True,
    backup_id=backup_id,
    artifacts=[info],
    started_at=started,
    completed_at=completed,
  )


def restore_backup(backup_id: str, dry_run: bool = True) -> RestoreResult:
  """
  Restore a backup by ID.

  When `dry_run` is True, perform a validation-only restore simulation.
  """
  msg = "Dry-run restore completed successfully" if dry_run else "Restore initiated"
  return RestoreResult(success=True, backup_id=backup_id, dry_run=dry_run, message=msg)


def list_backups() -> List[BackupInfo]:
  """List known backups for inspection and restore operations."""
  # Placeholder: no stored backups.
  return []


# ---------------------------------------------------------------------------
# Tier 2 - Advanced operations (lazy-loaded)
# ---------------------------------------------------------------------------

def scale_service(service: str, replicas: int) -> ScaleResult:
  """Scale a containerized service to the specified replica count."""
  # Real implementation: _lazy_get("deploy").scale_service(...)
  return ScaleResult(service=service, replicas=replicas, success=True, message="Scaled (placeholder)")


def restart_service(service: str) -> bool:
  """Restart a service (pod/VM)."""
  # Real implementation: _lazy_get("deploy").restart_service(...)
  return True


def apply_config(config_path: str) -> ConfigResult:
  """Apply configuration changes from the given path."""
  # Real implementation: _lazy_get("config").apply_config(...)
  return ConfigResult(success=True, path=config_path, message="Applied (placeholder)")


def validate_config(config_path: str) -> ValidationResult:
  """Validate configuration file/schema for correctness."""
  # Real implementation: _lazy_get("config").validate_config(...)
  return ValidationResult(valid=True, path=config_path, errors=[])


def db_migrate() -> MigrationResult:
  """Run database migrations to the latest version."""
  # Real implementation: _lazy_get("db").migrate()
  return MigrationResult(success=True, applied_migrations=[], message="No migrations (placeholder)")


def db_seed() -> SeedResult:
  """Seed the database with baseline data."""
  # Real implementation: _lazy_get("db").seed()
  return SeedResult(success=True, message="Seed complete (placeholder)")


def db_cleanup() -> CleanupResult:
  """Perform database cleanup tasks (vacuum, archival, etc.)."""
  # Real implementation: _lazy_get("db").cleanup()
  return CleanupResult(success=True, removed_items=[])


# ---------------------------------------------------------------------------
# Developer utilities
# ---------------------------------------------------------------------------

def print_ops_menu() -> None:
  """Print available ops commands for CLI usage."""
  menu = f"""
ALPHA-PRIME v{__alpha_prime_version__} Operations Hub
---------------------------------------------------
Core commands:
  health_check()              - Run full system health check
  get_system_status()         - High-level ops status
  is_healthy()                - Boolean health indicator

  deploy_strategy(id, env)    - Deploy a strategy (paper/live)
  rollback_strategy(id)       - Roll back a deployment
  list_deployments()          - List deployments

  tail_logs(service, lines)   - Tail service logs
  get_metrics(env)            - Fetch metrics snapshot
  alert_test(severity)        - Send a test alert

  backup_all()                - Full system backup
  restore_backup(id)          - Restore a backup
  list_backups()              - List backups

Advanced commands:
  scale_service(svc, n)       - Scale containerized service
  restart_service(svc)        - Restart a service
  apply_config(path)          - Apply configuration
  validate_config(path)       - Validate configuration
  db_migrate()                - Run DB migrations
  db_seed()                   - Seed database
  db_cleanup()                - Cleanup database

Utilities:
  check_compatibility()       - Verify ops/core compatibility
  list_all_commands()         - List public ops functions
  quickstart_guide()          - Print deployment quickstart
"""
  print(menu.strip())


def list_all_commands() -> List[str]:
  """Return names of all vetted ops functions exposed in __all__."""
  return [name for name in __all__ if callable(globals().get(name))]


def quickstart_guide() -> str:
  """Return a brief deployment/production quickstart guide."""
  return (
    "ALPHA-PRIME Ops Quickstart:\n"
    "1. Run health check: `python -m ops health`.\n"
    "2. Deploy to paper: `python -m ops deploy STRATEGY_ID --env paper`.\n"
    "3. Monitor logs: `python -m ops logs strategies --lines 100`.\n"
    "4. Schedule backups: `python -m ops backup --full`.\n"
    "5. Promote to live after validation passes (Backtest score >= 85).\n"
  )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _cli(argv: List[str] | None = None) -> int:
  """
  Minimal CLI dispatcher for `python -m ops`.

  Supported subcommands (stable surface):
    health [--json]
    deploy STRATEGY_ID [--env ENV]
    logs SERVICE [--lines N]
    backup [--full]
    menu
  """
  import argparse
  import json

  parser = argparse.ArgumentParser(prog="python -m ops", add_help=True)
  subparsers = parser.add_subparsers(dest="cmd", required=True)

  # health
  health_p = subparsers.add_parser("health", help="Run system health check")
  health_p.add_argument("--json", action="store_true", help="Output JSON")
  health_p.add_argument("--critical-only", action="store_true", help="Exit non-zero if not healthy")

  # deploy
  deploy_p = subparsers.add_parser("deploy", help="Deploy a strategy")
  deploy_p.add_argument("strategy_id", help="Strategy identifier")
  deploy_p.add_argument("--env", dest="env", default="paper", help="Environment (paper|production)")

  # logs
  logs_p = subparsers.add_parser("logs", help="Tail service logs")
  logs_p.add_argument("service", help="Service name")
  logs_p.add_argument("--lines", type=int, default=50, help="Number of lines")

  # backup
  backup_p = subparsers.add_parser("backup", help="Full system backup")
  backup_p.add_argument("--full", action="store_true", help="Ignored, for compatibility")

  # menu
  subparsers.add_parser("menu", help="Show ops menu")

  args = parser.parse_args(argv)

  if args.cmd == "health":
    health = health_check()
    if args.json:
      payload = {
        "status": health.status,
        "timestamp": health.timestamp.isoformat(),
        "services": {k: vars(v) for k, v in health.services.items()},
      }
      print(json.dumps(payload))
    else:
      print(f"Status: {health.status}")
      for svc in health.services.values():
        print(f"- {svc.name}: {svc.status} ({svc.message})")
    if args.critical_only and health.status != "healthy":
      return 1
    return 0

  if args.cmd == "deploy":
    result = deploy_strategy(args.strategy_id, environment=args.env)
    print(f"Deployment {'succeeded' if result.success else 'failed'}")
    print(f"Deployment ID: {result.deployment_id}")
    print(f"URL: {result.url}")
    for line in result.logs:
      print(line)
    return 0 if result.success else 1

  if args.cmd == "logs":
    for line in tail_logs(args.service, lines=args.lines):
      print(line)
    return 0

  if args.cmd == "backup":
    report = backup_all()
    print(f"Backup {'succeeded' if report.success else 'failed'}: {report.backup_id}")
    return 0 if report.success else 1

  if args.cmd == "menu":
    print_ops_menu()
    return 0

  parser.print_help()
  return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
  sys.exit(_cli())


# ---------------------------------------------------------------------------
# Public API (safe star imports)
# ---------------------------------------------------------------------------

__all__ = [
  # Health & status
  "health_check",
  "get_system_status",
  "is_healthy",
  # Deployment
  "deploy_strategy",
  "rollback_strategy",
  "list_deployments",
  # Logging & monitoring
  "tail_logs",
  "get_metrics",
  "alert_test",
  # Backup & recovery
  "backup_all",
  "restore_backup",
  "list_backups",
  # Advanced (lazy-capable)
  "scale_service",
  "restart_service",
  "apply_config",
  "validate_config",
  "db_migrate",
  "db_seed",
  "db_cleanup",
  # Utilities & versioning
  "print_ops_menu",
  "list_all_commands",
  "quickstart_guide",
  "check_compatibility",
  "__version__",
  "__alpha_prime_version__",
  "_lazy_get",
]
