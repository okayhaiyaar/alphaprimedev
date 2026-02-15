#!/usr/bin/env python3
"""
ALPHA-PRIME v2.0 - Portfolio Reset Script
=========================================
Safely reset portfolio state with backup and recovery options.

⚠️  WARNING: This script performs destructive operations!
    Always use --backup and review --dry-run before execution.

Usage:
    # Soft reset (close all positions, keep history)
    python scripts/reset_portfolio.py --mode soft --backup

    # Hard reset (clear positions and open trades)
    python scripts/reset_portfolio.py --mode hard --backup --confirm

    # Full reset (delete all data)
    python scripts/reset_portfolio.py --mode full --backup --confirm

    # Selective reset (specific strategy)
    python scripts/reset_portfolio.py --strategy EMA_Cross_v2 --backup

    # Restore from backup
    python scripts/reset_portfolio.py --restore backups/backup_20260211_200000.sql
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set

import json
import subprocess
import asyncio
import os
import re
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_db_session
from database.models import Portfolio, Position, Trade, Order, Signal  # noqa: F401
from core.portfolio import PortfolioManager
from integrations.redis_client import get_redis_client
from utils.logger import setup_logger
from config import get_settings

# === CONFIGURATION ===
BACKUPS_DIR = PROJECT_ROOT / "backups"
STATE_DIR = PROJECT_ROOT / "data" / "state"
LOGS_DIR = PROJECT_ROOT / "logs"

# === RESET MODES ===
RESET_MODES: Dict[str, Dict[str, Any]] = {
    "soft": {
        "description": "Close all open positions, keep trade history",
        "actions": ["close_positions", "clear_cache"],
    },
    "hard": {
        "description": "Clear positions and open trades, keep closed trades",
        "actions": [
            "delete_positions",
            "delete_open_trades",
            "clear_cache",
            "reset_state",
        ],
    },
    "full": {
        "description": "Delete ALL portfolio data (positions, trades, orders)",
        "actions": [
            "delete_all_positions",
            "delete_all_trades",
            "delete_all_orders",
            "clear_cache",
            "reset_state",
            "clear_logs",
        ],
    },
    "cache": {
        "description": "Clear cache only (Redis, state files)",
        "actions": ["clear_cache", "reset_state"],
    },
}


# === ARGUMENT PARSING ===
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 Portfolio Reset Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reset Modes:
  soft   - Close positions, keep history (safest)
  hard   - Clear positions & open trades (moderate)
  full   - Delete ALL data (most destructive)
  cache  - Clear cache only (no DB changes)

Safety Features:
  - Automatic backup before reset (--backup)
  - Interactive confirmation (--confirm)
  - Dry-run preview (--dry-run)
  - Selective reset (--strategy, --symbol)

Examples:
  # Soft reset with backup
  python scripts/reset_portfolio.py --mode soft --backup

  # Hard reset specific strategy
  python scripts/reset_portfolio.py --mode hard --strategy EMA_Cross_v2 --backup --confirm

  # Full reset (requires confirmation)
  python scripts/reset_portfolio.py --mode full --backup --confirm

  # Preview without execution
  python scripts/reset_portfolio.py --mode hard --dry-run

  # Restore from backup
  python scripts/reset_portfolio.py --restore backup_20260211_200000.sql

  # Clear cache only
  python scripts/reset_portfolio.py --mode cache
        """,
    )

    # Reset mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RESET_MODES.keys()),
        help="Reset mode (soft/hard/full/cache)",
    )

    # Selective reset options
    parser.add_argument("--strategy", type=str, help="Reset only specific strategy")
    parser.add_argument("--symbol", type=str, help="Reset only specific symbol")
    parser.add_argument(
        "--before",
        type=str,
        help="Delete data before date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Delete data after date (YYYY-MM-DD)",
    )

    # Safety options
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before reset (RECOMMENDED)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip interactive confirmation (for automation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without execution",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reset without any confirmation (DANGEROUS)",
    )

    # Restore options
    parser.add_argument(
        "--restore",
        type=Path,
        help="Restore from backup file",
    )

    # Component-specific resets
    parser.add_argument(
        "--positions-only",
        action="store_true",
        help="Reset positions only",
    )
    parser.add_argument(
        "--trades-only",
        action="store_true",
        help="Reset trades only",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Clear cache only",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Clear state files only",
    )

    # Verification
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify reset completion",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


# === ENVIRONMENT & SAFETY HELPERS ===
def is_production_environment() -> bool:
    """Detect if running in production (based on settings/env)."""
    try:
        settings = get_settings()
        env = getattr(settings, "ENVIRONMENT", None) or os.getenv("ENVIRONMENT")
        if env:
            env = env.lower()
            return env in {"prod", "production", "live"}
    except Exception:
        pass
    return False


async def count_open_positions() -> int:
    """Count open positions."""
    async with get_db_session() as session:
        return await session.query(Position).filter(Position.status == "OPEN").count()


async def count_recent_trades(hours: int = 24) -> int:
    """Count trades in recent period."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    async with get_db_session() as session:
        return await session.query(Trade).filter(Trade.timestamp >= cutoff).count()


async def are_strategies_running() -> bool:
    """Detect active strategies (basic check via signals or state)."""
    async with get_db_session() as session:
        active = await session.query(Signal).filter(
            Signal.status.in_(["ACTIVE", "RUNNING"])
        ).count()
    return active > 0


# === SAFETY CHECKS & CONFIRMATIONS ===
async def perform_safety_checks(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Perform safety checks before reset."""
    logger.info("Performing safety checks...")

    warnings: List[str] = []

    if is_production_environment():
        warnings.append("⚠️  PRODUCTION environment detected!")

    position_count = await count_open_positions()
    if position_count > 0:
        warnings.append(f"⚠️  {position_count} open positions will be affected")

    recent_trades = await count_recent_trades(hours=24)
    if recent_trades > 0:
        warnings.append(f"⚠️  {recent_trades} trades in last 24 hours")

    if await are_strategies_running():
        warnings.append("⚠️  Active strategies detected - consider stopping first")

    if warnings:
        logger.warning("\n" + "\n".join(warnings) + "\n")

    if config.get("force"):
        logger.warning("Force flag enabled - bypassing safety confirmation checks")
        return True

    if is_production_environment() and not config.get("backup"):
        logger.error(
            "In production environments, --backup is strongly required before reset"
        )
        return False

    return True


def get_user_confirmation(mode: str, config: Dict[str, Any]) -> bool:
    """Get interactive user confirmation."""
    print("\n" + "=" * 60)
    print(f"  PORTFOLIO RESET - {mode.upper()} MODE")
    print("=" * 60)
    print("\nDescription:")
    print(f"  {RESET_MODES[mode]['description']}")
    print("\nActions to be performed:")

    for action in RESET_MODES[mode]["actions"]:
        print(f"  ✓ {action.replace('_', ' ').title()}")

    if config.get("strategy"):
        print(f"\nFiltered by strategy: {config['strategy']}")
    if config.get("symbol"):
        print(f"Filtered by symbol: {config['symbol']}")
    if config.get("before"):
        print(f"Filtered before date: {config['before']}")
    if config.get("after"):
        print(f"Filtered after date: {config['after']}")

    print("\n" + "=" * 60)

    if config.get("force"):
        return True

    response = input("\nType 'YES' to continue, anything else to abort: ")
    return response.strip().upper() == "YES"


# === BACKUP OPERATIONS ===
async def backup_state_files(timestamp: str, logger: logging.Logger) -> None:
    """Backup state files and cache metadata."""
    state_backup_dir = BACKUPS_DIR / f"state_{timestamp}"
    state_backup_dir.mkdir(parents=True, exist_ok=True)

    if STATE_DIR.exists():
        shutil.copytree(STATE_DIR, state_backup_dir / "state", dirs_exist_ok=True)
        logger.info("State files backed up to %s", state_backup_dir)


async def create_backup(logger: logging.Logger) -> Path:
    """Create full database backup (SQL dump) and state backup."""
    logger.info("Creating backup...")

    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUPS_DIR / f"backup_{timestamp}.sql"

    settings = get_settings()
    db_url = settings.DATABASE_URL

    match = re.match(
        r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", db_url
    )
    if not match:
        raise ValueError("Invalid DATABASE_URL format, expected postgresql+asyncpg://")

    user, password, host, port, dbname = match.groups()

    cmd = [
        "pg_dump",
        "-h",
        host,
        "-p",
        port,
        "-U",
        user,
        "-d",
        dbname,
        "-f",
        str(backup_file),
        "--no-owner",
        "--no-acl",
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Backup failed: %s", result.stderr)
        raise RuntimeError("Backup failed")

    size_mb = backup_file.stat().st_size / 1024 / 1024
    logger.info("✓ Backup created: %s (%.2f MB)", backup_file, size_mb)

    await backup_state_files(timestamp, logger)

    return backup_file


# === RESET EXECUTOR ===
class ResetExecutor:
    """Execute reset operations."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.dry_run = config.get("dry_run", False)

        # Pre-parse date filters
        self.before_dt: Optional[datetime] = (
            datetime.fromisoformat(config["before"])
            if config.get("before")
            else None
        )
        self.after_dt: Optional[datetime] = (
            datetime.fromisoformat(config["after"]) if config.get("after") else None
        )

    async def execute_reset(self, mode: str) -> Dict[str, int]:
        """Execute reset based on mode and component flags."""
        self.logger.info("Executing %s reset...", mode)

        stats = {
            "positions_affected": 0,
            "trades_deleted": 0,
            "orders_deleted": 0,
            "cache_keys_cleared": 0,
            "state_files_cleared": 0,
        }

        actions = self._resolve_actions(mode)

        for action in actions:
            if action == "close_positions":
                stats["positions_affected"] = await self.close_positions()
            elif action == "delete_positions":
                stats["positions_affected"] = await self.delete_positions()
            elif action == "delete_all_positions":
                stats["positions_affected"] = await self.delete_all_positions()
            elif action == "delete_open_trades":
                stats["trades_deleted"] = await self.delete_open_trades()
            elif action == "delete_all_trades":
                stats["trades_deleted"] = await self.delete_all_trades()
            elif action == "delete_all_orders":
                stats["orders_deleted"] = await self.delete_all_orders()
            elif action == "clear_cache":
                stats["cache_keys_cleared"] = await self.clear_cache()
            elif action == "reset_state":
                stats["state_files_cleared"] = await self.reset_state()
            elif action == "clear_logs":
                await self.clear_logs()

        return stats

    def _resolve_actions(self, mode: str) -> List[str]:
        """Resolve actions considering component-only flags."""
        base_actions = RESET_MODES[mode]["actions"]
        actions: Set[str] = set(base_actions)

        if self.config.get("positions_only"):
            actions &= {"close_positions", "delete_positions", "delete_all_positions"}
        if self.config.get("trades_only"):
            actions &= {"delete_open_trades", "delete_all_trades"}
        if self.config.get("cache_only"):
            actions &= {"clear_cache"}
        if self.config.get("state_only"):
            actions &= {"reset_state"}

        if self.config.get("cache_only") or self.config.get("state_only"):
            actions -= {"delete_all_orders", "delete_positions", "delete_all_trades"}

        return list(actions)

    async def close_positions(self) -> int:
        """Close all open positions (soft reset)."""
        self.logger.info("Closing open positions...")

        async with get_db_session() as session:
            query = session.query(Position).filter(Position.status == "OPEN")

            if self.config.get("strategy"):
                query = query.filter(
                    Position.strategy_id == self.config["strategy"]
                )
            if self.config.get("symbol"):
                query = query.filter(Position.symbol == self.config["symbol"])
            if self.before_dt:
                query = query.filter(Position.entry_date < self.before_dt)
            if self.after_dt:
                query = query.filter(Position.entry_date > self.after_dt)

            positions = await query.all()
            count = len(positions)

            if self.dry_run:
                self.logger.info("  [DRY RUN] Would close %d positions", count)
                return count

            pm = PortfolioManager()
            for p in positions:
                await pm.close_position(p.id, reason="MANUAL_RESET")

            await session.commit()
            self.logger.info("  ✓ Closed %d positions", count)
            return count

    async def delete_positions(self) -> int:
        """Delete open positions (hard reset)."""
        self.logger.info("Deleting open positions...")

        async with get_db_session() as session:
            query = session.query(Position).filter(Position.status == "OPEN")

            if self.config.get("strategy"):
                query = query.filter(
                    Position.strategy_id == self.config["strategy"]
                )
            if self.config.get("symbol"):
                query = query.filter(Position.symbol == self.config["symbol"])
            if self.before_dt:
                query = query.filter(Position.entry_date < self.before_dt)
            if self.after_dt:
                query = query.filter(Position.entry_date > self.after_dt)

            count = await query.count()

            if self.dry_run:
                self.logger.info("  [DRY RUN] Would delete %d positions", count)
                return count

            await query.delete(synchronize_session=False)
            await session.commit()

            self.logger.info("  ✓ Deleted %d positions", count)
            return count

    async def delete_all_positions(self) -> int:
        """Delete ALL positions (full reset)."""
        self.logger.info("Deleting ALL positions...")

        async with get_db_session() as session:
            query = session.query(Position)

            if self.config.get("strategy"):
                query = query.filter(
                    Position.strategy_id == self.config["strategy"]
                )
            if self.config.get("symbol"):
                query = query.filter(Position.symbol == self.config["symbol"])
            if self.before_dt:
                query = query.filter(Position.entry_date < self.before_dt)
            if self.after_dt:
                query = query.filter(Position.entry_date > self.after_dt)

            count = await query.count()

            if self.dry_run:
                self.logger.info("  [DRY RUN] Would delete %d positions", count)
                return count

            await query.delete(synchronize_session=False)
            await session.commit()

            self.logger.info("  ✓ Deleted %d positions", count)
            return count

    async def delete_open_trades(self) -> int:
        """Delete open/pending trades."""
        self.logger.info("Deleting open trades...")

        async with get_db_session() as session:
            query = session.query(Trade).filter(
                Trade.status.in_(["OPEN", "PENDING"])
            )

            if self.config.get("strategy"):
                query = query.filter(
                    Trade.strategy_id == self.config["strategy"]
                )
            if self.config.get("symbol"):
                query = query.filter(Trade.symbol == self.config["symbol"])
            if self.before_dt:
                query = query.filter(Trade.timestamp < self.before_dt)
            if self.after_dt:
                query = query.filter(Trade.timestamp > self.after_dt)

            count = await query.count()

            if self.dry_run:
                self.logger.info("  [DRY RUN] Would delete %d trades", count)
                return count

            await query.delete(synchronize_session=False)
            await session.commit()

            self.logger.info("  ✓ Deleted %d trades", count)
            return count

    async def delete_all_trades(self) -> int:
        """Delete ALL trades."""
        self.logger.info("Deleting ALL trades...")

        async with get_db_session() as session:
            query = session.query(Trade)

            if self.config.get("strategy"):
                query = query.filter(
                    Trade.strategy_id == self.config["strategy"]
                )
            if self.before_dt:
                query = query.filter(Trade.timestamp < self.before_dt)
            if self.after_dt:
                query = query.filter(Trade.timestamp > self.after_dt)

            count = await query.count()

            if self.dry_run:
                self.logger.info("  [DRY RUN] Would delete %d trades", count)
                return count

            await query.delete(synchronize_session=False)
            await session.commit()

            self.logger.info("  ✓ Deleted %d trades", count)
            return count

    async def delete_all_orders(self) -> int:
        """Delete ALL orders."""
        self.logger.info("Deleting ALL orders...")

        async with get_db_session() as session:
            query = session.query(Order)

            if self.config.get("strategy"):
                query = query.filter(
                    Order.strategy_id == self.config["strategy"]
                )
            if self.before_dt:
                query = query.filter(Order.timestamp < self.before_dt)
            if self.after_dt:
                query = query.filter(Order.timestamp > self.after_dt)

            count = await query.count()

            if self.dry_run:
                self.logger.info("  [DRY RUN] Would delete %d orders", count)
                return count

            await query.delete(synchronize_session=False)
            await session.commit()

            self.logger.info("  ✓ Deleted %d orders", count)
            return count

    async def clear_cache(self) -> int:
        """Clear Redis cache."""
        self.logger.info("Clearing Redis cache...")

        redis = await get_redis_client()

        cursor = b"0"
        pattern = "alpha_prime:*"
        total_deleted = 0

        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=500)
            if self.dry_run:
                total_deleted += len(keys)
            else:
                if keys:
                    await redis.delete(*keys)
                    total_deleted += len(keys)
            if cursor == b"0":
                break

        if self.dry_run:
            self.logger.info("  [DRY RUN] Would delete %d cache keys", total_deleted)
        else:
            self.logger.info("  ✓ Cleared %d cache keys", total_deleted)
        return total_deleted

    async def reset_state(self) -> int:
        """Clear state files."""
        self.logger.info("Clearing state files...")

        if not STATE_DIR.exists():
            self.logger.info("  No state directory found")
            return 0

        state_files = list(STATE_DIR.glob("*.json")) + list(STATE_DIR.glob("*.pkl"))

        if self.dry_run:
            self.logger.info(
                "  [DRY RUN] Would delete %d state files", len(state_files)
            )
            return len(state_files)

        for f in state_files:
            f.unlink()

        self.logger.info("  ✓ Cleared %d state files", len(state_files))
        return len(state_files)

    async def clear_logs(self) -> None:
        """Clear old log files (keep last 7 days)."""
        self.logger.info("Clearing old logs...")

        if not LOGS_DIR.exists():
            self.logger.info("  No logs directory found")
            return

        cutoff = datetime.now() - timedelta(days=7)
        log_files = [
            f
            for f in LOGS_DIR.glob("*.log")
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff
        ]

        if self.dry_run:
            self.logger.info(
                "  [DRY RUN] Would delete %d log files", len(log_files)
            )
            return

        for f in log_files:
            f.unlink()

        self.logger.info("  ✓ Cleared %d old log files", len(log_files))


# === RESTORE OPERATIONS ===
async def restore_from_backup(backup_file: Path, logger: logging.Logger) -> None:
    """Restore database from backup file."""
    logger.info("Restoring from backup: %s", backup_file)

    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    print("\n" + "=" * 60)
    print("  RESTORE FROM BACKUP")
    print("=" * 60)
    print(f"\nBackup file: {backup_file}")
    print(f"Size: {backup_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Created: {datetime.fromtimestamp(backup_file.stat().st_mtime)}")
    print("\n⚠️  This will REPLACE current database!")
    print("=" * 60 + "\n")

    response = input("Type 'RESTORE' to continue: ")
    if response.strip().upper() != "RESTORE":
        logger.info("Restoration cancelled by user")
        return

    settings = get_settings()
    db_url = settings.DATABASE_URL

    match = re.match(
        r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", db_url
    )
    if not match:
        raise ValueError("Invalid DATABASE_URL format, expected postgresql+asyncpg://")

    user, password, host, port, dbname = match.groups()

    cmd = [
        "psql",
        "-h",
        host,
        "-p",
        port,
        "-U",
        user,
        "-d",
        dbname,
        "-f",
        str(backup_file),
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Restore failed: %s", result.stderr)
        raise RuntimeError("Restore failed")

    logger.info("✓ Database restored successfully")


# === VERIFICATION ===
async def verify_reset(logger: logging.Logger) -> Dict[str, int]:
    """Verify reset completion."""
    logger.info("Verifying reset...")

    verification: Dict[str, int] = {}

    async with get_db_session() as session:
        verification["open_positions"] = await session.query(Position).filter(
            Position.status == "OPEN"
        ).count()

        verification["open_trades"] = await session.query(Trade).filter(
            Trade.status.in_(["OPEN", "PENDING"])
        ).count()

        verification["pending_orders"] = await session.query(Order).filter(
            Order.status == "PENDING"
        ).count()

    redis = await get_redis_client()
    verification["cache_keys"] = len(await redis.keys("alpha_prime:*"))

    if STATE_DIR.exists():
        verification["state_files"] = len(list(STATE_DIR.glob("*.json")))
    else:
        verification["state_files"] = 0

    logger.info("\nVerification Results:")
    logger.info("  Open positions: %d", verification["open_positions"])
    logger.info("  Open trades: %d", verification["open_trades"])
    logger.info("  Pending orders: %d", verification["pending_orders"])
    logger.info("  Cache keys: %d", verification["cache_keys"])
    logger.info("  State files: %d", verification["state_files"])

    return verification


# === MAIN EXECUTION ===
async def main() -> int:
    """Main execution flow."""
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("portfolio_reset", level=log_level)

    try:
        config = vars(args)

        if args.restore:
            await restore_from_backup(args.restore, logger)
            return 0

        if not args.mode:
            logger.error("--mode is required (or use --restore)")
            return 1

        if not await perform_safety_checks(config, logger):
            logger.error("Safety checks failed, aborting")
            return 1

        if not args.confirm and not args.dry_run and not args.force:
            if not get_user_confirmation(args.mode, config):
                logger.info("Reset cancelled by user")
                return 0

        backup_file: Optional[Path] = None
        if args.backup and not args.dry_run:
            backup_file = await create_backup(logger)

        executor = ResetExecutor(config, logger)
        stats = await executor.execute_reset(args.mode)

        logger.info("\n" + "=" * 60)
        logger.info("RESET COMPLETE")
        logger.info("=" * 60)
        logger.info("Positions affected: %d", stats["positions_affected"])
        logger.info("Trades deleted: %d", stats["trades_deleted"])
        logger.info("Orders deleted: %d", stats["orders_deleted"])
        logger.info("Cache keys cleared: %d", stats["cache_keys_cleared"])
        logger.info("State files cleared: %d", stats["state_files_cleared"])

        if backup_file:
            logger.info("\nBackup saved: %s", backup_file)

        if args.verify:
            await verify_reset(logger)

        logger.info("\n✓ Portfolio reset completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nReset cancelled by user (KeyboardInterrupt)")
        return 1
    except Exception as exc:
        logger.error("Reset failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    asyncio.run(main())
