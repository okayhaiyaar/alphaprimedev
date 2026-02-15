#!/usr/bin/env python3
"""
ALPHA-PRIME v2.0 - System Stress Testing Script
==============================================
Comprehensive stress testing and performance benchmarking.

Usage:
    python scripts/stress_test.py --scenario high_load --duration 300
    python scripts/stress_test.py --scenario concurrent_trades --workers 50
    python scripts/stress_test.py --scenario database_load --connections 100
    python scripts/stress_test.py --profile --report --output stress_report.html
"""

import sys
import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import time
import psutil  # [web:704]
import tracemalloc  # [web:702]
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy_engine import StrategyEngine
from core.order_manager import OrderManager
from core.execution_engine import ExecutionEngine
from core.portfolio import PortfolioManager
from database.connection import get_db_session, DatabasePool
from integrations.redis_client import get_redis_client
from utils.logger import setup_logger

# === CONFIGURATION ===
RESULTS_DIR = PROJECT_ROOT / "stress_test_results"
DEFAULT_DURATION = 60  # seconds
DEFAULT_WORKERS = 10

# === TEST SCENARIOS ===
TEST_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "high_load": {
        "description": "High throughput signal processing",
        "duration": 300,
        "signal_rate": 100,
    },
    "concurrent_trades": {
        "description": "Simultaneous trade execution",
        "workers": 50,
        "trades_per_worker": 100,
    },
    "database_load": {
        "description": "Database connection pooling stress",
        "connections": 100,
        "queries_per_connection": 1000,
    },
    "api_burst": {
        "description": "Burst API traffic",
        "burst_size": 500,
        "burst_interval": 10,
    },
    "memory_leak": {
        "description": "Memory leak detection",
        "duration": 600,
        "sample_interval": 5,
    },
    "cache_storm": {
        "description": "Redis cache stampede",
        "workers": 100,
        "operations": 10000,
    },
    "websocket_flood": {
        "description": "WebSocket connection flood",
        "connections": 500,
        "messages_per_connection": 1000,
    },
    "error_injection": {
        "description": "Fault injection and recovery",
        "error_rate": 0.2,
        "duration": 180,
    },
}


# === METRICS DATACLASS ===
@dataclass
class StressTestMetrics:
    """Stress test metrics."""
    scenario: str
    start_time: datetime
    end_time: datetime
    duration: float

    # Throughput
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float

    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float

    # Resources
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float

    # Database
    db_connections_peak: int
    db_query_time_avg_ms: float
    db_errors: int

    # Cache
    cache_hit_rate: float
    cache_operations: int

    # Errors
    error_rate: float
    error_types: Dict[str, int]


# === ARGUMENT PARSING ===
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 Stress Testing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Scenarios:
  high_load         - High throughput signal processing
  concurrent_trades - Simultaneous trade execution
  database_load     - Database connection stress
  api_burst         - Burst traffic patterns
  memory_leak       - Memory leak detection
  cache_storm       - Redis cache stampede
  websocket_flood   - WebSocket stress test
  error_injection   - Fault injection testing

Examples:
  # High load test for 5 minutes
  python scripts/stress_test.py --scenario high_load --duration 300

  # Concurrent trades with 50 workers
  python scripts/stress_test.py --scenario concurrent_trades --workers 50

  # Database stress test
  python scripts/stress_test.py --scenario database_load --connections 100

  # All scenarios with profiling
  python scripts/stress_test.py --all --profile --report

  # Custom scenario configuration
  python scripts/stress_test.py --config stress_test_config.json
        """,
    )

    # Scenario selection
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(TEST_SCENARIOS.keys()),
        help="Test scenario to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test scenarios",
    )

    # Test parameters
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--operations",
        type=int,
        help="Number of operations to perform",
    )
    parser.add_argument(
        "--rate",
        type=int,
        help="Operations per second (rate limiting)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file for custom scenarios",
    )

    # Profiling options
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable CPU and memory profiling",
    )
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        help="Enable CPU profiling only",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling only",
    )

    # Monitoring
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=5,
        help="Resource monitoring interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--realtime-stats",
        action="store_true",
        help="Display real-time statistics",
    )

    # Output options
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (HTML/JSON)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export results as JSON",
    )

    # Thresholds
    parser.add_argument(
        "--max-latency",
        type=float,
        help="Maximum acceptable latency (ms)",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.01,
        help="Maximum acceptable error rate (default: 0.01)",
    )
    parser.add_argument(
        "--min-throughput",
        type=float,
        help="Minimum acceptable throughput (ops/sec)",
    )

    # Database options
    parser.add_argument(
        "--db-pool-size",
        type=int,
        default=20,
        help="Database connection pool size (default: 20)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


# === COMMON HELPERS ===
def build_scenario_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args and optional JSON config into base scenario settings."""
    config: Dict[str, Any] = vars(args).copy()

    if args.config:
        cfg_path: Path = args.config
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        config.update(file_cfg)

    return config


def build_metrics(
    scenario: str,
    start_time: datetime,
    end_time: datetime,
    operations: int,
    successful: int,
    failed: int,
    latencies_ms: List[float],
    resource_stats: Optional[Dict[str, float]] = None,
    db_stats: Optional[Dict[str, Any]] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
    error_types: Optional[Dict[str, int]] = None,
) -> StressTestMetrics:
    """Build StressTestMetrics from collected stats."""
    duration = (end_time - start_time).total_seconds() or 1.0

    total_ops = operations
    succ = successful
    fail = failed
    ops_per_sec = total_ops / duration if duration > 0 else 0.0

    if latencies_ms:
        lat_arr = np.array(latencies_ms)
        avg_lat = float(lat_arr.mean())
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        max_lat = float(lat_arr.max())
    else:
        avg_lat = p50 = p95 = p99 = max_lat = 0.0

    res = resource_stats or {}
    db = db_stats or {}
    cache = cache_stats or {}
    err_types = error_types or {}

    total_errors = fail
    error_rate = total_errors / total_ops if total_ops > 0 else 0.0

    return StressTestMetrics(
        scenario=scenario,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        total_operations=total_ops,
        successful_operations=succ,
        failed_operations=fail,
        operations_per_second=ops_per_sec,
        avg_latency_ms=avg_lat,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        max_latency_ms=max_lat,
        peak_memory_mb=float(res.get("peak_memory_mb", 0.0)),
        avg_memory_mb=float(res.get("avg_memory_mb", 0.0)),
        peak_cpu_percent=float(res.get("peak_cpu_percent", 0.0)),
        avg_cpu_percent=float(res.get("avg_cpu_percent", 0.0)),
        db_connections_peak=int(db.get("db_connections_peak", 0)),
        db_query_time_avg_ms=float(db.get("db_query_time_avg_ms", 0.0)),
        db_errors=int(db.get("db_errors", 0)),
        cache_hit_rate=float(cache.get("cache_hit_rate", 0.0)),
        cache_operations=int(cache.get("cache_operations", 0)),
        error_rate=float(error_rate),
        error_types=err_types,
    )


# === RESOURCE MONITOR ===
class ResourceMonitor:
    """Monitor process CPU and memory usage during stress tests."""

    def __init__(self) -> None:
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.process = psutil.Process()  # [web:704]

    async def monitor(self, interval: int = 1, realtime: bool = False, logger: Optional[logging.Logger] = None):
        """Continuously monitor resources."""
        try:
            while True:
                cpu_percent = self.process.cpu_percent(interval=0.1)
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024

                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(mem_mb)

                if realtime and logger:
                    logger.info(
                        "CPU: %.1f%% | Memory: %.1f MB",
                        cpu_percent,
                        mem_mb,
                    )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    def get_statistics(self) -> Dict[str, float]:
        """Return peak and average CPU/memory usage."""
        return {
            "peak_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "avg_cpu_percent": float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0,
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0.0,
            "avg_memory_mb": float(np.mean(self.memory_samples)) if self.memory_samples else 0.0,
        }


# === HIGH LOAD TESTING ===
class HighLoadTester:
    """Test system under high signal processing load."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.strategy_engine: Optional[StrategyEngine] = None
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}

    async def run(self) -> StressTestMetrics:
        """Run high load test."""
        self.logger.info("Starting high load test...")

        start_time = datetime.now()
        duration = int(self.config.get("duration", 60))
        scenario_defaults = TEST_SCENARIOS["high_load"]
        signal_rate = int(
            self.config.get("signal_rate", scenario_defaults["signal_rate"])
        )

        self.strategy_engine = StrategyEngine()
        await self.strategy_engine.initialize()

        operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []

        end_time = start_time + timedelta(seconds=duration)

        while datetime.now() < end_time:
            batch_start = time.time()
            tasks: List[asyncio.Task] = []
            for _ in range(signal_rate):
                signal = self._generate_test_signal()
                tasks.append(asyncio.create_task(self._process_signal(signal)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                operations += 1
                if isinstance(result, Exception):
                    failed += 1
                    etype = type(result).__name__
                    self.error_types[etype] = self.error_types.get(etype, 0) + 1
                    self.logger.debug("Signal failed: %s", result)
                else:
                    successful += 1
                    latencies.append(result)

            batch_duration = time.time() - batch_start
            sleep_time = max(0.0, 1.0 - batch_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        metrics = build_metrics(
            scenario="high_load",
            start_time=start_time,
            end_time=end_time,
            operations=operations,
            successful=successful,
            failed=failed,
            latencies_ms=latencies,
            resource_stats=resource_stats,
            db_stats={},
            cache_stats={},
            error_types=self.error_types,
        )
        return metrics

    async def _process_signal(self, signal: Dict[str, Any]) -> float:
        """Process a single signal and return latency in ms."""
        assert self.strategy_engine is not None
        start = time.time()
        await self.strategy_engine.process_signal(signal)
        latency_ms = (time.time() - start) * 1000.0
        return latency_ms

    def _generate_test_signal(self) -> Dict[str, Any]:
        """Generate random test signal."""
        import random

        return {
            "symbol": random.choice(["AAPL", "MSFT", "GOOGL", "TSLA"]),
            "action": random.choice(["BUY", "SELL"]),
            "confidence": random.uniform(0.6, 0.99),
            "timestamp": datetime.utcnow().isoformat(),
        }


# === CONCURRENT TRADE EXECUTION ===
class ConcurrentTradesTester:
    """Test concurrent trade execution."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}

    async def run(self) -> StressTestMetrics:
        """Run concurrent trades test."""
        self.logger.info("Starting concurrent trades test...")

        start_time = datetime.now()
        scenario_defaults = TEST_SCENARIOS["concurrent_trades"]
        workers = int(self.config.get("workers", scenario_defaults["workers"]))
        trades_per_worker = int(
            self.config.get("trades_per_worker", scenario_defaults["trades_per_worker"])
        )

        order_manager = OrderManager()
        execution_engine = ExecutionEngine()

        worker_tasks: List[asyncio.Task] = []
        for worker_id in range(workers):
            worker_tasks.append(
                asyncio.create_task(
                    self._worker(
                        worker_id,
                        trades_per_worker,
                        order_manager,
                        execution_engine,
                    )
                )
            )

        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        total_operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []

        for result in results:
            if isinstance(result, Exception):
                failed += 1
                etype = type(result).__name__
                self.error_types[etype] = self.error_types.get(etype, 0) + 1
                self.logger.error("Worker failed: %s", result)
                continue

            total_operations += result["operations"]
            successful += result["successful"]
            failed += result["failed"]
            latencies.extend(result["latencies"])
            for et, count in result["error_types"].items():
                self.error_types[et] = self.error_types.get(et, 0) + count

        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        metrics = build_metrics(
            scenario="concurrent_trades",
            start_time=start_time,
            end_time=end_time,
            operations=total_operations,
            successful=successful,
            failed=failed,
            latencies_ms=latencies,
            resource_stats=resource_stats,
            db_stats={},
            cache_stats={},
            error_types=self.error_types,
        )
        return metrics

    async def _worker(
        self,
        worker_id: int,
        num_trades: int,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ) -> Dict[str, Any]:
        """Worker that creates and executes trades."""
        operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []
        error_types: Dict[str, int] = {}

        for i in range(num_trades):
            try:
                start = time.time()

                order = await order_manager.create_order(
                    symbol=f"TEST_{worker_id % 10}",
                    side="BUY" if i % 2 == 0 else "SELL",
                    quantity=100,
                    order_type="MARKET",
                )
                await execution_engine.submit_order(order)

                latency_ms = (time.time() - start) * 1000.0
                latencies.append(latency_ms)
                successful += 1
            except Exception as exc:
                failed += 1
                etype = type(exc).__name__
                error_types[etype] = error_types.get(etype, 0) + 1
                self.logger.debug(
                    "Worker %d trade %d failed: %s", worker_id, i, exc
                )
            operations += 1

        return {
            "operations": operations,
            "successful": successful,
            "failed": failed,
            "latencies": latencies,
            "error_types": error_types,
        }


# === DATABASE LOAD TESTING ===
class DatabaseLoadTester:
    """Test database under heavy load."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}

    async def run(self) -> StressTestMetrics:
        """Run database load test."""
        self.logger.info("Starting database load test...")

        start_time = datetime.now()
        scenario_defaults = TEST_SCENARIOS["database_load"]
        connections = int(
            self.config.get("connections", scenario_defaults["connections"])
        )
        queries_per_connection = int(
            self.config.get(
                "queries_per_connection",
                scenario_defaults["queries_per_connection"],
            )
        )

        pool = DatabasePool(max_connections=connections)

        tasks: List[asyncio.Task] = []
        for conn_id in range(connections):
            tasks.append(
                asyncio.create_task(
                    self._db_worker(conn_id, queries_per_connection, pool)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_queries = 0
        successful = 0
        failed = 0
        query_latencies: List[float] = []

        for result in results:
            if isinstance(result, Exception):
                failed += 1
                etype = type(result).__name__
                self.error_types[etype] = self.error_types.get(etype, 0) + 1
                self.logger.error("DB worker failed: %s", result)
                continue

            total_queries += result["queries"]
            successful += result["successful"]
            failed += result["failed"]
            query_latencies.extend(result["latencies"])
            for et, count in result["error_types"].items():
                self.error_types[et] = self.error_types.get(et, 0) + count

        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        avg_query_time = (
            float(np.mean(query_latencies)) if query_latencies else 0.0
        )
        db_stats = {
            "db_connections_peak": connections,
            "db_query_time_avg_ms": avg_query_time,
            "db_errors": failed,
        }

        metrics = build_metrics(
            scenario="database_load",
            start_time=start_time,
            end_time=end_time,
            operations=total_queries,
            successful=successful,
            failed=failed,
            latencies_ms=query_latencies,
            resource_stats=resource_stats,
            db_stats=db_stats,
            cache_stats={},
            error_types=self.error_types,
        )
        return metrics

    async def _db_worker(
        self,
        worker_id: int,
        num_queries: int,
        pool: DatabasePool,
    ) -> Dict[str, Any]:
        """Execute a mixture of read/write queries."""
        queries = 0
        successful = 0
        failed = 0
        latencies: List[float] = []
        error_types: Dict[str, int] = {}

        async with pool.get_session() as session:
            for i in range(num_queries):
                try:
                    start = time.time()

                    if i % 4 == 0:
                        await session.execute("SELECT COUNT(*) FROM trades")
                    elif i % 4 == 1:
                        await session.execute(
                            "INSERT INTO test_stress (data) VALUES (:data)",
                            {"data": f"worker_{worker_id}_q_{i}"},
                        )
                    elif i % 4 == 2:
                        await session.execute(
                            "UPDATE test_stress SET data = :data WHERE id = :id",
                            {"data": f"updated_{i}", "id": i},
                        )
                    else:
                        await session.execute(
                            "SELECT t.id, p.id FROM trades t "
                            "JOIN positions p ON t.position_id = p.id "
                            "LIMIT 10"
                        )

                    await session.commit()

                    latency_ms = (time.time() - start) * 1000.0
                    latencies.append(latency_ms)
                    successful += 1
                except Exception as exc:
                    failed += 1
                    etype = type(exc).__name__
                    error_types[etype] = error_types.get(etype, 0) + 1
                    self.logger.debug("DB query failed: %s", exc)
                queries += 1

        return {
            "queries": queries,
            "successful": successful,
            "failed": failed,
            "latencies": latencies,
            "error_types": error_types,
        }


# === API BURST TESTING (SIMULATED) ===
class APIBurstTester:
    """Simulate burst API traffic against internal components."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}

    async def run(self) -> StressTestMetrics:
        self.logger.info("Starting API burst test...")

        start_time = datetime.now()
        scenario_defaults = TEST_SCENARIOS["api_burst"]
        burst_size = int(self.config.get("burst_size", scenario_defaults["burst_size"]))
        burst_interval = float(
            self.config.get("burst_interval", scenario_defaults["burst_interval"])
        )

        duration = int(self.config.get("duration", DEFAULT_DURATION))
        end_time = start_time + timedelta(seconds=duration)

        operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []

        while datetime.now() < end_time:
            tasks: List[asyncio.Task] = []
            for _ in range(burst_size):
                tasks.append(asyncio.create_task(self._fake_api_call()))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                operations += 1
                if isinstance(result, Exception):
                    failed += 1
                    etype = type(result).__name__
                    self.error_types[etype] = self.error_types.get(etype, 0) + 1
                else:
                    successful += 1
                    latencies.append(result)

            await asyncio.sleep(burst_interval)

        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        metrics = build_metrics(
            scenario="api_burst",
            start_time=start_time,
            end_time=end_time,
            operations=operations,
            successful=successful,
            failed=failed,
            latencies_ms=latencies,
            resource_stats=resource_stats,
            db_stats={},
            cache_stats={},
            error_types=self.error_types,
        )
        return metrics

    async def _fake_api_call(self) -> float:
        """Simulate internal API call latency."""
        start = time.time()
        await asyncio.sleep(0.01)
        return (time.time() - start) * 1000.0


# === MEMORY PROFILING ===
class MemoryProfiler:
    """Profile memory usage during stress test."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.snapshots: List[tracemalloc.Snapshot] = []

    async def run(self) -> StressTestMetrics:
        """Run memory leak detection test."""
        self.logger.info("Starting memory profiling...")

        tracemalloc.start()  # [web:702]

        start_time = datetime.now()
        duration = int(self.config.get("duration", TEST_SCENARIOS["memory_leak"]["duration"]))
        sample_interval = int(
            self.config.get("sample_interval", TEST_SCENARIOS["memory_leak"]["sample_interval"])
        )

        monitor_task = asyncio.create_task(self._monitor_memory(sample_interval))
        workload_task = asyncio.create_task(self._memory_intensive_workload(duration))

        await workload_task
        monitor_task.cancel()

        try:
            self.snapshots.append(tracemalloc.take_snapshot())
        except RuntimeError:
            pass

        end_time = datetime.now()
        tracemalloc.stop()

        resource_stats = self.monitor.get_statistics()
        metrics = build_metrics(
            scenario="memory_leak",
            start_time=start_time,
            end_time=end_time,
            operations=len(self.snapshots),
            successful=len(self.snapshots),
            failed=0,
            latencies_ms=[],
            resource_stats=resource_stats,
            db_stats={},
            cache_stats={},
            error_types={},
        )
        return metrics

    async def _monitor_memory(self, interval: int) -> None:
        """Take tracemalloc snapshots periodically."""
        try:
            while True:
                snapshot = tracemalloc.take_snapshot()
                self.snapshots.append(snapshot)
                current, peak = tracemalloc.get_traced_memory()
                self.logger.info(
                    "Memory trace: current=%.1f MB peak=%.1f MB",
                    current / 1024 / 1024,
                    peak / 1024 / 1024,
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    async def _memory_intensive_workload(self, duration: int) -> None:
        """Simulate some memory-intensive operations."""
        end = datetime.now() + timedelta(seconds=duration)
        data_store: List[List[int]] = []
        while datetime.now() < end:
            data = list(range(20000))
            data_store.append(data)
            await asyncio.sleep(0.05)
            if len(data_store) > 200:
                data_store = data_store[-100:]


# === CACHE STORM (REDIS) ===
class CacheStormTester:
    """Stress test Redis cache under heavy load."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}
        self.cache_ops = 0
        self.cache_hits = 0

    async def run(self) -> StressTestMetrics:
        self.logger.info("Starting cache storm test...")

        start_time = datetime.now()
        scenario_defaults = TEST_SCENARIOS["cache_storm"]
        workers = int(self.config.get("workers", scenario_defaults["workers"]))
        operations = int(self.config.get("operations", scenario_defaults["operations"]))

        redis = await get_redis_client()
        ops_per_worker = max(1, operations // workers)

        tasks: List[asyncio.Task] = []
        for wid in range(workers):
            tasks.append(
                asyncio.create_task(self._worker(wid, ops_per_worker, redis))
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_operations = 0
        successful = 0
        failed = 0

        for result in results:
            if isinstance(result, Exception):
                failed += 1
                etype = type(result).__name__
                self.error_types[etype] = self.error_types.get(etype, 0) + 1
                self.logger.error("Cache worker failed: %s", result)
                continue

            total_operations += result["operations"]
            successful += result["successful"]
            failed += result["failed"]
            self.cache_hits += result["cache_hits"]

        self.cache_ops = total_operations
        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        hit_rate = self.cache_hits / self.cache_ops * 100 if self.cache_ops > 0 else 0.0
        cache_stats = {
            "cache_hit_rate": hit_rate,
            "cache_operations": self.cache_ops,
        }

        metrics = build_metrics(
            scenario="cache_storm",
            start_time=start_time,
            end_time=end_time,
            operations=total_operations,
            successful=successful,
            failed=failed,
            latencies_ms=[],
            resource_stats=resource_stats,
            db_stats={},
            cache_stats=cache_stats,
            error_types=self.error_types,
        )
        return metrics

    async def _worker(self, worker_id: int, ops: int, redis) -> Dict[str, Any]:
        operations = 0
        successful = 0
        failed = 0
        cache_hits = 0

        for i in range(ops):
            key = f"stress:test:{worker_id}:{i % 100}"
            try:
                val = await redis.get(key)
                if val is None:
                    await redis.set(key, f"value_{i}", ex=60)
                else:
                    cache_hits += 1
                successful += 1
            except Exception as exc:
                failed += 1
                self.logger.debug("Cache op failed: %s", exc)
            operations += 1

        return {
            "operations": operations,
            "successful": successful,
            "failed": failed,
            "cache_hits": cache_hits,
        }


# === WEBSOCKET FLOOD (SIMULATED) ===
class WebSocketFloodTester:
    """Stress test WebSocket connections (simulated client load)."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}

    async def run(self) -> StressTestMetrics:
        self.logger.info("Starting WebSocket flood test...")

        start_time = datetime.now()
        scenario_defaults = TEST_SCENARIOS["websocket_flood"]
        connections = int(
            self.config.get("connections", scenario_defaults["connections"])
        )
        messages_per_connection = int(
            self.config.get(
                "messages_per_connection",
                scenario_defaults["messages_per_connection"],
            )
        )

        operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []

        tasks: List[asyncio.Task] = []
        for cid in range(connections):
            tasks.append(
                asyncio.create_task(
                    self._connection_worker(cid, messages_per_connection)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                failed += 1
                etype = type(result).__name__
                self.error_types[etype] = self.error_types.get(etype, 0) + 1
                self.logger.error("WebSocket worker failed: %s", result)
                continue

            operations += result["operations"]
            successful += result["successful"]
            failed += result["failed"]
            latencies.extend(result["latencies"])
            for et, count in result["error_types"].items():
                self.error_types[et] = self.error_types.get(et, 0) + count

        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        metrics = build_metrics(
            scenario="websocket_flood",
            start_time=start_time,
            end_time=end_time,
            operations=operations,
            successful=successful,
            failed=failed,
            latencies_ms=latencies,
            resource_stats=resource_stats,
            db_stats={},
            cache_stats={},
            error_types=self.error_types,
        )
        return metrics

    async def _connection_worker(self, conn_id: int, messages: int) -> Dict[str, Any]:
        """Simulate a WebSocket connection sending messages."""
        operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []
        error_types: Dict[str, int] = {}

        for i in range(messages):
            try:
                start = time.time()
                await asyncio.sleep(0.005)
                latency_ms = (time.time() - start) * 1000.0
                latencies.append(latency_ms)
                successful += 1
            except Exception as exc:
                failed += 1
                etype = type(exc).__name__
                error_types[etype] = error_types.get(etype, 0) + 1
            operations += 1

        return {
            "operations": operations,
            "successful": successful,
            "failed": failed,
            "latencies": latencies,
            "error_types": error_types,
        }


# === ERROR INJECTION ===
class ErrorInjectionTester:
    """Inject failures to test resilience and recovery."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.error_types: Dict[str, int] = {}

    async def run(self) -> StressTestMetrics:
        self.logger.info("Starting error injection test...")

        start_time = datetime.now()
        scenario_defaults = TEST_SCENARIOS["error_injection"]
        duration = int(self.config.get("duration", scenario_defaults["duration"]))
        error_rate = float(
            self.config.get("error_rate", scenario_defaults["error_rate"])
        )

        end_time = start_time + timedelta(seconds=duration)
        operations = 0
        successful = 0
        failed = 0
        latencies: List[float] = []

        while datetime.now() < end_time:
            start = time.time()
            try:
                await self._faulty_operation(error_rate)
                successful += 1
            except Exception as exc:
                failed += 1
                etype = type(exc).__name__
                self.error_types[etype] = self.error_types.get(etype, 0) + 1
            operations += 1
            latencies.append((time.time() - start) * 1000.0)

        end_time = datetime.now()
        resource_stats = self.monitor.get_statistics()

        metrics = build_metrics(
            scenario="error_injection",
            start_time=start_time,
            end_time=end_time,
            operations=operations,
            successful=successful,
            failed=failed,
            latencies_ms=latencies,
            resource_stats=resource_stats,
            db_stats={},
            cache_stats={},
            error_types=self.error_types,
        )
        return metrics

    async def _faulty_operation(self, error_rate: float) -> None:
        """Randomly succeed or raise to simulate failures."""
        import random

        await asyncio.sleep(0.005)
        if random.random() < error_rate:
            raise RuntimeError("Injected failure")


# === THRESHOLD VALIDATION ===
def validate_thresholds(
    metrics: StressTestMetrics, config: Dict[str, Any], logger: logging.Logger
) -> bool:
    """Validate metrics against configured thresholds."""
    ok = True

    max_latency = config.get("max_latency")
    if max_latency is not None and metrics.p95_latency_ms > max_latency:
        logger.warning(
            "Latency threshold exceeded: p95=%.2f ms > %.2f ms",
            metrics.p95_latency_ms,
            max_latency,
        )
        ok = False

    max_error_rate = config.get("max_error_rate", 0.01)
    if metrics.error_rate > max_error_rate:
        logger.warning(
            "Error rate threshold exceeded: %.3f > %.3f",
            metrics.error_rate,
            max_error_rate,
        )
        ok = False

    min_throughput = config.get("min_throughput")
    if min_throughput is not None and metrics.operations_per_second < min_throughput:
        logger.warning(
            "Throughput threshold not met: %.2f < %.2f ops/sec",
            metrics.operations_per_second,
            min_throughput,
        )
        ok = False

    return ok


# === REPORTING ===
class StressTestReporter:
    """Generate stress test reports."""

    def __init__(self, results: List[StressTestMetrics]):
        self.results = results

    def generate_html_report(self, output_path: Path) -> None:
        """Generate HTML report."""
        from jinja2 import Template  # [web:683]

        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ALPHA-PRIME Stress Test Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2E86AB; }
        .metric { display: inline-block; margin: 10px; padding: 12px 16px;
                  background: #f5f5f5; border-radius: 4px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
        th { background-color: #2E86AB; color: white; }
        .scenario-block { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>ALPHA-PRIME v2.0 Stress Test Report</h1>
    <p>Generated: {{ timestamp }}</p>

    <h2>Summary</h2>
    {% for r in results %}
    <div class="scenario-block">
        <h3>{{ r.scenario }}</h3>
        <div class="metric"><strong>Throughput:</strong> {{ "%.2f"|format(r.operations_per_second) }} ops/sec</div>
        <div class="metric"><strong>Success Rate:</strong>
            {% if r.total_operations > 0 %}
            {{ "%.2f"|format(r.successful_operations / r.total_operations * 100) }}%
            {% else %}n/a{% endif %}
        </div>
        <div class="metric"><strong>P95 Latency:</strong> {{ "%.2f"|format(r.p95_latency_ms) }} ms</div>
        <div class="metric"><strong>Peak Memory:</strong> {{ "%.1f"|format(r.peak_memory_mb) }} MB</div>
        <div class="metric"><strong>Error Rate:</strong> {{ "%.2f"|format(r.error_rate * 100) }}%</div>
    </div>
    {% endfor %}

    <h2>Detailed Metrics</h2>
    <table>
        <tr>
            <th>Scenario</th>
            <th>Duration (s)</th>
            <th>Operations</th>
            <th>Success</th>
            <th>Fail</th>
            <th>Throughput (ops/s)</th>
            <th>P50 (ms)</th>
            <th>P95 (ms)</th>
            <th>P99 (ms)</th>
            <th>Peak CPU (%)</th>
            <th>Peak Mem (MB)</th>
        </tr>
        {% for r in results %}
        <tr>
            <td>{{ r.scenario }}</td>
            <td>{{ "%.2f"|format(r.duration) }}</td>
            <td>{{ r.total_operations }}</td>
            <td>{{ r.successful_operations }}</td>
            <td>{{ r.failed_operations }}</td>
            <td>{{ "%.2f"|format(r.operations_per_second) }}</td>
            <td>{{ "%.2f"|format(r.p50_latency_ms) }}</td>
            <td>{{ "%.2f"|format(r.p95_latency_ms) }}</td>
            <td>{{ "%.2f"|format(r.p99_latency_ms) }}</td>
            <td>{{ "%.1f"|format(r.peak_cpu_percent) }}</td>
            <td>{{ "%.1f"|format(r.peak_memory_mb) }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""
        template = Template(template_str)
        html = template.render(
            timestamp=datetime.utcnow().isoformat(),
            results=self.results,
        )
        output_path.write_text(html, encoding="utf-8")

    def print_summary(self, logger: logging.Logger) -> None:
        """Log summary of stress tests."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST SUMMARY")
        logger.info("=" * 70)
        for r in self.results:
            logger.info("\n%s", r.scenario.upper())
            logger.info("-" * 70)
            logger.info("  Duration: %.2fs", r.duration)
            logger.info("  Operations: %d (success=%d, fail=%d)", r.total_operations, r.successful_operations, r.failed_operations)
            logger.info("  Throughput: %.2f ops/sec", r.operations_per_second)
            logger.info("  Latency: avg=%.2f ms p95=%.2f ms p99=%.2f ms", r.avg_latency_ms, r.p95_latency_ms, r.p99_latency_ms)
            logger.info("  Resources: peak CPU=%.1f%% peak Mem=%.1f MB", r.peak_cpu_percent, r.peak_memory_mb)
            logger.info("  Error rate: %.2f%%", r.error_rate * 100.0)


# === MAIN EXECUTION ===
async def main() -> int:
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logger = setup_logger("stress_test", level=log_level)

    try:
        base_config = build_scenario_config(args)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if args.all:
            scenarios = list(TEST_SCENARIOS.keys())
        elif args.scenario:
            scenarios = [args.scenario]
        else:
            logger.error("Specify --scenario or --all")
            return 1

        if args.profile or args.profile_memory:
            tracemalloc.start()  # [web:702]

        all_results: List[StressTestMetrics] = []

        for scenario in scenarios:
            logger.info("\n%s", "=" * 70)
            logger.info("Running scenario: %s", scenario)
            logger.info("%s\n", "=" * 70)

            scenario_cfg = {**TEST_SCENARIOS[scenario], **base_config}

            monitor = ResourceMonitor()
            monitor_task = asyncio.create_task(
                monitor.monitor(
                    interval=int(args.monitor_interval),
                    realtime=bool(args.realtime_stats),
                    logger=logger if args.realtime_stats else None,
                )
            )

            if scenario == "high_load":
                tester = HighLoadTester(scenario_cfg, logger, monitor)
            elif scenario == "concurrent_trades":
                tester = ConcurrentTradesTester(scenario_cfg, logger, monitor)
            elif scenario == "database_load":
                tester = DatabaseLoadTester(scenario_cfg, logger, monitor)
            elif scenario == "api_burst":
                tester = APIBurstTester(scenario_cfg, logger, monitor)
            elif scenario == "memory_leak":
                tester = MemoryProfiler(scenario_cfg, logger, monitor)
            elif scenario == "cache_storm":
                tester = CacheStormTester(scenario_cfg, logger, monitor)
            elif scenario == "websocket_flood":
                tester = WebSocketFloodTester(scenario_cfg, logger, monitor)
            elif scenario == "error_injection":
                tester = ErrorInjectionTester(scenario_cfg, logger, monitor)
            else:
                logger.warning("No tester implemented for scenario '%s', skipping", scenario)
                monitor_task.cancel()
                continue

            try:
                result = await tester.run()
            finally:
                monitor_task.cancel()
                with contextlib_suppress(asyncio.CancelledError):
                    await monitor_task

            all_results.append(result)
            passed = validate_thresholds(result, base_config, logger)
            if not passed:
                logger.warning("Scenario '%s' did not meet thresholds", scenario)

        reporter = StressTestReporter(all_results)
        reporter.print_summary(logger)

        if args.report:
            report_path = args.output
            if not report_path:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                report_path = RESULTS_DIR / f"stress_test_report_{ts}.html"
            reporter.generate_html_report(report_path)
            logger.info("HTML report written to %s", report_path)

        if args.json:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            json_path = args.output if args.output and args.output.suffix == ".json" else RESULTS_DIR / f"stress_test_results_{ts}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
            logger.info("JSON results written to %s", json_path)

        if args.profile or args.profile_memory:
            tracemalloc.stop()

        logger.info("\nStress testing completed")
        return 0

    except Exception as exc:
        logger.error("Stress test failed: %s", exc, exc_info=True)
        return 1


class contextlib_suppress:
    """Minimal context manager for suppressing specific exceptions."""

    def __init__(self, *exceptions: Any):
        self._exceptions = exceptions

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, self._exceptions)


if __name__ == "__main__":
    import asyncio as _asyncio

    sys.exit(_asyncio.run(main()))
