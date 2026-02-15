"""
============================================================
ALPHA-PRIME v2.0 - Cache Manager
============================================================

High-performance caching infrastructure for data-intensive
trading workflows.

Goals:
- 10–100x acceleration for repeated I/O and computation.
- Multi-backend: memory, disk, Redis (where available).
- Smart serialization for pandas/numpy/models.
- Strict TTL and size limits with background eviction.
- Crash safety via write-ahead logging (WAL, best-effort).
- Metrics and observability hooks for production ops.

This module is intentionally self-contained and conservative:
- When optional dependencies (diskcache, redis, joblib, zstd)
  are unavailable, it degrades gracefully to a memory backend
  and logs warnings rather than failing.
- WAL is implemented as a lightweight JSONL log for disk
  backend only.

============================================================
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Literal, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

from config import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()

try:  # Optional backends
    import diskcache  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    diskcache = None

try:
    import redis  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    redis = None

try:
    import joblib  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    joblib = None

try:
    import zstandard as zstd  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    zstd = None


BackendType = Literal["memory", "disk", "redis"]
EvictionPolicy = Literal["lru", "lfu", "fifo"]


# ──────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────


@dataclass
class CacheConfig:
    """
    Configuration for CacheManager.

    Attributes:
        backend: 'memory', 'disk', or 'redis'.
        cache_dir: Base directory for disk cache and WAL.
        max_size_gb: Maximum on-disk cache size before eviction.
        default_ttl_hours: Default TTL for entries.
        redis_url: Redis connection URL, if using redis backend.
        eviction_policy: 'lru', 'lfu', or 'fifo' (used where supported).
        serialize_pandas: Use Parquet for pandas objects.
        background_threads: Number of worker threads for maintenance.
    """

    backend: BackendType = "disk"
    cache_dir: Path = Path(os.path.expanduser("~/.alpha_prime/cache"))
    max_size_gb: float = 10.0
    default_ttl_hours: float = 24.0
    redis_url: Optional[str] = None
    eviction_policy: EvictionPolicy = "lru"
    serialize_pandas: bool = True
    background_threads: int = 4


@dataclass
class CacheEntry:
    """
    Internal cache entry metadata (for memory backend and stats).

    Attributes:
        key: Full key (including namespace).
        value: Serialized bytes (or raw object for memory backend).
        created: Creation timestamp.
        expires: Expiration timestamp (None for no TTL).
        access_count: Access count (for LFU/LRU).
        size_bytes: Approximate size in bytes.
        namespace: Logical namespace (prefix before first ':').
    """

    key: str
    value: Any
    created: datetime
    expires: Optional[datetime]
    access_count: int
    size_bytes: int
    namespace: str


@dataclass
class CacheStats:
    """
    Summary statistics for cache health.

    Attributes:
        total_items: Number of items.
        total_size_mb: Approximate size in MB.
        hit_rate: Hit rate over recent window.
        evictions: Total evictions since start.
        invalidations: Total invalidations since start.
        oldest_item_age_hours: Age of oldest item.
        backend: Backend name.
        active_namespaces: List of active namespaces.
    """

    total_items: int
    total_size_mb: float
    hit_rate: float
    evictions: int
    invalidations: int
    oldest_item_age_hours: float
    backend: str
    active_namespaces: List[str]


# ──────────────────────────────────────────────────────────
# SERIALIZATION ENGINE
# ──────────────────────────────────────────────────────────


class Serializer:
    """
    Smart serializer for cache values.

    Supported types:
        - pandas.DataFrame / Series -> Parquet bytes.
        - numpy.ndarray -> Zstandard-compressed npy bytes if zstd, else np.save.
        - sklearn / models -> joblib with compression (if available).
        - dict/list/str/int/float -> JSON.
        - Fallback: pickle.

    Encoding format:
        {"type": "<dtype>", "payload": <bytes or json-serializable>}
    """

    @staticmethod
    def _to_bytes(obj: Any, config: CacheConfig) -> Tuple[str, bytes]:
        import pickle
        from io import BytesIO

        if isinstance(obj, (pd.DataFrame, pd.Series)) and config.serialize_pandas:
            buf = BytesIO()
            obj.to_parquet(buf, index=True)
            return ("pandas_parquet", buf.getvalue())

        if isinstance(obj, np.ndarray):
            buf = BytesIO()
            np.save(buf, obj)
            raw = buf.getvalue()
            if zstd is not None:
                cctx = zstd.ZstdCompressor(level=3)
                return ("numpy_zstd", cctx.compress(raw))
            return ("numpy_npy", raw)

        if joblib is not None:
            try:
                buf = BytesIO()
                joblib.dump(obj, buf, compress=("zlib", 3))
                return ("joblib", buf.getvalue())
            except Exception:  # noqa: BLE001
                pass

        if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
            try:
                return ("json", json.dumps(obj).encode("utf-8"))
            except Exception:  # noqa: BLE001
                pass

        try:
            return ("pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:  # noqa: BLE001
            logger.error("Serialization failed for object %r: %s", type(obj), exc)
            raise

    @staticmethod
    def _from_bytes(type_tag: str, payload: bytes, config: CacheConfig) -> Any:
        import pickle
        from io import BytesIO

        if type_tag == "pandas_parquet":
            buf = BytesIO(payload)
            return pd.read_parquet(buf)

        if type_tag in ("numpy_zstd", "numpy_npy"):
            raw = payload
            if type_tag == "numpy_zstd" and zstd is not None:
                dctx = zstd.ZstdDecompressor()
                raw = dctx.decompress(payload)
            buf = BytesIO(raw)
            return np.load(buf, allow_pickle=False)

        if type_tag == "joblib" and joblib is not None:
            buf = BytesIO(payload)
            return joblib.load(buf)

        if type_tag == "json":
            return json.loads(payload.decode("utf-8"))

        if type_tag == "pickle":
            return pickle.loads(payload)

        logger.warning("Unknown serialization type_tag=%s; returning raw bytes.", type_tag)
        return payload

    @classmethod
    def dumps(cls, obj: Any, config: CacheConfig) -> bytes:
        type_tag, payload = cls._to_bytes(obj, config)
        header = json.dumps({"t": type_tag}).encode("utf-8")
        return len(header).to_bytes(4, "big") + header + payload

    @classmethod
    def loads(cls, blob: bytes, config: CacheConfig) -> Any:
        if not blob:
            return None
        header_len = int.from_bytes(blob[:4], "big")
        header = json.loads(blob[4 : 4 + header_len].decode("utf-8"))
        payload = blob[4 + header_len :]
        return cls._from_bytes(header["t"], payload, config)


# ──────────────────────────────────────────────────────────
# BACKEND ABSTRACTIONS
# ──────────────────────────────────────────────────────────


class BaseBackend:
    """Abstract backend interface."""

    def get(self, key: str) -> Optional[bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int]) -> None:  # pragma: no cover
        raise NotImplementedError

    def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def keys(self, pattern: str = "*") -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def approximate_size_bytes(self) -> int:
        return 0


class MemoryBackend(BaseBackend):
    """Simple in-process memory cache with TTL."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[bytes, Optional[float]]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            val = self._store.get(key)
            if val is None:
                return None
            blob, expires_ts = val
            if expires_ts is not None and time.time() > expires_ts:
                self._store.pop(key, None)
                return None
            return blob

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int]) -> None:
        expires_ts = time.time() + ttl_seconds if ttl_seconds else None
        with self._lock:
            self._store[key] = (value, expires_ts)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def keys(self, pattern: str = "*") -> List[str]:
        with self._lock:
            return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]

    def approximate_size_bytes(self) -> int:
        with self._lock:
            return sum(len(v[0]) for v in self._store.values())


class DiskBackend(BaseBackend):
    """Disk-based backend using diskcache if available; shallow wrapper."""

    def __init__(self, directory: Path, eviction_policy: EvictionPolicy, size_limit_bytes: int) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size_limit_bytes = size_limit_bytes
        self.eviction_policy = eviction_policy
        if diskcache is None:
            logger.warning("diskcache not available; DiskBackend will behave as MemoryBackend.")
            self._fallback = MemoryBackend()
            self._cache = None
        else:
            self._fallback = None
            self._cache = diskcache.Cache(str(directory), size_limit=size_limit_bytes)

    def get(self, key: str) -> Optional[bytes]:
        if self._cache is None:
            return self._fallback.get(key)  # type: ignore[union-attr]
        blob = self._cache.get(key, default=None)
        return blob

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int]) -> None:
        if self._cache is None:
            self._fallback.set(key, value, ttl_seconds)  # type: ignore[union-attr]
            return
        expire = ttl_seconds or 0
        self._cache.set(key, value, expire=expire)

    def delete(self, key: str) -> None:
        if self._cache is None:
            self._fallback.delete(key)  # type: ignore[union-attr]
            return
        try:
            del self._cache[key]
        except KeyError:
            pass

    def keys(self, pattern: str = "*") -> List[str]:
        if self._cache is None:
            return self._fallback.keys(pattern)  # type: ignore[union-attr]
        return [k for k in self._cache.iterkeys() if fnmatch.fnmatch(k, pattern)]

    def approximate_size_bytes(self) -> int:
        if self._cache is None:
            return self._fallback.approximate_size_bytes()  # type: ignore[union-attr]
        return int(self._cache.volume())


class RedisBackend(BaseBackend):
    """Redis backend with TTL support."""

    def __init__(self, url: str) -> None:
        if redis is None:
            raise RuntimeError("redis backend requested but redis-py is not installed.")
        self._client = redis.Redis.from_url(url)

    def get(self, key: str) -> Optional[bytes]:
        blob = self._client.get(key)
        return blob

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int]) -> None:
        if ttl_seconds:
            self._client.set(key, value, ex=ttl_seconds)
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def keys(self, pattern: str = "*") -> List[str]:
        return [k.decode("utf-8") for k in self._client.keys(pattern)]

    def approximate_size_bytes(self) -> int:
        try:
            info = self._client.info()
            return int(info.get("used_memory", 0))
        except Exception:  # noqa: BLE001
            return 0


# ──────────────────────────────────────────────────────────
# WAL (WRITE-AHEAD LOG) FOR DISK BACKEND
# ──────────────────────────────────────────────────────────


class WriteAheadLog:
    """
    Simple JSONL-based write-ahead log for cache mutations (disk backend).

    For each mutation, we append an entry:
        {"ts": "...", "op": "set|delete", "key": "...", "ttl": <seconds>}

    On startup we can replay recent WAL (e.g. last 10 minutes) to
    restore consistency after an unclean shutdown.
    """

    def __init__(self, wal_dir: Path) -> None:
        self.wal_dir = wal_dir
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._current_path = self.wal_dir / "wal_current.jsonl"

    def append(self, op: str, key: str, ttl_seconds: Optional[int]) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": op,
            "key": key,
            "ttl": ttl_seconds,
        }
        line = json.dumps(entry)
        with self._lock, open(self._current_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def replay(self, backend: BaseBackend, window_minutes: int = 10) -> None:
        if not self._current_path.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        try:
            with open(self._current_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = datetime.fromisoformat(entry.get("ts"))
                    if ts < cutoff:
                        continue
                    op = entry.get("op")
                    key = entry.get("key")
                    ttl = entry.get("ttl")
                    if op == "delete":
                        backend.delete(key)
                    elif op == "set":
                        # WAL doesn't store values; used mainly for delete ordering.
                        # In a full implementation, this would store value refs.
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("WAL replay failed: %s", exc)


# ──────────────────────────────────────────────────────────
# CACHE MANAGER
# ──────────────────────────────────────────────────────────


class CacheManager:
    """
    High-level cache manager with multi-backend support, smart serialization,
    pattern invalidation, warming, and stats.

    Typical usage:
        config = CacheConfig()
        cache = CacheManager(config)

        cache.set("features:AAPL:1d:20260109", df, ttl_hours=24)
        df2 = cache.get("features:AAPL:1d:20260109")

    Decorator:
        @cache.cached(ttl=1.0, namespace="features")
        def expensive(symbol):
            ...
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.backend = self._init_backend(config)
        self.serializer = Serializer
        self._lock = threading.RLock()
        self._namespace_locks: Dict[str, threading.RLock] = {}
        self._hit_count: Deque[bool] = deque(maxlen=1000)
        self._evictions = 0
        self._invalidations = 0
        self._entries: Dict[str, CacheEntry] = {}

        self._wal: Optional[WriteAheadLog] = None
        if config.backend == "disk":
            self._wal = WriteAheadLog(config.cache_dir / "wal")
            self._wal.replay(self.backend)

        self._executor = ThreadPoolExecutor(max_workers=config.background_threads)
        self._maintenance_stop = threading.Event()
        self._maintenance_thread = threading.Thread(
            target=self._background_maintenance_loop, name="cache-maintenance", daemon=True
        )
        self._maintenance_thread.start()

    # ---- backend init -------------------------------------------------------

    def _init_backend(self, config: CacheConfig) -> BaseBackend:
        if config.backend == "memory":
            logger.info("Initialising memory cache backend.")
            return MemoryBackend()
        if config.backend == "disk":
            size_bytes = int(config.max_size_gb * (1024**3))
            logger.info(
                "Initialising disk cache backend at %s (limit %.1f GB).",
                config.cache_dir,
                config.max_size_gb,
            )
            return DiskBackend(config.cache_dir, config.eviction_policy, size_bytes)
        if config.backend == "redis":
            if config.redis_url is None:
                raise ValueError("redis_url must be provided for redis backend.")
            logger.info("Initialising Redis cache backend at %s.", config.redis_url)
            return RedisBackend(config.redis_url)
        raise ValueError(f"Unknown backend: {config.backend}")

    # ---- namespace helpers --------------------------------------------------

    @staticmethod
    def _namespace_for_key(key: str) -> str:
        return key.split(":", 1)[0] if ":" in key else "default"

    def _get_namespace_lock(self, namespace: str) -> threading.RLock:
        with self._lock:
            if namespace not in self._namespace_locks:
                self._namespace_locks[namespace] = threading.RLock()
            return self._namespace_locks[namespace]

    # ---- core API -----------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """
        Fetch object from cache.

        Returns:
            Deserialized value or None if missing/expired.
        """
        ns = self._namespace_for_key(key)
        lock = self._get_namespace_lock(ns)
        with lock:
            blob = self.backend.get(key)
            hit = blob is not None
            self._hit_count.append(hit)
            if not hit:
                logger.debug(
                    '{"event": "cache_miss", "key": "%s", "namespace": "%s"}',
                    key,
                    ns,
                )
                return None
            try:
                value = self.serializer.loads(blob, self.config)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Deserialization failed for key %s: %s", key, exc)
                return None

            entry = self._entries.get(key)
            if entry:
                entry.access_count += 1
            return value

    def set(self, key: str, value: Any, ttl_hours: float | None = None) -> None:
        """
        Store object in cache.

        Args:
            key: Cache key (use namespaced patterns).
            value: Object to store.
            ttl_hours: Time-to-live in hours; defaults to config.default_ttl_hours.
        """
        ns = self._namespace_for_key(key)
        lock = self._get_namespace_lock(ns)
        ttl_hours = ttl_hours if ttl_hours is not None else self.config.default_ttl_hours
        ttl_seconds = int(ttl_hours * 3600) if ttl_hours > 0 else None

        with lock:
            try:
                blob = self.serializer.dumps(value, self.config)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to serialize value for key %s: %s", key, exc)
                return

            if self._wal is not None:
                self._wal.append("set", key, ttl_seconds)

            self.backend.set(key, blob, ttl_seconds=ttl_seconds)

            size_bytes = len(blob)
            entry = CacheEntry(
                key=key,
                value=None if self.config.backend != "memory" else blob,
                created=datetime.now(timezone.utc),
                expires=(
                    datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                    if ttl_seconds
                    else None
                ),
                access_count=0,
                size_bytes=size_bytes,
                namespace=ns,
            )
            self._entries[key] = entry

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.

        Examples:
            invalidate_pattern("features:AAPL:*")
            invalidate_pattern("model:*")
        """
        keys = self.backend.keys(pattern)
        count = 0
        for k in keys:
            ns = self._namespace_for_key(k)
            lock = self._get_namespace_lock(ns)
            with lock:
                if self._wal is not None:
                    self._wal.append("delete", k, None)
                self.backend.delete(k)
                self._entries.pop(k, None)
                count += 1
        self._invalidations += count
        logger.info("Invalidated %d keys matching pattern %s.", count, pattern)
        return count

    def warm_cache(self, patterns: List[str]) -> None:
        """
        Warm cache by touching keys/patterns.

        In practice, this will call `get` for each concrete key
        returned by backend.keys(pattern).
        """
        logger.info("Starting cache warm for %d patterns.", len(patterns))
        futures = []
        for pattern in patterns:
            keys = self.backend.keys(pattern)
            for k in keys:
                futures.append(self._executor.submit(self.get, k))
        for f in as_completed(futures):
            _ = f.result()
        logger.info("Cache warm complete.")

    # ---- stats & metrics ----------------------------------------------------

    def cache_stats(self) -> CacheStats:
        """Compute cache statistics snapshot."""
        total_items = len(self._entries)
        total_size_bytes = self.backend.approximate_size_bytes()
        total_size_mb = total_size_bytes / (1024 * 1024) if total_size_bytes else 0.0
        hit_rate = (
            sum(1 for h in self._hit_count if h) / len(self._hit_count)
            if self._hit_count
            else 0.0
        )
        oldest = min((e.created for e in self._entries.values()), default=None)
        oldest_age_hours = (
            (datetime.now(timezone.utc) - oldest).total_seconds() / 3600.0 if oldest else 0.0
        )
        namespaces = sorted({e.namespace for e in self._entries.values()})

        return CacheStats(
            total_items=total_items,
            total_size_mb=total_size_mb,
            hit_rate=hit_rate * 100.0,
            evictions=self._evictions,
            invalidations=self._invalidations,
            oldest_item_age_hours=oldest_age_hours,
            backend=self.config.backend,
            active_namespaces=namespaces,
        )

    # ─────────────────────────────────────────────────────
    # CONTEXT MANAGERS & DECORATORS
    # ─────────────────────────────────────────────────────

    @contextlib.contextmanager
    def temp_namespace(self, namespace: str, ttl_hours: float = 1.0) -> Any:
        """
        Temporary namespace context.

        Any keys written within this context under that namespace
        can be invalidated after exiting.

        Example:
            with cache.temp_namespace("research_session_123"):
                cache.set("research_session_123:temp_df", df)
        """
        try:
            yield
        finally:
            pattern = f"{namespace}:*"
            self.invalidate_pattern(pattern)

    def cached(
        self,
        ttl: float = 1.0,
        namespace: str = "default",
        key_func: Optional[Callable[..., str]] = None,
    ) -> Callable:
        """
        Decorator to cache function results.

        Args:
            ttl: TTL in hours.
            namespace: Namespace prefix for keys.
            key_func: Optional function(*args, **kwargs) -> str to generate key.

        Usage:
            @cache.cached(ttl=2.0, namespace="features")
            def expensive(symbol, freq):
                ...
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if key_func is not None:
                    suffix = key_func(*args, **kwargs)
                else:
                    suffix = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                key = f"{namespace}:{suffix}"
                val = self.get(key)
                if val is not None:
                    return val
                val = func(*args, **kwargs)
                self.set(key, val, ttl_hours=ttl)
                return val

            return wrapper

        return decorator

    # ─────────────────────────────────────────────────────
    # BACKGROUND MAINTENANCE
    # ─────────────────────────────────────────────────────

    def _background_maintenance_loop(self) -> None:
        """
        Periodic maintenance loop for:
            - TTL expiration.
            - Eviction based on approximate size and policy.
            - Stats/metrics aggregation (export hook).
        """
        while not self._maintenance_stop.is_set():
            try:
                self._cleanup_expired()
                self._evict_if_needed()
                self._export_metrics()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache maintenance loop error: %s", exc)
            self._maintenance_stop.wait(30.0)

    def _cleanup_expired(self) -> None:
        """Remove expired entries for memory/disk stats."""
        now = datetime.now(timezone.utc)
        expired_keys: List[str] = []
        for key, entry in list(self._entries.items()):
            if entry.expires is not None and entry.expires < now:
                expired_keys.append(key)
        for key in expired_keys:
            ns = self._namespace_for_key(key)
            lock = self._get_namespace_lock(ns)
            with lock:
                self.backend.delete(key)
                self._entries.pop(key, None)

    def _evict_if_needed(self) -> None:
        """Evict entries based on approximate size > limit (disk backend mainly)."""
        limit_bytes = int(self.config.max_size_gb * (1024**3))
        if limit_bytes <= 0:
            return
        approx_size = self.backend.approximate_size_bytes()
        if approx_size <= limit_bytes:
            return

        logger.warning(
            "Cache size %.2f MB exceeds limit %.2f MB; starting eviction.",
            approx_size / (1024 * 1024),
            limit_bytes / (1024 * 1024),
        )
        entries = list(self._entries.values())
        if self.config.eviction_policy == "fifo":
            entries.sort(key=lambda e: e.created)
        elif self.config.eviction_policy == "lfu":
            entries.sort(key=lambda e: e.access_count)
        else:  # lru
            entries.sort(key=lambda e: e.access_count)

        target = approx_size - limit_bytes
        freed = 0
        for e in entries:
            if freed >= target:
                break
            ns = self._namespace_for_key(e.key)
            lock = self._get_namespace_lock(ns)
            with lock:
                self.backend.delete(e.key)
                self._entries.pop(e.key, None)
                freed += e.size_bytes
                self._evictions += 1

        logger.info(
            "Eviction completed; freed %.2f MB (policy=%s).",
            freed / (1024 * 1024),
            self.config.eviction_policy,
        )

    def _export_metrics(self) -> None:
        """
        Hook for metrics export.

        Here we just log; in production, wire to Prometheus.
        Metrics names:
            alpha_cache_hits_total
            alpha_cache_misses_total
            alpha_cache_evictions_total
            alpha_cache_size_bytes
            alpha_cache_hit_rate
        """
        stats = self.cache_stats()
        logger.debug(
            "CACHE_METRICS %s",
            {
                "alpha_cache_size_bytes": stats.total_size_mb * 1024 * 1024,
                "alpha_cache_hit_rate": stats.hit_rate,
                "alpha_cache_evictions_total": stats.evictions,
                "alpha_cache_backend": stats.backend,
            },
        )

    # ─────────────────────────────────────────────────────
    # MAINTENANCE MODE / SHUTDOWN
    # ─────────────────────────────────────────────────────

    def freeze_writes(self) -> None:
        """
        Placeholder for maintenance mode (could wrap set/invalidate).

        In this simplified version, we just log; callers can co-ordinate
        freeze by not calling set/invalidate during backups.
        """
        logger.warning("Cache writes freeze requested (no-op placeholder).")

    def maintenance_mode(self, enable: bool = True) -> None:
        """Placeholder toggle for future use."""
        logger.info("Cache maintenance_mode set to %s.", enable)

    def shutdown(self) -> None:
        """Shutdown background threads cleanly."""
        self._maintenance_stop.set()
        self._executor.shutdown(wait=False)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def _init_default_manager() -> CacheManager:
    cfg = CacheConfig(
        backend=getattr(settings, "cache_backend", "disk"),
        cache_dir=Path(getattr(settings, "cache_dir", os.path.expanduser("~/.alpha_prime/cache"))),
        max_size_gb=float(getattr(settings, "cache_max_size_gb", 10.0)),
    )
    return CacheManager(cfg)


def _print_status(cm: CacheManager) -> None:
    stats = cm.cache_stats()
    print(
        f"CACHE STATUS ({stats.backend.upper()}: {cm.config.cache_dir})"
        if hasattr(cm.config, "cache_dir")
        else f"CACHE STATUS ({stats.backend.upper()})"
    )
    print(
        f"Items: {stats.total_items:,} | Size: {stats.total_size_mb:.1f}MB/"
        f"{cm.config.max_size_gb:.1f}GB | Hit Rate: {stats.hit_rate:.1f}%"
    )
    print(f"Evictions: {stats.evictions} | Invalidations: {stats.invalidations}")
    if stats.active_namespaces:
        ns_counts: Dict[str, int] = {}
        for e in cm._entries.values():
            ns_counts[e.namespace] = ns_counts.get(e.namespace, 0) + 1
        top = sorted(ns_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        formatted = ", ".join(f"{n}({c})" for n, c in top)
        print(f"Top Namespaces: {formatted}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME v2.0 - Cache Manager CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show cache status summary.")

    clear_p = sub.add_parser("clear", help="Clear keys matching a pattern.")
    clear_p.add_argument("--pattern", type=str, required=True)

    warm_p = sub.add_parser("warm", help="Warm cache from JSON config.")
    warm_p.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    cm = _init_default_manager()

    if args.command == "status":
        _print_status(cm)
    elif args.command == "clear":
        count = cm.invalidate_pattern(args.pattern)
        print(f"Cleared {count} items matching \"{args.pattern}\"")
    elif args.command == "warm":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        patterns = cfg.get("patterns", [])
        print(f"Warming {len(patterns)} patterns...")
        start = time.time()
        cm.warm_cache(patterns)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s.")
    cm.shutdown()


if __name__ == "__main__":
    main_cli()
