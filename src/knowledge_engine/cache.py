"""Bounded LRU cache with TTL for LLM and embedding calls.

Design:
    - In-memory LRU cache with configurable max size and TTL
    - Thread-safe (concurrent access from ingest pipeline)
    - Cache key = hash of (system_prompt, user_prompt) for LLM, hash of text for embeddings
    - No cache for ingestion-path calls (only query-path)
    - Cache is per-process, not shared (appropriate for single-server deployment)

Configuration:
    KE_CACHE_MAX_SIZE (default 1000 entries)
    KE_CACHE_TTL_SECONDS (default 3600 = 1 hour)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any


class RetrievalCache:
    """Bounded LRU cache with TTL for retrieval-path results.

    Thread-safe via a lock. Evicts least-recently-used entries when at capacity.
    Entries expire after TTL seconds.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @classmethod
    def from_env(cls) -> RetrievalCache:
        """Construct from environment variables."""
        max_size = int(os.environ.get("KE_CACHE_MAX_SIZE", "1000"))
        ttl = float(os.environ.get("KE_CACHE_TTL_SECONDS", "3600"))
        return cls(max_size=max_size, ttl_seconds=ttl)

    def get(self, key: str) -> Any | None:
        """Return cached value if present and not expired. Thread-safe."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            ts, value = self._cache[key]
            if time.monotonic() - ts > self._ttl:
                # Expired — remove and treat as miss
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Cache value with current timestamp. Evicts LRU if at capacity. Thread-safe."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.monotonic(), value)

            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, prefix: str) -> int:
        """Remove all entries whose key starts with prefix. Returns count removed."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(
                    self._hits / (self._hits + self._misses), 4
                ) if (self._hits + self._misses) > 0 else 0.0,
            }

    @staticmethod
    def make_key(*parts: str) -> str:
        """Deterministic cache key from parts. Uses SHA-256 for compactness."""
        combined = "\x00".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
