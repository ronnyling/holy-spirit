"""Tests for retrieval cache (bounded LRU with TTL)."""

from __future__ import annotations

import time

import pytest

from knowledge_engine.cache import RetrievalCache


class TestRetrievalCacheInit:
    """Verify cache construction and from_env."""

    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("KE_CACHE_MAX_SIZE", raising=False)
        monkeypatch.delenv("KE_CACHE_TTL_SECONDS", raising=False)
        cache = RetrievalCache.from_env()
        assert cache._max_size == 1000
        assert cache._ttl == 3600.0

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("KE_CACHE_MAX_SIZE", "500")
        monkeypatch.setenv("KE_CACHE_TTL_SECONDS", "600")
        cache = RetrievalCache.from_env()
        assert cache._max_size == 500
        assert cache._ttl == 600.0


class TestRetrievalCacheGetSet:
    """Verify basic get/set operations."""

    def test_set_and_get(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        cache.set("key1", [1.0, 2.0])
        assert cache.get("key1") == [1.0, 2.0]

    def test_get_missing_returns_none(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=0.01)
        cache.set("key1", "value1")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        cache = RetrievalCache(max_size=2, ttl_seconds=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")  # evicts k1
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    def test_lru_access_refreshes(self):
        cache = RetrievalCache(max_size=2, ttl_seconds=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        _ = cache.get("k1")  # refresh k1
        cache.set("k3", "v3")  # evicts k2 (k1 was recently accessed)
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None

    def test_overwrite_existing_key(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        cache.set("k1", "v1")
        cache.set("k1", "v2")
        assert cache.get("k1") == "v2"
        assert cache.stats["size"] == 1


class TestRetrievalCacheInvalidation:
    """Verify cache invalidation."""

    def test_invalidate_prefix(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        cache.set("embed:model1", [1.0])
        cache.set("embed:model2", [2.0])
        cache.set("llm:query1", [3.0])
        removed = cache.invalidate("embed:")
        assert removed == 2
        assert cache.get("embed:model1") is None
        assert cache.get("llm:query1") == [3.0]

    def test_clear(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.stats["size"] == 0


class TestRetrievalCacheStats:
    """Verify cache statistics."""

    def test_hit_miss_tracking(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        cache.set("k1", "v1")
        _ = cache.get("k1")  # hit
        _ = cache.get("k2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_empty_cache(self):
        cache = RetrievalCache(max_size=10, ttl_seconds=60)
        stats = cache.stats
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0


class TestRetrievalCacheMakeKey:
    """Verify deterministic key generation."""

    def test_deterministic(self):
        key1 = RetrievalCache.make_key("embed", "model", "text")
        key2 = RetrievalCache.make_key("embed", "model", "text")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        key1 = RetrievalCache.make_key("embed", "model", "text1")
        key2 = RetrievalCache.make_key("embed", "model", "text2")
        assert key1 != key2

    def test_key_length(self):
        key = RetrievalCache.make_key("test")
        assert len(key) == 32  # SHA-256 truncated to 32 hex chars
