"""Tests for caching system."""

import time

from houyi.verification.cache import (
    CacheEntry,
    CacheStats,
    ConstraintSolverCache,
    LRUCache,
    VerificationResultCache,
    clear_all_caches,
    get_constraint_cache,
    get_verification_cache,
)


class TestCacheEntry:
    """Test CacheEntry functionality."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert not entry.is_expired()

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time() - 100,  # 100 seconds ago
            last_accessed=time.time(),
            ttl=10.0,  # 10 second TTL
        )

        assert entry.is_expired()

    def test_cache_entry_no_expiration(self):
        """Test cache entry without TTL never expires."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time() - 1000,
            last_accessed=time.time(),
            ttl=None,
        )

        assert not entry.is_expired()

    def test_cache_entry_touch(self):
        """Test updating access metadata."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
        )

        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestCacheStats:
    """Test CacheStats functionality."""

    def test_cache_stats_initialization(self):
        """Test cache stats initialization."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.hit_rate == 0.0

    def test_cache_stats_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats()
        stats.hits = 7
        stats.misses = 3

        assert stats.hit_rate == 0.7

    def test_cache_stats_reset(self):
        """Test resetting statistics."""
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 5
        stats.evictions = 2

        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


class TestLRUCache:
    """Test LRUCache functionality."""

    def test_lru_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)

        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Miss
        assert cache.get("nonexistent") is None

    def test_lru_cache_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=3)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Add one more - should evict key1 (oldest)
        cache.put("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_cache_access_updates_order(self):
        """Test that accessing an entry updates its position."""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add new entry - should evict key2 (now oldest)
        cache.put("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted

    def test_lru_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)

        assert cache.get("key1") is None  # Expired

    def test_lru_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = LRUCache(max_size=10, default_ttl=0.1)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Wait for expiration
        time.sleep(0.15)

        # Cleanup
        removed = cache.cleanup_expired()

        assert removed == 3
        assert cache.get("key1") is None

    def test_lru_cache_invalidate(self):
        """Test invalidating specific entries."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        assert cache.invalidate("nonexistent") is False

    def test_lru_cache_clear(self):
        """Test clearing all entries."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Hits
        cache.get("key1")
        cache.get("key1")

        # Misses
        cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["current_size"] == 2
        assert stats["max_size"] == 3


class TestConstraintSolverCache:
    """Test ConstraintSolverCache functionality."""

    def test_constraint_cache_basic(self):
        """Test basic constraint caching."""
        cache = ConstraintSolverCache(max_size=10)

        variables = {"x": "Int", "y": "Int"}
        constraints = ["x > 0", "y < 10"]

        # Cache miss
        result = cache.get_result(variables, constraints)
        assert result is None

        # Cache result
        cache.put_result(variables, constraints, True, [])

        # Cache hit
        result = cache.get_result(variables, constraints)
        assert result == (True, [])

    def test_constraint_cache_different_constraints(self):
        """Test that different constraints produce different cache keys."""
        cache = ConstraintSolverCache(max_size=10)

        variables = {"x": "Int"}
        constraints1 = ["x > 0"]
        constraints2 = ["x < 0"]

        cache.put_result(variables, constraints1, True, [])
        cache.put_result(variables, constraints2, False, ["x_negative"])

        result1 = cache.get_result(variables, constraints1)
        result2 = cache.get_result(variables, constraints2)

        assert result1 == (True, [])
        assert result2 == (False, ["x_negative"])

    def test_constraint_cache_order_independence(self):
        """Test that constraint order doesn't affect cache key."""
        cache = ConstraintSolverCache(max_size=10)

        variables = {"x": "Int", "y": "Int"}
        constraints1 = ["x > 0", "y < 10"]
        constraints2 = ["y < 10", "x > 0"]  # Different order

        cache.put_result(variables, constraints1, True, [])

        # Should hit cache despite different order
        result = cache.get_result(variables, constraints2)
        assert result == (True, [])

    def test_constraint_cache_statistics(self):
        """Test constraint cache statistics."""
        cache = ConstraintSolverCache(max_size=10)

        variables = {"x": "Int"}
        constraints = ["x > 0"]

        # Miss
        cache.get_result(variables, constraints)

        # Put and hit
        cache.put_result(variables, constraints, True, [])
        cache.get_result(variables, constraints)

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestVerificationResultCache:
    """Test VerificationResultCache functionality."""

    def test_verification_cache_basic(self):
        """Test basic verification result caching."""
        cache = VerificationResultCache(max_size=10)

        code = "SELECT * FROM users;"
        rule_id = "sql_check"
        rule_spec = {"check_syntax": True}

        # Cache miss
        result = cache.get_result(code, rule_id, rule_spec)
        assert result is None

        # Cache result
        mock_result = {"passed": True}
        cache.put_result(code, rule_id, rule_spec, mock_result)

        # Cache hit
        result = cache.get_result(code, rule_id, rule_spec)
        assert result == mock_result

    def test_verification_cache_different_code(self):
        """Test that different code produces different cache keys."""
        cache = VerificationResultCache(max_size=10)

        code1 = "SELECT * FROM users;"
        code2 = "SELECT * FROM products;"
        rule_id = "sql_check"
        rule_spec = {"check_syntax": True}

        result1 = {"passed": True}
        result2 = {"passed": False}

        cache.put_result(code1, rule_id, rule_spec, result1)
        cache.put_result(code2, rule_id, rule_spec, result2)

        assert cache.get_result(code1, rule_id, rule_spec) == result1
        assert cache.get_result(code2, rule_id, rule_spec) == result2

    def test_verification_cache_different_rules(self):
        """Test that different rules produce different cache keys."""
        cache = VerificationResultCache(max_size=10)

        code = "SELECT * FROM users;"
        rule_id = "sql_check"
        rule_spec1 = {"check_syntax": True}
        rule_spec2 = {"check_syntax": True, "check_injection": True}

        result1 = {"passed": True}
        result2 = {"passed": False}

        cache.put_result(code, rule_id, rule_spec1, result1)
        cache.put_result(code, rule_id, rule_spec2, result2)

        assert cache.get_result(code, rule_id, rule_spec1) == result1
        assert cache.get_result(code, rule_id, rule_spec2) == result2


class TestGlobalCaches:
    """Test global cache instances."""

    def test_get_constraint_cache_singleton(self):
        """Test that get_constraint_cache returns singleton."""
        cache1 = get_constraint_cache()
        cache2 = get_constraint_cache()

        assert cache1 is cache2

    def test_get_verification_cache_singleton(self):
        """Test that get_verification_cache returns singleton."""
        cache1 = get_verification_cache()
        cache2 = get_verification_cache()

        assert cache1 is cache2

    def test_clear_all_caches(self):
        """Test clearing all global caches."""
        constraint_cache = get_constraint_cache()
        verification_cache = get_verification_cache()

        # Add some data
        constraint_cache.put_result({"x": "Int"}, ["x > 0"], True, [])
        verification_cache.put_result("code", "rule", {}, {"passed": True})

        # Clear all
        clear_all_caches()

        # Verify cleared
        assert constraint_cache.get_result({"x": "Int"}, ["x > 0"]) is None
        assert verification_cache.get_result("code", "rule", {}) is None


class TestCacheThreadSafety:
    """Test cache thread safety."""

    def test_concurrent_access(self):
        """Test concurrent cache access."""
        import threading

        cache = LRUCache(max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    cache.put(key, f"value_{thread_id}_{i}")
                    result = cache.get(key)
                    assert result == f"value_{thread_id}_{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
