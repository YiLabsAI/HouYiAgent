"""Advanced caching system for verification results.

This module provides a sophisticated multi-level caching system for:
1. Constraint solving results (Z3 solver results)
2. Verification results (complete verification outcomes)
3. LRU eviction with TTL support
4. Cache statistics and monitoring
"""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: float | None = None  # Time-to-live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics for monitoring and optimization."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


class LRUCache:
    """Thread-safe LRU cache with TTL support.

    Features:
    - LRU eviction policy
    - Optional TTL for entries
    - Thread-safe operations
    - Detailed statistics
    - Automatic cleanup of expired entries
    """

    def __init__(self, max_size: int = 1000, default_ttl: float | None = None):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self.stats = CacheStats()

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments.

        Uses SHA256 hash of string representation for consistency.
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            # Check expiration
            if entry.is_expired():
                self._cache.pop(key)
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            # Update access metadata and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)
            self.stats.hits += 1

            return entry.value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Put value into cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses default_ttl if None)
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._cache.pop(key)

            # Evict oldest entry if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
                self.stats.evictions += 1

            # Create and store new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl if ttl is not None else self.default_ttl,
            )
            self._cache[key] = entry
            self.stats.total_size = len(self._cache)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                self.stats.total_size = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats.total_size = 0

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

            for key in expired_keys:
                self._cache.pop(key)
                self.stats.expirations += 1

            self.stats.total_size = len(self._cache)
            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate,
                "evictions": self.stats.evictions,
                "expirations": self.stats.expirations,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0.0,
            }


class ConstraintSolverCache:
    """Specialized cache for constraint solver results.

    Caches Z3 solver results based on:
    - Variable declarations
    - Constraint expressions
    - Solver configuration

    This significantly improves performance for repeated constraint checks.
    """

    def __init__(self, max_size: int = 500, ttl: float = 3600.0):
        """Initialize constraint solver cache.

        Args:
            max_size: Maximum number of cached results
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl)

    def _make_constraint_key(
        self,
        variables: dict[str, str],
        constraints: list[str],
    ) -> str:
        """Generate cache key for constraint problem.

        Args:
            variables: Variable declarations {name: type}
            constraints: List of constraint expressions

        Returns:
            Cache key (SHA256 hash)
        """
        # Sort for consistency
        var_str = "|".join(f"{k}:{v}" for k, v in sorted(variables.items()))
        const_str = "|".join(sorted(constraints))
        key_str = f"vars:{var_str}|constraints:{const_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_result(
        self,
        variables: dict[str, str],
        constraints: list[str],
    ) -> tuple[bool, list[str]] | None:
        """Get cached constraint solving result.

        Args:
            variables: Variable declarations
            constraints: Constraint expressions

        Returns:
            Tuple of (is_satisfiable, violated_constraints) or None
        """
        key = self._make_constraint_key(variables, constraints)
        result = self._cache.get(key)

        if result is not None:
            logger.debug(f"Constraint cache hit for key: {key[:16]}...")

        return result

    def put_result(
        self,
        variables: dict[str, str],
        constraints: list[str],
        is_satisfiable: bool,
        violated_constraints: list[str],
    ) -> None:
        """Cache constraint solving result.

        Args:
            variables: Variable declarations
            constraints: Constraint expressions
            is_satisfiable: Whether constraints are satisfiable
            violated_constraints: List of violated constraint names
        """
        key = self._make_constraint_key(variables, constraints)
        self._cache.put(key, (is_satisfiable, violated_constraints))
        logger.debug(f"Cached constraint result for key: {key[:16]}...")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


class VerificationResultCache:
    """Specialized cache for complete verification results.

    Caches verification outcomes based on:
    - Input code/query
    - Verification rules
    - Verifier configuration

    This avoids redundant verification of identical inputs.
    """

    def __init__(self, max_size: int = 1000, ttl: float = 1800.0):
        """Initialize verification result cache.

        Args:
            max_size: Maximum number of cached results
            ttl: Time-to-live in seconds (default: 30 minutes)
        """
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl)

    def _make_verification_key(
        self,
        code: str,
        rule_id: str,
        rule_spec: dict[str, Any],
    ) -> str:
        """Generate cache key for verification request.

        Args:
            code: Code/query to verify
            rule_id: Verification rule ID
            rule_spec: Rule specification

        Returns:
            Cache key (SHA256 hash)
        """
        # Create deterministic representation
        spec_str = "|".join(f"{k}={v}" for k, v in sorted(rule_spec.items()))
        key_str = f"code:{code}|rule:{rule_id}|spec:{spec_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_result(
        self,
        code: str,
        rule_id: str,
        rule_spec: dict[str, Any],
    ) -> Any | None:
        """Get cached verification result.

        Args:
            code: Code/query to verify
            rule_id: Verification rule ID
            rule_spec: Rule specification

        Returns:
            Cached VerificationResult or None
        """
        key = self._make_verification_key(code, rule_id, rule_spec)
        result = self._cache.get(key)

        if result is not None:
            logger.debug(f"Verification cache hit for key: {key[:16]}...")

        return result

    def put_result(
        self,
        code: str,
        rule_id: str,
        rule_spec: dict[str, Any],
        result: Any,
    ) -> None:
        """Cache verification result.

        Args:
            code: Code/query that was verified
            rule_id: Verification rule ID
            rule_spec: Rule specification
            result: VerificationResult to cache
        """
        key = self._make_verification_key(code, rule_id, rule_spec)
        self._cache.put(key, result)
        logger.debug(f"Cached verification result for key: {key[:16]}...")

    def invalidate_for_rule(self, rule_id: str) -> int:
        """Invalidate all cached results for a specific rule.

        Args:
            rule_id: Rule ID to invalidate

        Returns:
            Number of entries invalidated
        """
        # Note: This is a simplified implementation
        # In production, you'd want to maintain a reverse index
        self._cache.clear()
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


# Global cache instances (singleton pattern)
_constraint_cache: ConstraintSolverCache | None = None
_verification_cache: VerificationResultCache | None = None


def get_constraint_cache() -> ConstraintSolverCache:
    """Get global constraint solver cache instance."""
    global _constraint_cache
    if _constraint_cache is None:
        _constraint_cache = ConstraintSolverCache()
    return _constraint_cache


def get_verification_cache() -> VerificationResultCache:
    """Get global verification result cache instance."""
    global _verification_cache
    if _verification_cache is None:
        _verification_cache = VerificationResultCache()
    return _verification_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    global _constraint_cache, _verification_cache
    if _constraint_cache:
        _constraint_cache.clear()
    if _verification_cache:
        _verification_cache.clear()
