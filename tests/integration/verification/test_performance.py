"""Performance benchmarks for verification.

Measures overhead and performance impact of verification.
"""

import asyncio
import time

import pytest

from houyi.assurance.verification import (
    PythonVerifier,
    SQLVerifier,
    VerificationRule,
)

pytestmark = pytest.mark.benchmark


class PerformanceMetrics:
    """Collect performance metrics."""

    def __init__(self):
        self.timings = []

    def record(self, duration: float):
        self.timings.append(duration)

    @property
    def avg_ms(self) -> float:
        return (sum(self.timings) / len(self.timings)) * 1000 if self.timings else 0

    @property
    def min_ms(self) -> float:
        return min(self.timings) * 1000 if self.timings else 0

    @property
    def max_ms(self) -> float:
        return max(self.timings) * 1000 if self.timings else 0

    @property
    def p95_ms(self) -> float:
        if not self.timings:
            return 0
        sorted_timings = sorted(self.timings)
        idx = int(len(sorted_timings) * 0.95)
        return sorted_timings[idx] * 1000


@pytest.mark.asyncio
async def test_sql_verifier_performance():
    """Benchmark SQL verifier performance."""
    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="perf_test",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": True},
    )

    metrics = PerformanceMetrics()
    queries = [
        "SELECT * FROM users WHERE id = 1;",
        "SELECT name, email FROM customers;",
        "SELECT COUNT(*) FROM orders WHERE status = 'pending';",
    ] * 100  # 300 queries

    for query in queries:
        start = time.perf_counter()
        await verifier.verify(query, rule)
        duration = time.perf_counter() - start
        metrics.record(duration)

    print("\n=== SQL Verifier Performance ===")
    print(f"Queries: {len(queries)}")
    print(f"Avg: {metrics.avg_ms:.2f}ms")
    print(f"Min: {metrics.min_ms:.2f}ms")
    print(f"Max: {metrics.max_ms:.2f}ms")
    print(f"P95: {metrics.p95_ms:.2f}ms")

    # Performance target: < 10ms average
    assert metrics.avg_ms < 10, f"SQL verification too slow: {metrics.avg_ms:.2f}ms"


@pytest.mark.asyncio
async def test_python_verifier_performance():
    """Benchmark Python verifier performance."""
    verifier = PythonVerifier()
    rule = VerificationRule(
        rule_id="perf_test",
        verifier_type="python",
        rule_spec={"check_syntax": True, "check_imports": True},
    )

    metrics = PerformanceMetrics()
    code_samples = [
        "x = 1 + 2\nprint(x)",
        "def hello():\n    return 'world'",
        "import json\ndata = json.loads('{}')",
    ] * 100  # 300 samples

    for code in code_samples:
        start = time.perf_counter()
        await verifier.verify(code, rule)
        duration = time.perf_counter() - start
        metrics.record(duration)

    print("\n=== Python Verifier Performance ===")
    print(f"Samples: {len(code_samples)}")
    print(f"Avg: {metrics.avg_ms:.2f}ms")
    print(f"Min: {metrics.min_ms:.2f}ms")
    print(f"Max: {metrics.max_ms:.2f}ms")
    print(f"P95: {metrics.p95_ms:.2f}ms")

    # Performance target: < 5ms average (AST parsing is fast)
    assert metrics.avg_ms < 5, f"Python verification too slow: {metrics.avg_ms:.2f}ms"


@pytest.mark.asyncio
async def test_auto_fix_performance():
    """Benchmark auto-fix performance."""
    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="perf_test",
        verifier_type="sql",
        rule_spec={"check_syntax": True},
    )

    metrics_verify = PerformanceMetrics()
    metrics_fix = PerformanceMetrics()

    queries = ["SELECT * FROM users WHERE id = 1"] * 100  # Missing semicolon

    for query in queries:
        # Measure verification
        start = time.perf_counter()
        result = await verifier.verify(query, rule)
        metrics_verify.record(time.perf_counter() - start)

        # Measure auto-fix
        if not result.passed and result.auto_fixable:
            start = time.perf_counter()
            await verifier.auto_fix(query, result)
            metrics_fix.record(time.perf_counter() - start)

    print("\n=== Auto-fix Performance ===")
    print(f"Verify avg: {metrics_verify.avg_ms:.2f}ms")
    print(f"Fix avg: {metrics_fix.avg_ms:.2f}ms")
    print(f"Total avg: {(metrics_verify.avg_ms + metrics_fix.avg_ms):.2f}ms")

    # Auto-fix should be fast
    assert metrics_fix.avg_ms < 5, f"Auto-fix too slow: {metrics_fix.avg_ms:.2f}ms"


@pytest.mark.asyncio
async def test_verification_overhead():
    """Measure verification overhead vs no verification."""
    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="overhead_test",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": True},
    )

    query = "SELECT * FROM users WHERE id = 1;"
    iterations = 1000

    # Baseline: no verification
    start = time.perf_counter()
    for _ in range(iterations):
        # Simulate just executing the query
        _ = query
    baseline_time = time.perf_counter() - start

    # With verification
    start = time.perf_counter()
    for _ in range(iterations):
        await verifier.verify(query, rule)
    verify_time = time.perf_counter() - start

    overhead = verify_time - baseline_time
    overhead_per_query = (overhead / iterations) * 1000

    print("\n=== Verification Overhead ===")
    print(f"Iterations: {iterations}")
    print(f"Baseline: {baseline_time * 1000:.2f}ms")
    print(f"With verification: {verify_time * 1000:.2f}ms")
    print(f"Overhead: {overhead * 1000:.2f}ms total")
    print(f"Per query: {overhead_per_query:.3f}ms")

    # Overhead should be minimal (< 5ms per query)
    assert overhead_per_query < 5, f"Verification overhead too high: {overhead_per_query:.3f}ms"


@pytest.mark.asyncio
async def test_concurrent_verification():
    """Test verification performance under concurrent load."""
    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="concurrent_test",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": True},
    )

    queries = [
        "SELECT * FROM users WHERE id = 1;",
        "SELECT name FROM customers;",
        "SELECT COUNT(*) FROM orders;",
    ] * 50  # 150 queries

    start = time.perf_counter()

    # Run concurrently
    tasks = [verifier.verify(query, rule) for query in queries]
    results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start

    print("\n=== Concurrent Verification ===")
    print(f"Queries: {len(queries)}")
    print(f"Total time: {duration * 1000:.2f}ms")
    print(f"Avg per query: {(duration / len(queries)) * 1000:.2f}ms")
    print(f"Throughput: {len(queries) / duration:.0f} queries/sec")

    # All should pass
    assert all(r.passed for r in results), "All queries should pass"

    # Should handle concurrent load efficiently
    assert duration < 1.0, f"Concurrent verification too slow: {duration:.2f}s"


if __name__ == "__main__":
    asyncio.run(test_sql_verifier_performance())
    asyncio.run(test_python_verifier_performance())
    asyncio.run(test_auto_fix_performance())
    asyncio.run(test_verification_overhead())
    asyncio.run(test_concurrent_verification())
