"""Stress tests for concurrent read/write, deadlock detection, and
connection lifecycle control (G10.6).

These tests verify industrial-grade reliability under concurrent load:
- No SQLite deadlock under multi-thread CRUD (the real bench scenario)
- No ResourceWarning on graceful shutdown
- Connection lifecycle cleanup after stop()
- asyncio + threading mix without deadlock
- Data consistency after concurrent writes

Marked pytest.mark.stress and lives under tests/integration/stress/ so
make check never picks them up.  Run via:  make test-stress
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_connection import SQLiteConnectionManager
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.types import (
    MemoryPolicy,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RawTurn,
)

pytestmark = pytest.mark.stress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(key: str, scope: MemoryScope = MemoryScope.SESSION) -> MemoryRecord:
    """Create a minimal MemoryRecord for stress testing."""
    return MemoryRecord(
        key=key,
        scope=scope,
        type=MemoryType.FACT,
        content=f"stress-content-{key}",
        metadata={"source": "stress"},
    )


def _file_db(tmp_path: Path, name: str = "stress.db") -> Path:
    """Return a file-backed DB path (avoids :memory: cross-thread issues)."""
    return tmp_path / name


def _raw_turn(text: str, idx: int = 0) -> RawTurn:
    """Create a minimal RawTurn for testing."""
    return RawTurn(
        session_id="stress-session",
        turn_index=idx,
        role="user",
        content=text,
    )


def _build_stub_engine(tmp_path: Path, db_name: str = "engine-stress.db") -> MemoryEngine:
    """Build a MemoryEngine with TurnWriter for stress testing."""
    backend = SQLiteMemoryBackend(db_path=_file_db(tmp_path, db_name))
    store = MemoryStore(backend=backend)
    turn_writer = TurnWriter(backend=backend)
    return MemoryEngine(
        store,
        policy=MemoryPolicy(auto_approve=True),
        turn_writer=turn_writer,
    )


# ---------------------------------------------------------------------------
# 1. Concurrent CRUD on SQLiteMemoryBackend — bench-level load
# ---------------------------------------------------------------------------


class TestBackendConcurrentCRUD:
    """Multi-threaded put/get/search at bench-level concurrency.

    Bench runs with --concurrency 5 hit deadlocks under multi-write.
    These tests reproduce that load pattern and verify no deadlock,
    no data loss, and no corruption.
    """

    def test_concurrent_put_get(self, tmp_path: Path) -> None:
        """8 threads x 500 ops: all writes visible, content intact."""
        backend = SQLiteMemoryBackend(db_path=_file_db(tmp_path))
        scope = MemoryScope.SESSION
        n_threads = 8
        n_ops = 500

        def worker(tid: int) -> list[str]:
            found = []
            for i in range(n_ops):
                key = f"t{tid}-k{i}"
                content = f"stress-content-t{tid}-k{i}"
                backend.put(
                    MemoryRecord(
                        key=key,
                        scope=scope,
                        type=MemoryType.FACT,
                        content=content,
                        metadata={"src": f"w{tid}"},
                    )
                )
                rec = backend.get(key, scope)
                if rec is not None:
                    # Verify content matches — catches corruption
                    assert rec.content == content, (
                        f"Data corruption: key={key} expected={content} got={rec.content}"
                    )
                    found.append(rec.key)
            return found

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(worker, t) for t in range(n_threads)]
            all_keys = []
            for fut in as_completed(futs):
                all_keys.extend(fut.result())

        assert len(all_keys) == n_threads * n_ops, (
            f"Data loss: expected {n_threads * n_ops} reads, got {len(all_keys)}"
        )
        backend.close()

    def test_crud_fts_mixed(self, tmp_path: Path) -> None:
        """6 threads: writers + deleters + searchers under contention.
        Mirrors the bench where extraction + ingestion + recall hit
        the same database simultaneously."""
        backend = SQLiteMemoryBackend(db_path=_file_db(tmp_path))
        scope = MemoryScope.SESSION

        for i in range(100):
            backend.put(_make_record(f"seed-{i}", scope))

        def writer(tid: int) -> int:
            for i in range(200):
                backend.put(_make_record(f"w{tid}-{i}", scope))
            return 200

        def deleter() -> int:
            count = 0
            for i in range(100):
                if backend.delete(f"seed-{i}", scope):
                    count += 1
            return count

        def searcher() -> int:
            total_hits = 0
            for _ in range(20):
                results = backend.search_fts("stress-content", scope=scope, limit=10)
                total_hits += len(results)
            return total_hits

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [
                pool.submit(writer, 0),
                pool.submit(writer, 1),
                pool.submit(writer, 2),
                pool.submit(deleter),
                pool.submit(searcher),
                pool.submit(searcher),
            ]
            for fut in as_completed(futs):
                fut.result()

        remaining = backend.list_by_scope(scope)
        for rec in remaining:
            assert rec.content.startswith("stress-content"), (
                f"Corrupted record: {rec.key} content={rec.content}"
            )
        backend.close()

    def test_concurrent_tx(self, tmp_path: Path) -> None:
        """4 threads x 100 BEGIN IMMEDIATE transactions — WAL prevents
        deadlock.  This pattern caused bench deadlocks before."""
        backend = SQLiteMemoryBackend(db_path=_file_db(tmp_path))
        scope = MemoryScope.SESSION

        for i in range(500):
            backend.put(_make_record(f"tx-seed-{i}", scope))

        def tx_worker(tid: int) -> int:
            count = 0
            for i in range(100):
                with backend.transaction():
                    key = f"tx{tid}-{i}"
                    backend.put(_make_record(key, scope))
                    rec = backend.get(key, scope)
                    assert rec is not None, f"Write not visible in own tx: {key}"
                    count += 1
            return count

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(tx_worker, t) for t in range(4)]
            total = sum(f.result() for f in as_completed(futs))

        assert total == 400
        backend.close()


# ---------------------------------------------------------------------------
# 2. Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    """Verify connection creation, registration, and cleanup."""

    def test_close_all_clears_set(self, tmp_path: Path) -> None:
        """close_all() empties the _connections set from 10 threads."""
        mgr = SQLiteConnectionManager(db_path=_file_db(tmp_path))
        conns = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = [pool.submit(lambda: mgr.get_connection()) for _ in range(10)]
            for f in as_completed(futs):
                conns.append(f.result())
        assert len(mgr._connections) >= 8
        mgr.close_all()
        assert len(mgr._connections) == 0

    def test_close_all_thread_local(self, tmp_path: Path) -> None:
        """close_all() nullifies caller's _local.conn only; other
        threads retain stale refs — documented limitation."""
        mgr = SQLiteConnectionManager(db_path=_file_db(tmp_path))
        alive = threading.Event()

        def other_thread() -> None:
            mgr.get_connection()
            alive.wait(0.5)

        t = threading.Thread(target=other_thread)
        t.start()
        mgr.get_connection()
        mgr.close_all()
        assert mgr._local.conn is None
        assert len(mgr._connections) == 0
        alive.set()
        t.join(timeout=2)

    def test_fresh_conn_after_close(self, tmp_path: Path) -> None:
        """After close_all(), get_connection() returns a fresh
        functional connection."""
        mgr = SQLiteConnectionManager(db_path=_file_db(tmp_path))
        mgr.get_connection()
        mgr.close_all()
        fresh = mgr.get_connection()
        fresh.execute("SELECT 1").fetchone()
        assert len(mgr._connections) == 1
        mgr.close_all()


# ---------------------------------------------------------------------------
# 3. MemoryEngine concurrent write + recall
# ---------------------------------------------------------------------------


class TestEngineConcurrentAccess:
    """asyncio + threading mix — the real bench scenario."""

    @pytest.mark.asyncio
    async def test_concurrent_write(self, tmp_path: Path) -> None:
        """20 concurrent write_turn calls — all succeed, no deadlock."""
        engine = _build_stub_engine(tmp_path)
        await engine.start()
        n = 20
        results = await asyncio.gather(
            *[
                engine.write_turn(_raw_turn(f"turn-{i}", idx=i), schedule_extract=False)
                for i in range(n)
            ],
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Write errors: {errors}"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_sustained_write_recall(self, tmp_path: Path) -> None:
        """3-second sustained load: concurrent writes + recalls.
        Mirrors the bench where extraction+recall overlap."""
        engine = _build_stub_engine(tmp_path)
        await engine.start()

        for i in range(20):
            await engine.write_turn(_raw_turn(f"seed-{i}", idx=i), schedule_extract=False)

        write_count = 0
        recall_count = 0
        stop_flag = asyncio.Event()

        async def writer_loop() -> None:
            nonlocal write_count
            i = 0
            while not stop_flag.is_set():
                try:
                    await engine.write_turn(
                        _raw_turn(f"concurrent-{i}", idx=i), schedule_extract=False
                    )
                    write_count += 1
                    i += 1
                except asyncio.CancelledError:
                    break

        async def recall_loop() -> None:
            nonlocal recall_count
            while not stop_flag.is_set():
                try:
                    await engine.recall("stress-content", top_k=5)
                    recall_count += 1
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    break

        w_task = asyncio.create_task(writer_loop())
        r_task = asyncio.create_task(recall_loop())
        await asyncio.sleep(3.0)
        stop_flag.set()
        await asyncio.sleep(0.2)
        w_task.cancel()
        r_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(w_task, r_task, return_exceptions=True)

        assert write_count > 0, "No writes completed in 3 seconds"
        assert recall_count > 0, "No recalls completed in 3 seconds"
        await engine.stop()


# ---------------------------------------------------------------------------
# 4. Graceful shutdown — 0 ResourceWarning (hard assertion)
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Verify stop() and close() produce no ResourceWarning.

    Uses pytest.mark.filterwarnings('error::ResourceWarning') so any
    leak causes immediate hard failure, not a soft post-check.
    """

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("error::ResourceWarning")
    async def test_stop_no_warning(self, tmp_path: Path) -> None:
        """engine.stop() after idle — no leak."""
        engine = _build_stub_engine(tmp_path)
        await engine.start()
        await engine.write_turn(_raw_turn("before-shutdown"), schedule_extract=False)
        await engine.stop()
        gc.collect()

    @pytest.mark.filterwarnings("error::ResourceWarning")
    def test_close_no_warning(self, tmp_path: Path) -> None:
        """backend.close() from multi-thread context — no leak."""
        backend = SQLiteMemoryBackend(db_path=_file_db(tmp_path))
        scope = MemoryScope.SESSION
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = [
                pool.submit(lambda: backend.put(_make_record("warmup", scope))) for _ in range(5)
            ]
            for f in as_completed(futs):
                f.result()
        backend.close()
        gc.collect()

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("error::ResourceWarning")
    async def test_async_with_no_warning(self, tmp_path: Path) -> None:
        """async with engine: lifecycle — no leak."""
        async with _build_stub_engine(tmp_path) as engine:
            await engine.write_turn(_raw_turn("lifecycle-test"), schedule_extract=False)
        gc.collect()

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("error::ResourceWarning")
    async def test_stop_mid_write(self, tmp_path: Path) -> None:
        """stop() called while writes are in-flight — no leak.
        In-flight writes may get ProgrammingError on closed
        connections; this is expected.  The test verifies that
        stop() itself does not leak resources."""
        engine = _build_stub_engine(tmp_path)
        await engine.start()

        # Start sustained writes for 2 seconds
        for i in range(10):
            await engine.write_turn(_raw_turn(f"pre-load-{i}", idx=i), schedule_extract=False)

        # Schedule a background write loop then stop after 1 second
        async def write_burst() -> None:
            for j in range(50):
                try:
                    await engine.write_turn(_raw_turn(f"burst-{j}", idx=j), schedule_extract=False)
                except Exception:
                    break  # expected: backend may close mid-write

        burst_task = asyncio.create_task(write_burst())
        await asyncio.sleep(0.5)  # let some writes land

        await engine.stop()  # close backend mid-stream

        with contextlib.suppress(Exception):
            await burst_task

        gc.collect()
