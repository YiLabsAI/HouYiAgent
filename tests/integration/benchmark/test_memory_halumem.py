"""Memory benchmark smoke tests.

Exercises the full HaluMem-aligned harness on the built-in synthetic
fixture, using a stub LLM for ingestion and the stub judge / answerer.
The goal is plumbing validation only — no real dataset, no network, no
LLM judge — so it can run on any machine in a few seconds.

Marked pytest.mark.benchmark and lives under tests/integration/
so the make check unit gate (which excludes tests/integration/)
never picks it up. Run via:

 make benchmark BENCH_TARGET=memory-halumem
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from houyi.adapters.memory.backends.sqlite import SQLiteMemoryBackend
from houyi.adapters.memory.backends.sqlite_candidate_inbox import SQLiteCandidateInbox
from houyi.adapters.memory.backends.sqlite_entity_state import SQLiteEntityStateView
from houyi.adapters.memory.extractor import AtomicFactExtractor
from houyi.adapters.memory.ingestor import MemoryIngestor
from houyi.adapters.memory.resolver import MemoryWriterTools
from houyi.adapters.memory.retraction import (
    RetractionDetector,
    RetractionOrchestrator,
)
from houyi.adapters.memory.triggers import all_of
from houyi.adapters.memory.turn_writer import TurnWriter
from houyi.adapters.memory.workers.extractor_worker import (
    ExtractorWorker,
    ExtractorWorkerConfig,
)
from houyi.arena.memory_bench import (
    PATH_KIND_SYNC_INLINE,
    PATH_KIND_TIERED_ASYNC,
    MemoryBenchRunner,
    StubMemoryJudge,
    SubstringAnswerer,
    TieredBenchIngestor,
    load_synthetic_fixture,
)
from houyi.arena.memory_bench.runner import MemoryReader

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Stub LLM (mimics AtomicFactExtractor's LLM contract)
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    content: str


class _RoutedLLM:
    """Stub chat client whose response depends on the user message content.

    Order-keyed scripting collapses when tests subset sessions, so the
    routing rule keys off substrings of the user turn. Anything that
    does not match any rule returns "[]" — the ingestor's defensive
    parse path absorbs that as "no facts extracted".
    """

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        # Stored as a list (not dict) so multiple substring rules can
        # share keys; first match wins per call.
        self._rules = list(rules)

    async def chat(self, messages, *, temperature: float, max_tokens: int):
        last_user = ""
        for msg in messages:
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
        for needle, payload in self._rules:
            if needle.lower() in last_user.lower():
                return _StubResponse(payload)
        return _StubResponse("[]")


# ---------------------------------------------------------------------------
# Fixture: build a real ingestor against tmp SQLite + stub LLM
# ---------------------------------------------------------------------------


def _facts(*items: dict) -> str:
    return json.dumps(list(items))


@pytest.fixture
def bench_setup(tmp_path: Path) -> Iterator[tuple[MemoryBenchRunner, _RoutedLLM]]:
    backend = SQLiteMemoryBackend(db_path=tmp_path / "bench.db")
    try:
        view = SQLiteEntityStateView(backend)
        inbox = SQLiteCandidateInbox(backend)
        tools = MemoryWriterTools(view, inbox, namespace="bench")

        # Content-keyed scripting: each rule maps a unique substring of
        # the user turn to the canned JSON the extractor LLM should
        # return. Tests can subset sessions without breaking ordering.
        llm = _RoutedLLM(
            [
                # Extract session (single turn introducing Alice).
                (
                    "my name is Alice",
                    _facts(
                        {
                            "subject": "user",
                            "predicate": "name",
                            "object": "Alice",
                            "certainty": "certain",
                        },
                        {
                            "subject": "user",
                            "predicate": "lives_in",
                            "object": "Beijing",
                            "certainty": "certain",
                        },
                        {
                            "subject": "user",
                            "predicate": "works_at",
                            "object": "Aurora",
                            "certainty": "certain",
                        },
                    ),
                ),
                # Update session, turn 1: initial Beijing claim.
                (
                    "I live in Beijing",
                    _facts(
                        {
                            "subject": "user",
                            "predicate": "lives_in",
                            "object": "Beijing",
                            "certainty": "certain",
                        },
                    ),
                ),
                # Update session, turn 2: Shanghai supersession.
                (
                    "moved to Shanghai",
                    _facts(
                        {
                            "subject": "user",
                            "predicate": "lives_in",
                            "object": "Shanghai",
                            "certainty": "certain",
                        },
                    ),
                ),
                # Vague session: project status hedged.
                (
                    "kind of stuck",
                    _facts(
                        {
                            "subject": "project",
                            "predicate": "status",
                            "object": "stuck",
                            "certainty": "vague",
                        },
                    ),
                ),
            ]
        )
        extractor = AtomicFactExtractor(llm)
        orchestrator = RetractionOrchestrator(RetractionDetector(), tools)
        ingestor = MemoryIngestor(extractor, orchestrator, tools, inbox)

        class _Reader(MemoryReader):
            def list_active_memories(self, ns: str) -> list[str]:
                rows = view.get_active(ns, entity="user")
                return [f"{r.attribute}: {r.value}" for r in rows]

        runner = MemoryBenchRunner(
            ingestor,
            _Reader(),
            judge=StubMemoryJudge(),
            answerer=SubstringAnswerer(),
            namespace="bench",
        )
        yield runner, llm
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestRunnerSmoke:
    """End-to-end plumbing exercises on the built-in fixture."""

    @pytest.mark.asyncio
    async def test_runner_produces_report(self, bench_setup) -> None:
        runner, _ = bench_setup
        sessions = load_synthetic_fixture()
        report = await runner.run(sessions)
        assert len(report.sessions) == len(sessions)
        # The fixture is too small for absolute-score gates, but the
        # extraction recall must be > 0 — the SUT did manage to absorb
        # at least one user fact.
        assert report.aggregate.extraction.memory_recall > 0.0

    @pytest.mark.asyncio
    async def test_update_supersession(self, bench_setup) -> None:
        runner, _ = bench_setup
        # Pluck only the update fixture so the assertion is targeted.
        sessions = [s for s in load_synthetic_fixture() if s.session_id == "fixture-update-001"]
        report = await runner.run(sessions)
        assert report.aggregate.update.target_total == 1
        # With the stub judge + scripted LLM, the supersession should land.
        assert report.aggregate.update.upd_acc == 1.0

    @pytest.mark.asyncio
    async def test_vague_skips_main_store(self, bench_setup) -> None:
        runner, _ = bench_setup
        sessions = [s for s in load_synthetic_fixture() if s.session_id == "fixture-vague-001"]
        report = await runner.run(sessions)
        # Gold has no memories for the vague session; the SUT must not
        # have parked the vague fact into the active store.
        assert report.aggregate.extraction.predicted_total == 0


# ---------------------------------------------------------------------------
# Sync vs tiered (L0+L1 async) comparison harness
# ---------------------------------------------------------------------------


def _routing_rules() -> list[tuple[str, str]]:
    """Shared scripted-LLM rules used by both sync and tiered setups.

    Kept inline rather than extracted from the existing fixture so this
    file remains the single source of truth for what the bench's stub LLM
    answers; the comparison test only needs the "extract" session's three
    facts plus the update + vague sessions to exercise the full surface.
    """
    return [
        (
            "my name is Alice",
            _facts(
                {"subject": "user", "predicate": "name", "object": "Alice", "certainty": "certain"},
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Beijing",
                    "certainty": "certain",
                },
                {
                    "subject": "user",
                    "predicate": "works_at",
                    "object": "Aurora",
                    "certainty": "certain",
                },
            ),
        ),
        (
            "I live in Beijing",
            _facts(
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Beijing",
                    "certainty": "certain",
                },
            ),
        ),
        (
            "moved to Shanghai",
            _facts(
                {
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Shanghai",
                    "certainty": "certain",
                },
            ),
        ),
        (
            "kind of stuck",
            _facts(
                {
                    "subject": "project",
                    "predicate": "status",
                    "object": "stuck",
                    "certainty": "vague",
                },
            ),
        ),
    ]


class _DelayingLLM(_RoutedLLM):
    """Routed stub LLM that sleeps before responding.

    A non-zero delay is what makes the sync-vs-tiered timing comparison
    informative: on the sync path the sleep happens inside ingest_turn,
    on the tiered path it happens inside the worker drain. Without a
    delay both paths trivially complete in microseconds and the relative
    distribution becomes noise-dominated.
    """

    def __init__(self, rules: list[tuple[str, str]], *, delay_s: float) -> None:
        super().__init__(rules)
        self._delay_s = delay_s

    async def chat(self, messages, *, temperature: float, max_tokens: int):
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        return await super().chat(messages, temperature=temperature, max_tokens=max_tokens)


def _build_sync_runner(tmp_path: Path, *, delay_s: float) -> tuple[MemoryBenchRunner, callable]:
    """Build a runner driving the legacy MemoryIngestor (LLM in ingest_turn)."""
    backend = SQLiteMemoryBackend(db_path=tmp_path / "sync.db")
    view = SQLiteEntityStateView(backend)
    inbox = SQLiteCandidateInbox(backend)
    tools = MemoryWriterTools(view, inbox, namespace="bench")
    llm = _DelayingLLM(_routing_rules(), delay_s=delay_s)
    extractor = AtomicFactExtractor(llm)
    orchestrator = RetractionOrchestrator(RetractionDetector(), tools)
    ingestor = MemoryIngestor(extractor, orchestrator, tools, inbox)

    class _Reader(MemoryReader):
        def list_active_memories(self, ns: str) -> list[str]:
            rows = view.get_active(ns, entity="user")
            return [f"{r.attribute}: {r.value}" for r in rows]

    runner = MemoryBenchRunner(
        ingestor,
        _Reader(),
        judge=StubMemoryJudge(),
        answerer=SubstringAnswerer(),
        namespace="bench",
        path_kind=PATH_KIND_SYNC_INLINE,
    )
    return runner, backend.close


def _build_tiered_runner(tmp_path: Path, *, delay_s: float) -> tuple[MemoryBenchRunner, callable]:
    """Build a runner driving TurnWriter + ExtractorWorker (LLM in worker drain)."""
    backend = SQLiteMemoryBackend(db_path=tmp_path / "tiered.db")
    view = SQLiteEntityStateView(backend)
    inbox = SQLiteCandidateInbox(backend)
    tools = MemoryWriterTools(view, inbox, namespace="bench")
    llm = _DelayingLLM(_routing_rules(), delay_s=delay_s)
    extractor = AtomicFactExtractor(llm)
    retraction = RetractionOrchestrator(RetractionDetector(), tools)
    # all_of() is the empty composite -> always-extract policy. The bench
    # script feeds short fixture turns that the default trigger would skip,
    # so we force every turn into the L1 queue here.
    turn_writer = TurnWriter(backend, extract_trigger=all_of())

    def _worker_factory(counted_extractor) -> ExtractorWorker:
        return ExtractorWorker(
            backend=backend,
            extractor=counted_extractor,
            entity_state=view,
            candidate_inbox=inbox,
            config=ExtractorWorkerConfig(namespace="bench"),
        )

    tiered = TieredBenchIngestor(
        turn_writer=turn_writer,
        extractor=extractor,
        worker_factory=_worker_factory,
        retraction=retraction,
        namespace="bench",
    )

    class _Reader(MemoryReader):
        def list_active_memories(self, ns: str) -> list[str]:
            rows = view.get_active(ns, entity="user")
            return [f"{r.attribute}: {r.value}" for r in rows]

    runner = MemoryBenchRunner(
        tiered,
        _Reader(),
        judge=StubMemoryJudge(),
        answerer=SubstringAnswerer(),
        namespace="bench",
        path_kind=PATH_KIND_TIERED_ASYNC,
        drain_callback=tiered.drain,
        cost_probe=lambda: tiered.extractor_calls,
    )
    return runner, backend.close


# Sleep budget per LLM call. Large enough that p50/p95 differences between
# sync (LLM-in-ingest) and tiered (LLM-in-drain) are well above timer noise
# but small enough that the smoke completes in well under a second.
_LLM_DELAY_S = 0.005


class TestSyncVsTiered:
    """Compare latency / cost attribution between the two write paths."""

    @pytest.mark.asyncio
    async def test_tiered_cost_in_drain(self, tmp_path: Path) -> None:
        sessions = load_synthetic_fixture()

        sync_runner, sync_close = _build_sync_runner(tmp_path, delay_s=_LLM_DELAY_S)
        try:
            sync_report = await sync_runner.run(sessions)
        finally:
            sync_close()

        tiered_runner, tiered_close = _build_tiered_runner(tmp_path, delay_s=_LLM_DELAY_S)
        try:
            tiered_report = await tiered_runner.run(sessions)
        finally:
            tiered_close()

        assert sync_report.timings.path_kind == PATH_KIND_SYNC_INLINE
        assert tiered_report.timings.path_kind == PATH_KIND_TIERED_ASYNC

        # Sync attributes the LLM cost to ingest. Tiered keeps ingest fast
        # (L0 + enqueue only) and pushes the LLM into drain.
        assert sync_report.timings.drain_total_ms == 0.0
        assert tiered_report.timings.drain_total_ms > 0.0
        assert tiered_report.timings.ingest_total_ms < sync_report.timings.ingest_total_ms

        # The bench cost proxy reflects per-turn extractor calls on the
        # tiered path (one per ingested user turn that survived triggers).
        assert tiered_report.timings.extractor_calls > 0

    @pytest.mark.asyncio
    async def test_tiered_recall_matches_sync(self, tmp_path: Path) -> None:
        # Equivalence: after drain, the tiered path must surface the same
        # set of active memories as the sync baseline. Sleep is set to 0
        # because correctness does not depend on timing here.
        sessions = load_synthetic_fixture()

        sync_runner, sync_close = _build_sync_runner(tmp_path, delay_s=0.0)
        try:
            sync_report = await sync_runner.run(sessions)
        finally:
            sync_close()

        tiered_runner, tiered_close = _build_tiered_runner(tmp_path, delay_s=0.0)
        try:
            tiered_report = await tiered_runner.run(sessions)
        finally:
            tiered_close()

        assert (
            sync_report.aggregate.extraction.memory_recall
            == tiered_report.aggregate.extraction.memory_recall
        )
        assert (
            sync_report.aggregate.extraction.predicted_total
            == tiered_report.aggregate.extraction.predicted_total
        )
