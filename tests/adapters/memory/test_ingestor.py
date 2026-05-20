"""Memory write-path validation: extraction, resolution, retraction, and update.

The LLM is mocked with hand-built JSON responses so the test exercises
the entire write path without any network or model dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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
    RetractionTarget,
)

# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------


@dataclass
class _Response:
    content: str


class _ScriptedLLM:
    """Stub LLM that returns the next pre-scripted response per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def chat(self, messages, *, temperature: float, max_tokens: int):
        if not self._responses:
            return _Response("[]")
        return _Response(self._responses.pop(0))


def _items(*items: dict) -> str:
    return json.dumps(list(items))


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------


@dataclass
class _Pipeline:
    pipeline: MemoryIngestor
    view: SQLiteEntityStateView
    inbox: SQLiteCandidateInbox
    llm: _ScriptedLLM


@pytest.fixture
def make_pipeline(tmp_path):
    """Factory so each test can preload its own LLM script."""

    def _factory(llm_responses: list[str]) -> _Pipeline:
        backend = SQLiteMemoryBackend(db_path=tmp_path / "memory.db")
        view = SQLiteEntityStateView(backend)
        inbox = SQLiteCandidateInbox(backend)
        tools = MemoryWriterTools(view, inbox, namespace="ws")
        llm = _ScriptedLLM(llm_responses)
        extractor = AtomicFactExtractor(llm)
        orchestrator = RetractionOrchestrator(RetractionDetector(), tools)
        pipe = MemoryIngestor(extractor, orchestrator, tools, inbox)
        return _Pipeline(pipeline=pipe, view=view, inbox=inbox, llm=llm)

    yield _factory


# ---------------------------------------------------------------------------
# Extract cells (E1-E5)
# ---------------------------------------------------------------------------


class TestCellE1MemoryIntegrity:
    """E1: gold facts present in the conversation must end up in the store."""

    @pytest.mark.asyncio
    async def test_gold_facts_admitted(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
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
                )
            ]
        )
        result = await env.pipeline.ingest_turn(
            "My name is Alice and I live in Beijing", source_anchor="m1"
        )
        admitted = [d for d in result.decisions if d.decision == "admitted"]
        assert len(admitted) == 2
        active = env.view.get_active("ws", "user")
        assert {(r.attribute, r.value) for r in active} == {
            ("name", "Alice"),
            ("lives_in", "Beijing"),
        }


class TestCellE2MemoryAccuracy:
    """E2: the value the store records must match what the LLM extracted."""

    @pytest.mark.asyncio
    async def test_value_matches_extraction(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "user",
                        "predicate": "lives_in",
                        "object": "Beijing",
                        "certainty": "certain",
                    },
                )
            ]
        )
        await env.pipeline.ingest_turn("I live in Beijing", source_anchor="m1")
        active = env.view.get_active("ws", "user", "lives_in")
        assert active[0].value == "Beijing"
        # Provenance is preserved for the retrieval-time accuracy audit.
        assert active[0].source_unit_id == "m1"


class TestCellE3VagueFilter:
    """E3: vague extractions must NEVER reach the main store."""

    @pytest.mark.asyncio
    async def test_vague_routed_to_inbox(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "project",
                        "predicate": "status",
                        "object": "stuck",
                        "certainty": "vague",
                    },
                )
            ]
        )
        result = await env.pipeline.ingest_turn("The project is kind of stuck", source_anchor="m1")
        # Main store must be empty.
        assert env.view.get_active("ws", "project") == []
        # Inbox must hold the vague candidate.
        deferred = [d for d in result.decisions if d.decision == "deferred_vague"]
        assert len(deferred) == 1
        parked = env.inbox.list_for("ws", reason="vague")
        assert len(parked) == 1
        assert parked[0].object == "stuck"


class TestCellE4RetractionSignal:
    """E4: a retraction utterance must close the speaker's recent fact."""

    @pytest.mark.asyncio
    async def test_retraction_invalidates_recent_fact(self, make_pipeline) -> None:
        # Turn 1: write a normal fact.
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "user",
                        "predicate": "city",
                        "object": "Beijing",
                        "certainty": "certain",
                    }
                ),
                # Turn 2 LLM script is irrelevant - retraction stage short-circuits.
                "[]",
            ]
        )
        await env.pipeline.ingest_turn("I live in Beijing", source_anchor="m1")
        assert env.view.get_active("ws", "user", "city")[0].value == "Beijing"

        # Turn 2: retraction utterance with the recent target.
        result = await env.pipeline.ingest_turn(
            "Actually, that's not right",
            source_anchor="m2",
            recent_targets=[RetractionTarget("user", "city")],
        )
        assert result.retraction is not None
        assert result.retraction.signal is not None
        assert env.view.get_active("ws", "user", "city") == []
        # No new facts should be written on a pure retraction turn.
        assert result.decisions == []


class TestCellE5SourceAnchor:
    """E5: extractions without a source anchor must not enter the main store."""

    @pytest.mark.asyncio
    async def test_no_anchor_to_inbox(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "user",
                        "predicate": "lives_in",
                        "object": "Beijing",
                        "certainty": "certain",
                    },
                )
            ]
        )
        result = await env.pipeline.ingest_turn("I live in Beijing", source_anchor=None)
        # Main store untouched.
        assert env.view.get_active("ws", "user") == []
        # Inbox holds the sourceless raw payload.
        assert len(result.sourceless_candidates) == 1
        sourceless = env.inbox.list_sourceless("ws")
        assert len(sourceless) == 1
        assert sourceless[0]["object"] == "Beijing"


# ---------------------------------------------------------------------------
# Update cells (U1-U3)
# ---------------------------------------------------------------------------


class TestCellU1UpdateAccCorrect:
    """U1: a newer fact for the same (entity, attribute) replaces the old one."""

    @pytest.mark.asyncio
    async def test_supersession(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "user",
                        "predicate": "city",
                        "object": "Beijing",
                        "certainty": "certain",
                    }
                ),
                _items(
                    {
                        "subject": "user",
                        "predicate": "city",
                        "object": "Shanghai",
                        "certainty": "certain",
                    }
                ),
            ]
        )
        await env.pipeline.ingest_turn("I live in Beijing", source_anchor="m1")
        await env.pipeline.ingest_turn("I moved to Shanghai", source_anchor="m2")

        active = env.view.get_active("ws", "user", "city")
        assert len(active) == 1
        assert active[0].value == "Shanghai"

        history = env.view.get_history("ws", "user", "city")
        assert [h.value for h in history] == ["Shanghai", "Beijing"]
        # Closed-open contract: the older row got a valid_to.
        assert history[1].valid_to is not None


class TestCellU2ConflictResolution:
    """U2: a conflicting fact write must close the prior row in the same call."""

    @pytest.mark.asyncio
    async def test_conflict_invalidates_prior(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "user",
                        "predicate": "job",
                        "object": "engineer",
                        "certainty": "certain",
                    }
                ),
                _items(
                    {
                        "subject": "user",
                        "predicate": "job",
                        "object": "manager",
                        "certainty": "certain",
                    }
                ),
            ]
        )
        await env.pipeline.ingest_turn("I'm an engineer", source_anchor="m1")
        await env.pipeline.ingest_turn("I'm now a manager", source_anchor="m2")

        active = env.view.get_active("ws", "user", "job")
        assert len(active) == 1
        assert active[0].value == "manager"
        # Prior row was closed in the same conflict-handling write.
        history = env.view.get_history("ws", "user", "job")
        prior = next(h for h in history if h.value == "engineer")
        assert prior.valid_to is not None


class TestCellU3BiTemporalValidity:
    """U3: as-of queries must return the value active at that instant."""

    @pytest.mark.asyncio
    async def test_as_of_historical(self, make_pipeline) -> None:
        env = make_pipeline(
            [
                _items(
                    {
                        "subject": "user",
                        "predicate": "city",
                        "object": "Beijing",
                        "certainty": "certain",
                    }
                ),
                _items(
                    {
                        "subject": "user",
                        "predicate": "city",
                        "object": "Shanghai",
                        "certainty": "certain",
                    }
                ),
            ]
        )
        # Force deterministic valid_from values via direct view writes
        # so the as-of query has a known timeline.
        env.view.upsert("ws", "user", "city", "Beijing", valid_from=100.0)
        env.view.upsert("ws", "user", "city", "Shanghai", valid_from=200.0)

        # Mid-Beijing era.
        as_150 = env.view.get_as_of("ws", "user", 150.0, "city")
        assert as_150[0].value == "Beijing"
        # Mid-Shanghai era.
        as_250 = env.view.get_as_of("ws", "user", 250.0, "city")
        assert as_250[0].value == "Shanghai"
