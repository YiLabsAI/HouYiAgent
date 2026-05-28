"""Integration tests for Memory ↔ Chat pipeline.

Validates the full flow:
1. Multi-turn conversation → memory extraction → stored
2. New turn → memory recall → injected into ContextPlanner
3. Degraded path (no embedding, no memories) works gracefully
"""

from __future__ import annotations

import pytest

from houyi.adapters.embedding import NoOpEmbeddingProvider
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import (
    CandidateStatus,
    ExtractionContext,
    MemoryPolicy,
)
from houyi.application.context.context_planner import ContextPlanner
from houyi.application.context.context_renderer import ContextRenderer
from houyi.application.context.token_estimator import TokenEstimator


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Engine with auto-approve and NoOp embeddings.

    Module scope avoids repeated MemoryStore/MemoryEngine creation
    (~0.15s each) while still providing fresh state across modules.
    """
    tmp_path = tmp_path_factory.mktemp("memory_chat")
    store = MemoryStore(data_dir=tmp_path)
    emb = NoOpEmbeddingProvider(dim=32)
    policy = MemoryPolicy(auto_approve=True)
    yield MemoryEngine(store, embedding_provider=emb, policy=policy)
    store.close()


@pytest.fixture(scope="module")
def planner() -> ContextPlanner:
    estimator = TokenEstimator(model="gpt-4o-mini")
    return ContextPlanner(
        token_estimator=estimator,
        system_instructions="You are a helpful assistant.",
    )


@pytest.fixture(scope="module")
def renderer() -> ContextRenderer:
    return ContextRenderer()


class TestRecallInjectsContext:
    """Memory recall results appear in ContextPlanner output."""

    async def test_recall_injects_into_plan(
        self,
        engine: MemoryEngine,
        planner: ContextPlanner,
        renderer: ContextRenderer,
    ):
        turn1 = [
            {"role": "user", "content": "Remember that my preferred language is Python."},
            {"role": "assistant", "content": "Got it, I'll remember that."},
        ]
        await engine.process_messages(turn1)

        memory_text = await engine.build_context(
            query="What programming language should I use?",
        )
        assert memory_text is not None
        assert "python" in memory_text.lower() or "prefer" in memory_text.lower()

        messages = [
            {"role": "user", "content": "What language should I use for this project?"},
        ]
        plan = planner.plan(messages, memory_context=memory_text)
        rendered = renderer.render(plan)

        memory_messages = [m for m in rendered if "[Memory Context]" in m.get("content", "")]
        assert len(memory_messages) >= 1

    async def test_no_memory_no_block(
        self,
        engine: MemoryEngine,
        planner: ContextPlanner,
        renderer: ContextRenderer,
    ):
        memory_text = await engine.build_context(query="hello")
        assert memory_text is None

        messages = [{"role": "user", "content": "Hello!"}]
        plan = planner.plan(messages, memory_context=memory_text)
        rendered = renderer.render(plan)

        memory_messages = [m for m in rendered if "[Memory Context]" in m.get("content", "")]
        assert len(memory_messages) == 0


class TestExtractionAfterTurn:
    """Memory extraction processes user messages after each turn."""

    async def test_extracts_from_user_messages(self, engine: MemoryEngine):
        messages = [
            {"role": "user", "content": "My name is Bob and I prefer dark themes."},
            {"role": "assistant", "content": "Nice to meet you, Bob!"},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) >= 1
        approved = [c for c in candidates if c.status == CandidateStatus.APPROVED]
        assert len(approved) >= 1

    async def test_empty_turn_no_candidates(self, engine: MemoryEngine):
        candidates = await engine.process_messages([])
        assert candidates == []

    async def test_assistant_only_no_candidates(self, engine: MemoryEngine):
        messages = [
            {"role": "assistant", "content": "I remember everything."},
        ]
        candidates = await engine.process_messages(messages)
        assert candidates == []


class TestMultiTurnConversation:
    """Simulate realistic multi-turn chat with memory accumulation."""

    async def test_accumulated_recall(self, engine: MemoryEngine):
        turn1 = [
            {"role": "user", "content": "Remember that the database is PostgreSQL."},
            {"role": "assistant", "content": "Noted."},
        ]
        await engine.process_messages(
            turn1,
            ExtractionContext(turn_index=1),
        )

        turn2 = [
            {"role": "user", "content": "Remember that the API framework is FastAPI."},
            {"role": "assistant", "content": "Got it."},
        ]
        await engine.process_messages(
            turn2,
            ExtractionContext(turn_index=2),
        )

        records = engine.store.all_records()
        assert len(records) >= 2

        recalls = await engine.recall("database framework")
        assert len(recalls) >= 1

    async def test_context_grows_over_turns(
        self,
        engine: MemoryEngine,
        planner: ContextPlanner,
        renderer: ContextRenderer,
    ):
        for i in range(3):
            turn = [
                {"role": "user", "content": f"Remember that fact number {i} is important."},
                {"role": "assistant", "content": "Stored."},
            ]
            await engine.process_messages(
                turn,
                ExtractionContext(turn_index=i),
            )

        memory_text = await engine.build_context(query="important facts")
        assert memory_text is not None

        plan = planner.plan(
            [{"role": "user", "content": "What are the important facts?"}],
            memory_context=memory_text,
        )
        rendered = renderer.render(plan)
        full_text = " ".join(m.get("content", "") for m in rendered)
        assert "Memory Context" in full_text


class TestNoEmbeddingDegradation:
    """Works with lexical-only retrieval."""

    async def test_lexical_recall_works(self, tmp_path):
        store = MemoryStore(data_dir=tmp_path)
        engine = MemoryEngine(store, policy=MemoryPolicy(auto_approve=True))

        turn = [
            {"role": "user", "content": "Remember that Redis runs on port 6379."},
            {"role": "assistant", "content": "OK."},
        ]
        await engine.process_messages(turn)

        text = await engine.build_context("Redis port number")
        assert text is not None
        assert "6379" in text or "redis" in text.lower()
