"""Live integration test: Memory extraction with real LLM.

Validates that LLM-based memory extraction actually works with a real
LLM provider. Requires LLM_API_KEY / provider configuration.

Run: HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 uv run pytest tests/integration/live/test_memory_extraction_live.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore
from houyi.adapters.memory.types import MemoryPolicy, MemoryType

load_dotenv()

_SKIP_REASON = "Set HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 to run"
_should_skip = os.getenv("HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS") != "1"


def _create_engine(tmp_path) -> MemoryEngine:
    from houyi.adapters.llm.factory import LLMAdapterFactory

    llm = LLMAdapterFactory.create()
    store = MemoryStore(data_dir=tmp_path / "memory")
    return MemoryEngine(
        store,
        llm_adapter=llm,
        policy=MemoryPolicy(auto_approve=True),
    )


@pytest.mark.skipif(_should_skip, reason=_SKIP_REASON)
class TestLiveExtraction:
    """Real LLM extraction — validates prompt + parsing works end-to-end."""

    async def test_english_preferences(self, tmp_path):
        engine = _create_engine(tmp_path)
        messages = [
            {
                "role": "user",
                "content": "My name is Alice. I prefer Python for data science. Never use tabs.",
            },
            {"role": "assistant", "content": "Got it, Alice! I'll use Python with spaces."},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) >= 2

        types = {c.memory_type for c in candidates}
        assert MemoryType.PROFILE in types or any("Alice" in c.content for c in candidates)

    async def test_chinese_preferences(self, tmp_path):
        engine = _create_engine(tmp_path)
        messages = [
            {"role": "user", "content": "我是梅西，我喜欢研究 AI Agent，不要用 Java。"},
            {"role": "assistant", "content": "好的，梅西。"},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) >= 1
        contents = " ".join(c.content for c in candidates)
        has_name = any(k in contents for k in ("梅西", "Mei", "name"))
        has_pref = any(k in contents for k in ("AI", "Agent", "prefer"))
        assert has_name or has_pref

    async def test_no_memory_in_greeting(self, tmp_path):
        engine = _create_engine(tmp_path)
        messages = [
            {"role": "user", "content": "Hello, how are you today?"},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) == 0

    async def test_recall_after_extraction(self, tmp_path):
        engine = _create_engine(tmp_path)
        messages = [
            {"role": "user", "content": "Remember that our database is PostgreSQL on port 5432."},
            {"role": "assistant", "content": "Noted."},
        ]
        await engine.process_messages(messages)
        records = engine.store.all_records()
        assert len(records) >= 1

        text = await engine.build_context("What database do we use?")
        assert text is not None
        has_pg = "postgres" in text.lower() or "5432" in text
        assert has_pg

    async def test_multi_language_batch(self, tmp_path):
        engine = _create_engine(tmp_path)
        messages = [
            {"role": "user", "content": "I prefer dark mode. 我的邮箱是 test@example.com"},
            {"role": "assistant", "content": "OK."},
        ]
        candidates = await engine.process_messages(messages)
        assert len(candidates) >= 1
        contents = " ".join(c.content for c in candidates)
        has_dark = "dark" in contents.lower()
        has_email = "test@example.com" in contents or "email" in contents.lower()
        assert has_dark or has_email
