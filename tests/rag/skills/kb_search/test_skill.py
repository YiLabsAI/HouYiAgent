"""Tests for kb_search skill."""

from __future__ import annotations

import pytest

from houyi.rag.skills.kb_search.skill import KBSearchInput, execute_kb_search


class TestKBSearchSkillExecution:
    @pytest.mark.asyncio
    async def test_execute_skill(self, write_knowledge_files) -> None:
        kb_dir = write_knowledge_files({"doc.md": "Python is a programming language."})

        input_data = KBSearchInput(
            query="What is Python?",
            knowledge_dir=str(kb_dir),
            mode="agentic",
        )
        output = await execute_kb_search(input_data)

        assert output.answer
        assert isinstance(output.confidence, float)
