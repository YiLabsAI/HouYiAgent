"""Tests for the top-level RAG facade."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from houyi.rag import RAG, search
from houyi.rag.config import RAGConfig
from houyi.rag.types import RAGMode


class FakeAdapter:
    """Fake LLM adapter for facade tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def chat(
        self,
        messages: list[Any],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Any:
        response_content = self._responses[self._index % len(self._responses)]
        self._index += 1

        class MockResponse:
            content = response_content

        return MockResponse()


class TestRAGFacade:
    def test_creation_default(self) -> None:
        rag = RAG()
        assert rag.config.mode == RAGMode.AUTO

    def test_instantiation_agentic(self, tmp_path: Path) -> None:
        rag = RAG(mode="agentic", knowledge_dir=str(tmp_path))
        assert rag.config.mode == RAGMode.AGENTIC

    def test_instantiation_indexed(self, tmp_path: Path) -> None:
        rag = RAG(mode="indexed", knowledge_dir=str(tmp_path))
        assert rag.config.mode == RAGMode.INDEXED

    def test_instantiation_auto(self, tmp_path: Path) -> None:
        rag = RAG(mode="auto", knowledge_dir=str(tmp_path))
        assert rag.config.mode == RAGMode.AUTO

    def test_with_config(self) -> None:
        config = RAGConfig(
            mode=RAGMode.AGENTIC,
            knowledge_dir="/custom/path",
        )
        rag = RAG(config=config)
        assert rag.config.mode == RAGMode.AGENTIC
        assert rag.config.knowledge_dir == "/custom/path"

    def test_with_strategies(self) -> None:
        rag = RAG(
            mode="indexed",
            strategies=["bm25", "vector"],
        )
        assert rag.config.mode == RAGMode.INDEXED

    def test_mode_selection_empty_dir(self) -> None:
        rag = RAG(knowledge_dir="/nonexistent/path")
        mode = rag._select_mode("test query")
        assert mode == RAGMode.AGENTIC

    def test_knowledge_dir_property(self) -> None:
        rag = RAG(knowledge_dir="/test/path")
        assert rag.knowledge_dir == "/test/path"

    def test_llm_string_parsing(self) -> None:
        rag = RAG(llm="openai")
        assert rag._llm_adapter is None

        rag2 = RAG(llm="anthropic:claude-3-opus")
        assert rag2._llm_adapter is None

    @pytest.mark.asyncio
    async def test_agentic_query_simple(self, write_knowledge_files) -> None:
        kb_dir = write_knowledge_files({"test.md": "Python is a programming language."})

        rag = RAG(mode="agentic", knowledge_dir=str(kb_dir))
        result = await rag.query("What is Python?")

        assert result.answer is not None
        assert result.mode_used == RAGMode.AGENTIC

    @pytest.mark.asyncio
    async def test_agentic_query_empty_kb(self, knowledge_dir: Path) -> None:
        rag = RAG(mode="agentic", knowledge_dir=str(knowledge_dir))
        result = await rag.query("What is Python?")

        assert result is not None
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_search_function(self, write_knowledge_files) -> None:
        kb_dir = write_knowledge_files({"doc.md": "The answer is 42."})

        result = await search("What is the answer?", knowledge_dir=str(kb_dir))

        assert result.answer is not None

    def test_small_kb_selects_agentic(self, tmp_path: Path) -> None:
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        for i in range(10):
            (kb_dir / f"doc{i}.md").write_text(f"Content {i}")

        rag = RAG(mode="auto", knowledge_dir=str(kb_dir))
        selected = rag._select_mode("test")

        assert selected == RAGMode.AGENTIC

    def test_large_kb_selects_indexed(self, tmp_path: Path) -> None:
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        for i in range(150):
            (kb_dir / f"doc{i}.md").write_text(f"Content {i}")

        rag = RAG(mode="auto", knowledge_dir=str(kb_dir))
        selected = rag._select_mode("test")

        assert selected == RAGMode.INDEXED

    @pytest.mark.asyncio
    async def test_agentic_query_with_llm(self, write_knowledge_files) -> None:
        kb_dir = write_knowledge_files({"doc.md": "RAG is a powerful technique for AI systems."})
        fake_adapter = FakeAdapter(
            [
                '{"keywords": ["RAG", "technique"], "synonyms": {}}',
                "RAG is a technique that combines retrieval with generation. [1]",
            ]
        )

        service = RAG(
            mode="agentic",
            knowledge_dir=str(kb_dir),
            llm_adapter=fake_adapter,
        )
        result = await service.query("What is RAG?")

        assert result.answer

    @pytest.mark.asyncio
    async def test_service_with_llm_adapter(self, write_knowledge_files) -> None:
        kb_dir = write_knowledge_files({"doc.md": "Test content about AI"})
        fake_adapter = FakeAdapter(
            [
                '{"keywords": ["AI"], "synonyms": {}}',
                "AI stands for Artificial Intelligence.",
            ]
        )

        service = RAG(
            mode="agentic",
            knowledge_dir=str(kb_dir),
            llm_adapter=fake_adapter,
        )
        result = await service.query("What is AI?")

        assert result.answer
        assert fake_adapter._index > 0

    def test_service_with_llm_model_config(self) -> None:
        service = RAG(mode="indexed", llm_model="gpt-4")
        assert service.config.llm_model == "gpt-4"

    def test_llm_provider_param(self) -> None:
        rag = RAG(llm_provider="openai", llm_model="gpt-4")
        assert rag.config is not None

    def test_llm_adapter_param(self) -> None:
        from unittest.mock import MagicMock

        mock_adapter = MagicMock()
        rag = RAG(llm_adapter=mock_adapter)
        assert rag._llm_adapter is mock_adapter


class TestRAGIndex:
    @pytest.mark.asyncio
    async def test_index_agentic_mode_noop(self) -> None:
        rag = RAG(mode="agentic")
        stats = await rag.index()
        assert stats["mode"] == "agentic"
        assert stats["documents"] == 0

    @pytest.mark.asyncio
    async def test_index_with_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAG(knowledge_dir=tmpdir, mode="indexed")
            stats = await rag.index()
            assert stats["documents"] == 0
            assert stats["chunks"] == 0


class TestRAGLazyRuntimeCreation:
    @pytest.mark.asyncio
    async def test_ensure_agentic_mode_caches_created_runtime(self) -> None:
        rag = RAG(mode="agentic", knowledge_dir="/tmp/test-agentic")
        created = object()
        calls = 0

        def fake_create() -> object:
            nonlocal calls
            calls += 1
            return created

        rag._create_agentic_mode = fake_create  # type: ignore[method-assign]

        await rag._ensure_agentic_mode()
        await rag._ensure_agentic_mode()

        assert calls == 1
        assert rag._agentic_mode is created

    @pytest.mark.asyncio
    async def test_ensure_indexed_mode_caches_created_runtime(self) -> None:
        rag = RAG(mode="indexed", knowledge_dir="/tmp/test-indexed")
        created = object()
        calls = 0

        def fake_create() -> object:
            nonlocal calls
            calls += 1
            return created

        rag._create_indexed_mode = fake_create  # type: ignore[method-assign]

        await rag._ensure_indexed_mode()
        await rag._ensure_indexed_mode()

        assert calls == 1
        assert rag._indexed_mode is created

    def test_create_indexed_mode_uses_config_owned_index_dir(self) -> None:
        config = RAGConfig(mode=RAGMode.INDEXED, knowledge_dir="/kb", index_dir="/indexes/custom")
        rag = RAG(config=config)

        indexed_mode = rag._create_indexed_mode()

        assert indexed_mode._index_dir == "/indexes/custom"
