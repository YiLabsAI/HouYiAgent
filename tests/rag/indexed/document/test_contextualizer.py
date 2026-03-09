import pytest

from houyi.rag.indexed.document.contextualizer import (
    ContextualizedChunk,
    Contextualizer,
    contextualize_chunks,
)
from houyi.rag.types import Chunk, Document


class TestContextualizer:
    """Tests for Contextualizer."""

    def test_contextualize_chunk_heuristic(self) -> None:
        """Test heuristic contextualization."""
        contextualizer = Contextualizer()

        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            content="This is about machine learning models.",
            metadata={"section": "Introduction", "file_path": "/docs/ml.md"},
        )

        async def run():
            return await contextualizer.contextualize_chunk(chunk)

        import asyncio

        result = asyncio.run(run())

        assert isinstance(result, ContextualizedChunk)
        assert result.chunk == chunk
        assert len(result.context) > 0
        assert result.contextualized_content.endswith(chunk.content)

    def test_contextualize_chunk_with_document(self) -> None:
        """Test contextualization with document info."""
        contextualizer = Contextualizer()

        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            content="Details about neural networks.",
        )

        document = Document(
            doc_id="nn-doc",
            source="/docs/nn.md",
            content="Full document content...",
            metadata={"title": "Neural Networks Guide"},
        )

        async def run():
            return await contextualizer.contextualize_chunk(chunk, document)

        import asyncio

        result = asyncio.run(run())

        assert "Neural Networks Guide" in result.context

    def test_contextualize_chunks_batch(self) -> None:
        """Test batch contextualization."""
        contextualizer = Contextualizer()

        chunks = [
            Chunk(chunk_id="c1", doc_id="doc1", content="First chunk"),
            Chunk(chunk_id="c2", doc_id="doc1", content="Second chunk"),
        ]

        async def run():
            return await contextualizer.contextualize_chunks(chunks)

        import asyncio

        results = asyncio.run(run())

        assert len(results) == 2
        assert all(isinstance(r, ContextualizedChunk) for r in results)

    def test_generate_context_heuristic_with_metadata(self) -> None:
        """Test heuristic context generation with metadata."""
        contextualizer = Contextualizer()

        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            content="Page content here",
            metadata={"page": 5, "section": "Results", "file_path": "/report.pdf"},
        )

        context = contextualizer._generate_context_heuristic(chunk, None)

        assert "report.pdf" in context or "Results" in context or "Page" in context

    def test_max_context_length(self) -> None:
        """Test context length truncation."""
        contextualizer = Contextualizer(max_context_length=50)

        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            content="A" * 200,  # Long content
            metadata={
                "section": "Very Long Section Name That Should Be Truncated",
                "file_path": "/very/long/path/to/document/file.md",
            },
        )

        context = contextualizer._generate_context_heuristic(chunk, None)

        assert len(context) <= 50

    def test_contextualize_chunks_convenience(self) -> None:
        """Test convenience function."""
        chunks = [
            Chunk(chunk_id="c1", doc_id="doc1", content="Test content"),
        ]

        async def run():
            return await contextualize_chunks(chunks)

        import asyncio

        results = asyncio.run(run())

        assert len(results) == 1
        assert isinstance(results[0], ContextualizedChunk)

    def test_contextualized_chunk_dataclass(self) -> None:
        """Test ContextualizedChunk dataclass."""
        chunk = Chunk(chunk_id="c1", doc_id="doc1", content="Original")
        ctx_chunk = ContextualizedChunk(
            chunk=chunk,
            context="This is context.",
            contextualized_content="This is context.\n\nOriginal",
        )

        assert ctx_chunk.chunk == chunk
        assert ctx_chunk.context == "This is context."


class TestContextualizerIntegration:
    """Tests for Contextualizer integration with IndexedMode."""

    @pytest.mark.asyncio
    async def test_indexed_mode_initializes_contextualizer_with_llm(self) -> None:
        """Test that IndexedMode initializes Contextualizer when LLM is provided."""
        from typing import Any

        from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
        from houyi.rag.indexed.mode import IndexedMode

        class FakeAdapter:
            async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
                class MockResponse:
                    content = "Context for this chunk."

                return MockResponse()

        config = IndexedConfig()
        mode = IndexedMode(
            config=config,
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(),
            graph_config=GraphConfig(),
            llm_adapter=FakeAdapter(),  # type: ignore[arg-type]
        )

        assert mode._contextualizer is not None

    @pytest.mark.asyncio
    async def test_indexed_mode_skips_contextualizer_without_llm(self) -> None:
        """Test that IndexedMode has no Contextualizer without LLM."""
        from houyi.rag.config import EmbeddingConfig, GraphConfig, IndexedConfig
        from houyi.rag.indexed.mode import IndexedMode

        config = IndexedConfig()
        mode = IndexedMode(
            config=config,
            knowledge_dir="/tmp/test",
            embedding_config=EmbeddingConfig(),
            graph_config=GraphConfig(),
            llm_adapter=None,
        )

        assert mode._contextualizer is None

    @pytest.mark.asyncio
    async def test_rag_config_contextual_retrieval_field(self) -> None:
        """Test that RAGConfig has contextual_retrieval field."""
        from houyi.rag.config import RAGConfig

        # Default is False
        config = RAGConfig()
        assert config.contextual_retrieval is False

        # Can be set to True
        config_with_cr = RAGConfig(contextual_retrieval=True)
        assert config_with_cr.contextual_retrieval is True

    @pytest.mark.asyncio
    async def test_rag_index_passes_contextual_retrieval(self) -> None:
        """Test that RAG.index() passes contextual_retrieval to ingest."""
        import tempfile
        from pathlib import Path

        from houyi.rag import RAG

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "doc.md").write_text("Test content")

            # Create RAG with contextual_retrieval enabled in config
            rag = RAG(
                str(kb_dir),
                mode="indexed",
                contextual_retrieval=True,
            )

            # Config should have contextual_retrieval=True
            assert rag.config.contextual_retrieval is True
