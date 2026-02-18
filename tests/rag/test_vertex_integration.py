"""Integration tests using real Vertex AI Gemini model.

These tests require:
1. Google Cloud project with Vertex AI enabled
2. Application Default Credentials or service account key
3. Environment variables:
   - GOOGLE_CLOUD_PROJECT: GCP project ID
   - GOOGLE_CLOUD_LOCATION: Region (default: us-central1)
   - GEMINI_MODEL: Model name (default: gemini-2.5-pro)

Run with: pytest tests/rag/test_vertex_integration.py -v -s
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env so Vertex AI credentials are available
load_dotenv()

# Check if Vertex AI is configured
VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_AVAILABLE = VERTEX_PROJECT is not None

# Check if Vertex AI SDK is installed
VERTEX_SDK_INSTALLED = importlib.util.find_spec("vertexai") is not None

skip_if_no_vertex = pytest.mark.skipif(
    not (VERTEX_AVAILABLE and VERTEX_SDK_INSTALLED),
    reason="Vertex AI not configured or SDK not installed. "
    "Set GOOGLE_CLOUD_PROJECT and install google-cloud-aiplatform",
)


@skip_if_no_vertex
class TestVertexAIAdapter:
    """Test Vertex AI adapter directly."""

    @pytest.mark.asyncio
    async def test_vertex_adapter_chat(self) -> None:
        """Test basic chat with Vertex AI."""
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

        adapter = GoogleVertexGeminiAdapter.from_env()

        from houyi.llm.base import LLMMessage, MessageRole

        messages = [LLMMessage(role=MessageRole.USER, content="Say 'Hello' and nothing else.")]

        response = await adapter.chat(messages, temperature=0.1, max_tokens=50)

        assert response.content
        assert "Hello" in response.content or "hello" in response.content.lower()
        print(f"Vertex AI response: {response.content}")


@skip_if_no_vertex
class TestVertexAIWithRAG:
    """Test RAG components with real Vertex AI."""

    @pytest.mark.asyncio
    async def test_keyword_extractor_vertex(self) -> None:
        """Test keyword extraction with Vertex AI."""
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag.llm import KeywordExtractor

        adapter = GoogleVertexGeminiAdapter.from_env()
        extractor = KeywordExtractor(adapter)

        result = await extractor.extract("What is Retrieval-Augmented Generation?")

        print(f"Extracted keywords: {result}")
        assert "keywords" in result
        assert len(result["keywords"]) > 0
        # Should extract relevant keywords like RAG, Retrieval, Generation
        keywords_lower = [k.lower() for k in result["keywords"]]
        assert any("retrieval" in k or "rag" in k or "generation" in k for k in keywords_lower)

    @pytest.mark.asyncio
    async def test_answer_generator_vertex(self) -> None:
        """Test answer generation with Vertex AI."""
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag.llm import AnswerGenerator
        from houyi.rag.types import SearchResult, Source

        adapter = GoogleVertexGeminiAdapter.from_env()
        generator = AnswerGenerator(adapter)

        results = [
            SearchResult(
                chunk_id="chunk-1",
                content="RAG (Retrieval-Augmented Generation) is a technique that combines "
                "retrieval of relevant documents with language model generation.",
                score=0.95,
                source=Source(file_path="rag_overview.md"),
            ),
            SearchResult(
                chunk_id="chunk-2",
                content="RAG systems first retrieve relevant passages from a knowledge base, "
                "then use these passages as context for the language model.",
                score=0.90,
                source=Source(file_path="rag_details.md"),
            ),
        ]

        answer, confidence = await generator.generate(
            "What is RAG?",
            results,
            include_sources=True,
        )

        print(f"Generated answer: {answer}")
        print(f"Confidence: {confidence}")

        assert answer
        assert "RAG" in answer or "retrieval" in answer.lower()
        assert confidence > 0

    @pytest.mark.asyncio
    async def test_reranker_vertex(self) -> None:
        """Test reranking with Vertex AI."""
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag.llm import LLMReranker
        from houyi.rag.types import SearchResult

        adapter = GoogleVertexGeminiAdapter.from_env()
        reranker = LLMReranker(adapter)

        results = [
            SearchResult(
                chunk_id="chunk-1",
                content="Python is a programming language known for readability.",
                score=0.5,
            ),
            SearchResult(
                chunk_id="chunk-2",
                content="Machine learning uses Python for data science tasks.",
                score=0.5,
            ),
            SearchResult(
                chunk_id="chunk-3",
                content="The weather today is sunny and warm.",
                score=0.5,
            ),
        ]

        reranked = await reranker.rerank(
            "What programming language is good for machine learning?",
            results,
            top_k=3,
        )

        print("Reranked results:")
        for r in reranked:
            print(f"  Score {r.score:.2f}: {r.content[:50]}...")

        assert len(reranked) == 3
        # The ML-related content should rank higher than weather
        ml_chunk = next(r for r in reranked if "machine learning" in r.content.lower())
        weather_chunk = next(r for r in reranked if "weather" in r.content.lower())
        assert ml_chunk.score > weather_chunk.score


@skip_if_no_vertex
class TestVertexAIEndToEnd:
    """End-to-end tests with Vertex AI and RAG."""

    @pytest.mark.asyncio
    async def test_full_rag_pipeline_vertex(self) -> None:
        """Test complete RAG pipeline with Vertex AI."""
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag import RAG

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create test knowledge base
            (kb_dir / "python.md").write_text("""
# Python Programming

Python is a high-level, interpreted programming language.
It emphasizes code readability and allows developers to express concepts in fewer lines of code.

## Key Features
- Easy to learn and use
- Extensive standard library
- Strong community support
- Great for data science and AI
""")

            (kb_dir / "javascript.md").write_text("""
# JavaScript

JavaScript is a programming language primarily used for web development.
It runs in web browsers and enables interactive web pages.

## Key Features
- Client-side scripting
- Event-driven programming
- Supports both OOP and functional paradigms
""")

            # Create RAG service with Vertex AI
            adapter = GoogleVertexGeminiAdapter.from_env()

            service = RAG(
                mode="agentic",
                knowledge_dir=str(kb_dir),
                llm_adapter=adapter,
            )

            # Query about Python
            result = await service.query("What are the key features of Python?")

            print("\nQuery: What are the key features of Python?")
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence}")
            print(f"Sources: {[s.file_path for s in result.sources]}")

            assert result.answer
            # Answer should mention Python features
            answer_lower = result.answer.lower()
            assert any(term in answer_lower for term in ["python", "easy", "library", "data"])

    @pytest.mark.asyncio
    async def test_entity_extraction_vertex(self) -> None:
        """Test entity extraction with Vertex AI."""
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag.llm import LLMEntityExtractor
        from houyi.rag.types import Chunk

        adapter = GoogleVertexGeminiAdapter.from_env()
        extractor = LLMEntityExtractor(adapter)

        chunk = Chunk(
            chunk_id="test-chunk",
            doc_id="doc-1",
            content="""
            OpenAI developed GPT-4, a large language model.
            Google created Gemini, which competes with GPT-4.
            Both companies are leading the AI industry.
            """,
        )

        entities, relations = await extractor.extract(chunk)

        print("\nExtracted entities:")
        for e in entities:
            print(f"  - {e.name} ({e.entity_type})")

        print("\nExtracted relations:")
        for r in relations:
            print(
                f"  - {r.metadata.get('source_name')} -> {r.metadata.get('target_name')} ({r.rel_type})"
            )

        assert len(entities) >= 2
        # Should extract company and product entities
        entity_names = [e.name.lower() for e in entities]
        assert any("openai" in name or "google" in name for name in entity_names)
        assert any("gpt" in name or "gemini" in name for name in entity_names)


@skip_if_no_vertex
class TestVertexAIStreaming:
    """Test streaming with Vertex AI."""

    @pytest.mark.asyncio
    async def test_stream_chat(self) -> None:
        """Test streaming response from Vertex AI."""
        from houyi.llm.base import LLMMessage, MessageRole
        from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

        adapter = GoogleVertexGeminiAdapter.from_env()

        messages = [LLMMessage(role=MessageRole.USER, content="Count from 1 to 5.")]

        chunks = []
        async for chunk in adapter.stream_chat(messages, temperature=0.1, max_tokens=100):
            chunks.append(chunk)
            print(f"Received chunk: {chunk}")

        full_response = "".join(chunks)
        assert full_response
        print(f"\nFull response: {full_response}")


if __name__ == "__main__":
    # Run a quick test to verify setup
    import asyncio

    async def quick_test():
        if not VERTEX_AVAILABLE:
            print("❌ GOOGLE_CLOUD_PROJECT not set")
            print("Set environment variable: export GOOGLE_CLOUD_PROJECT=your-project-id")
            return

        if not VERTEX_SDK_INSTALLED:
            print("❌ Vertex AI SDK not installed")
            print("Install with: pip install google-cloud-aiplatform")
            return

        print(f"✅ Vertex AI configured for project: {VERTEX_PROJECT}")

        try:
            from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

            adapter = GoogleVertexGeminiAdapter.from_env()

            from houyi.llm.base import LLMMessage, MessageRole

            messages = [LLMMessage(role=MessageRole.USER, content="Say 'OK' if you can hear me.")]

            response = await adapter.chat(messages, max_tokens=10)
            print(f"✅ Vertex AI connection successful: {response.content}")
        except Exception as e:
            print(f"❌ Vertex AI connection failed: {e}")

    asyncio.run(quick_test())
