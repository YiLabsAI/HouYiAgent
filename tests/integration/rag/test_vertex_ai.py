"""Integration tests using real Vertex AI Gemini model.

These tests require:
1. Google Cloud project with Vertex AI enabled
2. Application Default Credentials or service account key
3. Environment variables:
   - GOOGLE_CLOUD_PROJECT: GCP project ID
   - GOOGLE_CLOUD_LOCATION: Region (default: us-central1)
   - GEMINI_MODEL: Model name (default: gemini-2.5-pro)

Run with: pytest tests/integration/rag/test_vertex_ai.py -v -s
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

from houyi.infrastructure.config.env_config import (
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GOOGLE_CLOUD_PROJECT,
    EnvConfig,
)


def _is_google_genai_installed() -> bool:
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        return False


EnvConfig._reset()
_env = EnvConfig.get()
VERTEX_PROJECT = _env.google_project
VERTEX_API_KEY = _env.google_api_key
VERTEX_CREDENTIALS = _env.google_credentials_path
GOOGLE_GENAI_INSTALLED = _is_google_genai_installed()


def _vertex_skip_reason() -> str | None:
    missing: list[str] = []

    if not GOOGLE_GENAI_INSTALLED:
        missing.append("install google-genai")

    if not VERTEX_PROJECT and not VERTEX_API_KEY:
        if VERTEX_CREDENTIALS:
            missing.append(
                f"set {ENV_GOOGLE_CLOUD_PROJECT} or use a credentials file with project_id"
            )
        else:
            missing.append(f"set {ENV_GOOGLE_CLOUD_PROJECT} or {ENV_GOOGLE_API_KEY}")

    if missing:
        env_snapshot = (
            f"{ENV_GOOGLE_CLOUD_PROJECT}={'set' if VERTEX_PROJECT else 'unset'}, "
            f"{ENV_GOOGLE_API_KEY}={'set' if VERTEX_API_KEY else 'unset'}, "
            f"{ENV_GOOGLE_APPLICATION_CREDENTIALS}={'set' if VERTEX_CREDENTIALS else 'unset'}"
        )
        return (
            "Vertex/Gemini integration prerequisites missing: "
            + "; ".join(missing)
            + ". "
            + env_snapshot
        )

    return None


VERTEX_SKIP_REASON = _vertex_skip_reason()

skip_if_no_vertex = pytest.mark.skipif(
    VERTEX_SKIP_REASON is not None,
    reason=VERTEX_SKIP_REASON or "",
)


@skip_if_no_vertex
class TestVertexAIAdapter:
    @pytest.mark.asyncio
    async def test_vertex_adapter_chat(self) -> None:
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

        adapter = GoogleVertexGeminiAdapter.from_env()

        from houyi.adapters.llm.base import LLMMessage, MessageRole

        messages = [LLMMessage(role=MessageRole.USER, content="Say 'Hello' and nothing else.")]

        response = await adapter.chat(messages, temperature=0.1, max_tokens=50)

        assert response.content
        assert "Hello" in response.content or "hello" in response.content.lower()
        print(f"Vertex AI response: {response.content}")


@skip_if_no_vertex
class TestVertexAIWithRAG:
    @pytest.mark.asyncio
    async def test_keyword_extractor_vertex(self) -> None:
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag.llm import KeywordExtractor

        adapter = GoogleVertexGeminiAdapter.from_env()
        extractor = KeywordExtractor(adapter)

        result = await extractor.extract("What is Retrieval-Augmented Generation?")

        print(f"Extracted keywords: {result}")
        assert "keywords" in result
        assert len(result["keywords"]) > 0
        keywords_lower = [k.lower() for k in result["keywords"]]
        assert any("retrieval" in k or "rag" in k or "generation" in k for k in keywords_lower)

    @pytest.mark.asyncio
    async def test_answer_generator_vertex(self) -> None:
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
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
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
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
        ml_chunk = next(r for r in reranked if "machine learning" in r.content.lower())
        weather_chunk = next(r for r in reranked if "weather" in r.content.lower())
        assert ml_chunk.score > weather_chunk.score


@skip_if_no_vertex
class TestVertexAIEndToEnd:
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_vertex(self) -> None:
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
        from houyi.rag import RAG

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            (kb_dir / "python.md").write_text(
                """
# Python Programming

Python is a high-level, interpreted programming language.
It emphasizes code readability and allows developers to express concepts in fewer lines of code.

## Key Features
- Easy to learn and use
- Extensive standard library
- Strong community support
- Great for data science and AI
"""
            )

            (kb_dir / "javascript.md").write_text(
                """
# JavaScript

JavaScript is a programming language primarily used for web development.
It runs in web browsers and enables interactive web pages.

## Key Features
- Client-side scripting
- Event-driven programming
- Supports both OOP and functional paradigms
"""
            )

            adapter = GoogleVertexGeminiAdapter.from_env()

            service = RAG(
                mode="agentic",
                knowledge_dir=str(kb_dir),
                llm_adapter=adapter,
            )

            result = await service.query("What are the key features of Python?")

            print("\nQuery: What are the key features of Python?")
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence}")
            print(f"Sources: {[s.file_path for s in result.sources]}")

            assert result.answer
            answer_lower = result.answer.lower()
            assert any(term in answer_lower for term in ["python", "easy", "library", "data"])

    @pytest.mark.asyncio
    async def test_entity_extraction_vertex(self) -> None:
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter
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
        entity_names = [e.name.lower() for e in entities]
        assert any("openai" in name or "google" in name for name in entity_names)
        assert any("gpt" in name or "gemini" in name for name in entity_names)


@skip_if_no_vertex
class TestVertexAIStreaming:
    @pytest.mark.asyncio
    async def test_stream_chat(self) -> None:
        from houyi.adapters.llm.base import LLMMessage, MessageRole
        from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

        adapter = GoogleVertexGeminiAdapter.from_env()

        messages = [LLMMessage(role=MessageRole.USER, content="Count from 1 to 5.")]

        chunks = []
        async for chunk in adapter.stream_chat(messages, temperature=0.1, max_tokens=100):
            chunks.append(chunk.content_delta)
            print(f"Received chunk: {chunk.content_delta}")

        full_response = "".join(chunks)
        assert full_response
        print(f"\nFull response: {full_response}")


if __name__ == "__main__":
    import asyncio

    async def quick_test():
        if VERTEX_SKIP_REASON:
            print(VERTEX_SKIP_REASON)
            return

        print(f"Vertex AI configured for project: {VERTEX_PROJECT}")

        try:
            from houyi.adapters.llm.base import LLMMessage, MessageRole
            from houyi.adapters.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter

            adapter = GoogleVertexGeminiAdapter.from_env()
            messages = [LLMMessage(role=MessageRole.USER, content="Say 'OK' if you can hear me.")]
            response = await adapter.chat(messages, max_tokens=10)
            print(f"Vertex AI connection successful: {response.content}")
        except Exception as e:
            print(f"Vertex AI connection failed: {e}")

    asyncio.run(quick_test())
