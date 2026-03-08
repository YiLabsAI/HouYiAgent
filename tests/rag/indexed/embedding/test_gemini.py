"""Tests for Gemini embedder limits and auth behavior."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.infrastructure.config.env_config import (
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_CLOUD_PROJECT,
)


class TestGeminiEmbedderConfig:
    def test_default_batch_size_is_2(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        assert GeminiEmbedder.DEFAULT_BATCH_SIZE == 2, (
            f"DEFAULT_BATCH_SIZE should be 2 to stay under 20K token limit, "
            f"but got {GeminiEmbedder.DEFAULT_BATCH_SIZE}"
        )

    def test_default_delay_seconds(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        assert GeminiEmbedder.DEFAULT_DELAY_SECONDS >= 2.0, (
            "DEFAULT_DELAY_SECONDS should be at least 2.0 to avoid rate limiting"
        )

    def test_max_retries_configured(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        assert GeminiEmbedder.MAX_RETRIES >= 3, (
            "MAX_RETRIES should be at least 3 for robust retry logic"
        )

    def test_custom_batch_size_and_delay(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            batch_size=5,
            delay_seconds=2.0,
        )

        assert embedder.batch_size == 5
        assert embedder.delay_seconds == 2.0


class TestGeminiEmbedderAuth:
    def test_api_key_auth_preferred(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(model="text-embedding-004", dimension=768)
        with patch.dict(os.environ, {ENV_GOOGLE_API_KEY: "test-api-key"}):
            mock_genai = MagicMock()
            with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                with patch("houyi.rag.indexed.embedding.gemini.GeminiEmbedder._ensure_client"):
                    embedder._ensure_client()

    def test_error_message_mentions_both_auth_methods(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(model="text-embedding-004", dimension=768)
        env_override = {
            ENV_GOOGLE_API_KEY: "",
            ENV_GOOGLE_CLOUD_PROJECT: "",
        }
        with patch.dict(os.environ, env_override):
            mock_genai_module = MagicMock()
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "google": MagicMock(genai=mock_genai_module),
                        "google.genai": mock_genai_module,
                    },
                ),
                pytest.raises(ValueError, match=ENV_GOOGLE_API_KEY),
            ):
                embedder._client = None
                embedder._ensure_client()

    @pytest.mark.asyncio
    async def test_embed_batch_respects_batch_size(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            location="us-central1",
            batch_size=2,
            delay_seconds=0.0001,
        )

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768

        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        embedder._client = mock_client

        texts = ["text1", "text2", "text3", "text4", "text5"]

        async def mock_embed_batch_single(batch):
            await mock_client.aio.models.embed_content(model="test", contents=batch)
            return [[0.1] * 768 for _ in batch], False

        embedder._embed_batch_single = mock_embed_batch_single
        await embedder.embed_batch(texts)

        assert mock_client.aio.models.embed_content.call_count == 3

        calls = mock_client.aio.models.embed_content.call_args_list
        batch_sizes = [len(call.kwargs["contents"]) for call in calls]
        assert batch_sizes == [2, 2, 1], f"Expected [2, 2, 1], got {batch_sizes}"

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            batch_size=2,
            delay_seconds=0.0001,
        )

        async def mock_embed_batch_single(batch):
            return [[0.1] * 768 for _ in batch], False

        embedder._embed_batch_single = mock_embed_batch_single
        embedder._client = MagicMock()

        progress_calls = []

        def progress_callback(processed, total, batch_size):
            progress_calls.append((processed, total, batch_size))

        texts = ["text1", "text2", "text3", "text4"]
        await embedder.embed_batch(texts, progress_callback=progress_callback)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (2, 4, 2)
        assert progress_calls[1] == (4, 4, 2)

    @pytest.mark.asyncio
    async def test_adaptive_delay_on_rate_limit(self):
        from houyi.rag.indexed.embedding.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            batch_size=2,
            delay_seconds=1.0,
        )

        call_count = 0

        async def mock_embed_batch_single(batch):
            nonlocal call_count
            call_count += 1
            rate_limited = call_count == 1
            return [[0.1] * 768 for _ in batch], rate_limited

        embedder._embed_batch_single = mock_embed_batch_single
        embedder._client = MagicMock()

        texts = ["text1", "text2", "text3", "text4"]
        with patch("houyi.rag.indexed.embedding.gemini.asyncio.sleep", new=AsyncMock()):
            result = await embedder.embed_batch(texts)

        assert len(result) == 4
        assert call_count == 2
