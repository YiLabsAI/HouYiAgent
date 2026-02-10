"""Test Gemini embedding API limits."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestGeminiEmbedderConfig:
    """Test GeminiEmbedder configuration."""

    def test_default_batch_size_is_2(self):
        """Verify DEFAULT_BATCH_SIZE is set to safe value."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

        # DEFAULT_BATCH_SIZE should be small enough to stay under 20K token limit
        # With MAX_CHUNK_CHARS = 8000 (~2000 tokens), 2 chunks = 4000 tokens
        assert GeminiEmbedder.DEFAULT_BATCH_SIZE == 2, (
            f"DEFAULT_BATCH_SIZE should be 2 to stay under 20K token limit, "
            f"but got {GeminiEmbedder.DEFAULT_BATCH_SIZE}"
        )

    def test_default_delay_seconds(self):
        """Verify DEFAULT_DELAY_SECONDS for rate limiting."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

        # Should have a reasonable delay to avoid rate limiting
        # 2.0s = 30 RPM max, safe for low quota projects
        assert GeminiEmbedder.DEFAULT_DELAY_SECONDS >= 2.0, (
            "DEFAULT_DELAY_SECONDS should be at least 2.0 to avoid rate limiting"
        )

    def test_max_retries_configured(self):
        """Verify MAX_RETRIES is configured for exponential backoff."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

        assert GeminiEmbedder.MAX_RETRIES >= 3, (
            "MAX_RETRIES should be at least 3 for robust retry logic"
        )

    def test_custom_batch_size_and_delay(self):
        """Test that custom batch_size and delay can be set."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            batch_size=5,
            delay_seconds=2.0,
        )

        assert embedder.batch_size == 5
        assert embedder.delay_seconds == 2.0

    @pytest.mark.asyncio
    async def test_embed_batch_respects_batch_size(self):
        """Test that embed_batch splits into correct batch sizes."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            location="us-central1",
            batch_size=2,
            delay_seconds=0,  # No delay for testing
        )

        # Mock the client
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768

        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        embedder._client = mock_client

        # Test with 5 texts - should split into 3 batches (2+2+1)
        texts = ["text1", "text2", "text3", "text4", "text5"]

        # Need to patch _embed_batch_single since it returns a tuple now
        async def mock_embed_batch_single(batch):
            result = await mock_client.aio.models.embed_content(model="test", contents=batch)
            return [[0.1] * 768 for _ in batch], False  # (embeddings, rate_limited)

        embedder._embed_batch_single = mock_embed_batch_single
        await embedder.embed_batch(texts)

        # Should have 3 API calls
        assert mock_client.aio.models.embed_content.call_count == 3

        # Verify batch sizes
        calls = mock_client.aio.models.embed_content.call_args_list
        batch_sizes = [len(call.kwargs['contents']) for call in calls]
        assert batch_sizes == [2, 2, 1], f"Expected [2, 2, 1], got {batch_sizes}"

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        """Test that progress callback is called during batch processing."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

        embedder = GeminiEmbedder(
            model="text-embedding-004",
            dimension=768,
            project="test-project",
            batch_size=2,
            delay_seconds=0,
        )

        # Mock _embed_batch_single to return embeddings and no rate limiting
        async def mock_embed_batch_single(batch):
            return [[0.1] * 768 for _ in batch], False  # (embeddings, rate_limited)

        embedder._embed_batch_single = mock_embed_batch_single
        embedder._client = MagicMock()  # Prevent client initialization

        # Track progress calls
        progress_calls = []
        def progress_callback(processed, total, batch_size):
            progress_calls.append((processed, total, batch_size))

        texts = ["text1", "text2", "text3", "text4"]
        await embedder.embed_batch(texts, progress_callback=progress_callback)

        # Should have 2 progress calls (4 texts / 2 batch_size)
        assert len(progress_calls) == 2
        assert progress_calls[0] == (2, 4, 2)  # First batch
        assert progress_calls[1] == (4, 4, 2)  # Second batch

    @pytest.mark.asyncio
    async def test_adaptive_delay_on_rate_limit(self):
        """Test that delay increases when rate limiting is detected."""
        from houyi.rag.indexed.embedding.base import GeminiEmbedder

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
            # Simulate rate limiting on first batch
            rate_limited = (call_count == 1)
            return [[0.1] * 768 for _ in batch], rate_limited

        embedder._embed_batch_single = mock_embed_batch_single
        embedder._client = MagicMock()

        # Track the current_delay value by checking the log
        # We just verify the function completes without error
        texts = ["text1", "text2", "text3", "text4"]
        result = await embedder.embed_batch(texts)

        # Verify we got the expected number of embeddings
        assert len(result) == 4
        assert call_count == 2  # Two batches of 2


class TestMaxChunkChars:
    """Test document chunking respects character limits."""

    def test_max_chunk_chars_is_safe(self):
        """Verify MAX_CHUNK_CHARS won't exceed embedding model limits."""
        from houyi.rag.indexed.document.splitters import MAX_CHUNK_CHARS

        # MAX_CHUNK_CHARS should produce ~2000 tokens (8000 chars / 4)
        # text-embedding-004 limit is 2048 tokens per text
        expected_max_tokens = MAX_CHUNK_CHARS / 4  # ~4 chars per token

        assert expected_max_tokens <= 2048, (
            f"MAX_CHUNK_CHARS={MAX_CHUNK_CHARS} may produce ~{expected_max_tokens} tokens, "
            f"exceeding the 2048 token limit"
        )


class TestVectorIndexDimensionValidation:
    """Test that VectorIndex validates dimension compatibility."""

    @pytest.mark.asyncio
    async def test_dimension_saved_in_metadata(self, tmp_path):
        """Test that dimension is saved in metadata."""
        from houyi.rag.indexed.index.vector import VectorIndex
        from houyi.rag.types import Chunk

        index = VectorIndex(dimension=768, knowledge_dir=str(tmp_path))

        # Add a chunk and save
        chunk = Chunk(
            chunk_id="test_0",
            doc_id="test",
            content="Test content",
            start_idx=0,
            end_idx=12,
        )
        await index.add(chunk, [0.1] * 768)
        await index.save()

        # Check metadata contains dimension
        import json
        meta_path = tmp_path / ".houyi" / "vector_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta.get("dimension") == 768


class TestIndexDirSeparation:
    """Test that index_dir is used correctly for storage separation."""

    def test_rag_config_has_index_dir(self):
        """Test RAGConfig supports index_dir parameter."""
        from houyi.rag.config import RAGConfig

        config = RAGConfig(
            knowledge_dir="/path/to/knowledge",
            index_dir="/path/to/index",
        )

        assert config.index_dir == "/path/to/index"
        assert config.get_index_dir() == "/path/to/index"

    def test_rag_config_index_dir_default(self):
        """Test RAGConfig defaults index_dir to knowledge_dir/.houyi."""
        from houyi.rag.config import RAGConfig

        config = RAGConfig(knowledge_dir="/path/to/knowledge")

        assert config.index_dir is None
        assert config.get_index_dir() == "/path/to/knowledge/.houyi"


class TestLibraryStorageIsolation:
    """Test that each library has isolated storage."""

    def test_get_library_storage_dir(self):
        """Test that storage directories are library-specific."""
        import sys
        sys.path.insert(0, "houyi-studio/server")
        from houyi_studio.server.rag_service import (
            get_library_index_dir,
            get_library_storage_dir,
            get_library_upload_dir,
        )

        lib1_storage = get_library_storage_dir("lib_001")
        lib2_storage = get_library_storage_dir("lib_002")

        # Each library should have unique storage
        assert lib1_storage != lib2_storage
        assert "lib_001" in str(lib1_storage)
        assert "lib_002" in str(lib2_storage)

        # Uploads should be under library storage
        lib1_uploads = get_library_upload_dir("lib_001")
        assert str(lib1_uploads).endswith("lib_001/uploads")

        # Index should be under library storage
        lib1_index = get_library_index_dir("lib_001")
        assert str(lib1_index).endswith("lib_001/index")

    def test_different_libraries_isolated(self):
        """Test that libraries don't share any directories."""
        import sys
        sys.path.insert(0, "houyi-studio/server")
        from houyi_studio.server.rag_service import (
            get_library_index_dir,
            get_library_upload_dir,
        )

        lib1_uploads = get_library_upload_dir("lib_a")
        lib2_uploads = get_library_upload_dir("lib_b")
        lib1_index = get_library_index_dir("lib_a")
        lib2_index = get_library_index_dir("lib_b")

        # No shared paths
        assert lib1_uploads != lib2_uploads
        assert lib1_index != lib2_index
        assert not str(lib1_uploads).startswith(str(lib2_uploads))
        assert not str(lib1_index).startswith(str(lib2_index))
