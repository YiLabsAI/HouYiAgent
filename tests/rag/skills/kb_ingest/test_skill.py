"""Tests for kb-ingest skill."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import houyi.rag.skills.kb_ingest.skill as kb_ingest_module
from houyi.rag.skills.kb_ingest.skill import (
    KBIngestInput,
    KBIngestOutput,
    execute_kb_ingest,
    kb_ingest_skill,
)


class TestModels:
    """Tests for data models."""

    def test_kb_ingest_input_defaults(self) -> None:
        """Test KBIngestInput defaults."""
        input_data = KBIngestInput(paths=["/path/to/docs"])
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.mode == "incremental"
        assert input_data.build_graph is False
        assert input_data.chunk_size == 512
        assert input_data.chunk_overlap == 64

    def test_kb_ingest_input_custom(self) -> None:
        """Test KBIngestInput custom values."""
        input_data = KBIngestInput(
            paths=["/docs", "/more"],
            knowledge_dir="/kb/",
            mode="full",
            build_graph=True,
            chunk_size=1024,
            chunk_overlap=128,
        )
        assert input_data.paths == ["/docs", "/more"]
        assert input_data.knowledge_dir == "/kb/"
        assert input_data.mode == "full"
        assert input_data.build_graph is True

    def test_kb_ingest_output_defaults(self) -> None:
        """Test KBIngestOutput defaults."""
        output = KBIngestOutput(success=True)
        assert output.success is True
        assert output.documents_processed == 0
        assert output.chunks_created == 0
        assert output.index_path == ""
        assert output.message == ""

    def test_kb_ingest_output_full(self) -> None:
        """Test KBIngestOutput with all fields."""
        output = KBIngestOutput(
            success=True,
            documents_processed=10,
            chunks_created=50,
            index_path="/kb/.houyi",
            message="Done",
        )
        assert output.documents_processed == 10
        assert output.chunks_created == 50


class TestExecuteKBIngest:
    """Tests for execute_kb_ingest function."""

    @pytest.mark.asyncio
    async def test_execute_success(self, patch_skill_rag_builder) -> None:
        """Test successful ingest execution."""
        mock_result = MagicMock()
        mock_result.documents_processed = 5
        mock_result.chunks_created = 25
        mock_result.index_path = "/kb/.houyi"

        mock_rag = MagicMock()
        mock_rag.index = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_ingest_module, mock_rag):
            input_data = KBIngestInput(paths=["/docs"])
            result = await execute_kb_ingest(input_data)

        assert result.success is True
        assert result.documents_processed == 5
        assert result.chunks_created == 25
        assert result.index_path == "/kb/.houyi"
        assert "Successfully" in result.message

    @pytest.mark.asyncio
    async def test_execute_with_graph(self, patch_skill_rag_builder) -> None:
        """Test ingest with graph building."""
        mock_result = MagicMock()
        mock_result.documents_processed = 3
        mock_result.chunks_created = 15
        mock_result.index_path = "/kb/.houyi"

        mock_rag = MagicMock()
        mock_rag.index = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_ingest_module, mock_rag):
            input_data = KBIngestInput(
                paths=["/docs"],
                build_graph=True,
                chunk_size=256,
            )
            result = await execute_kb_ingest(input_data)

        mock_rag.index.assert_called_once()
        call_kwargs = mock_rag.index.call_args.kwargs
        assert call_kwargs["build_graph"] is True
        assert call_kwargs["chunk_size"] == 256
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_failure(self, patch_skill_rag_builder) -> None:
        """Test ingest failure handling."""
        mock_rag = MagicMock()
        mock_rag.index = AsyncMock(side_effect=Exception("Index build failed"))

        with patch_skill_rag_builder(kb_ingest_module, mock_rag):
            input_data = KBIngestInput(paths=["/docs"])
            result = await execute_kb_ingest(input_data)

        assert result.success is False
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_paths(self, patch_skill_rag_builder) -> None:
        """Test ingest with multiple paths."""
        mock_result = MagicMock()
        mock_result.documents_processed = 10
        mock_result.chunks_created = 50
        mock_result.index_path = "/kb/.houyi"

        mock_rag = MagicMock()
        mock_rag.index = AsyncMock(return_value=mock_result)

        with patch_skill_rag_builder(kb_ingest_module, mock_rag):
            input_data = KBIngestInput(
                paths=["/docs/a", "/docs/b", "/docs/c"],
            )
            result = await execute_kb_ingest(input_data)

        call_kwargs = mock_rag.index.call_args.kwargs
        assert call_kwargs["paths"] == ["/docs/a", "/docs/b", "/docs/c"]
        assert result.success is True


class TestKBIngestSkill:
    """Tests for skill definition."""

    def test_skill_definition(self) -> None:
        """Test skill is properly defined."""
        assert kb_ingest_skill.name == "kb-ingest"
        assert kb_ingest_skill.input_schema == KBIngestInput
        assert kb_ingest_skill.output_schema == KBIngestOutput
        assert kb_ingest_skill.executor == execute_kb_ingest
        assert kb_ingest_skill.version == "1.0.0"
        assert kb_ingest_skill.user_invocable is True
        assert "Read" in kb_ingest_skill.allowed_tools
        assert "Write" in kb_ingest_skill.allowed_tools
        assert "Glob" in kb_ingest_skill.allowed_tools
        assert len(kb_ingest_skill.hooks) == 3
