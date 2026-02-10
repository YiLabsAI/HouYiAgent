"""Tests for kb-analyze skill."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from houyi.rag.skills.kb_analyze.skill import (
    Health,
    Issue,
    KBAnalyzeInput,
    KBAnalyzeOutput,
    Stats,
    execute_kb_analyze,
    kb_analyze_skill,
)


class TestModels:
    """Tests for data models."""

    def test_stats_defaults(self) -> None:
        """Test Stats default values."""
        stats = Stats()
        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_entities == 0
        assert stats.index_size_mb == 0.0
        assert stats.file_types == {}

    def test_health_defaults(self) -> None:
        """Test Health default values."""
        health = Health()
        assert health.status == "unknown"
        assert health.index_integrity is True
        assert health.coverage_percent == 0.0

    def test_issue_creation(self) -> None:
        """Test Issue creation."""
        issue = Issue(severity="high", message="Test issue")
        assert issue.severity == "high"
        assert issue.message == "Test issue"

    def test_kb_analyze_input_defaults(self) -> None:
        """Test KBAnalyzeInput defaults."""
        input_data = KBAnalyzeInput()
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.analysis_type == "full"
        assert input_data.include_content is False

    def test_kb_analyze_output_defaults(self) -> None:
        """Test KBAnalyzeOutput defaults."""
        output = KBAnalyzeOutput()
        assert output.stats.total_documents == 0
        assert output.health.status == "unknown"
        assert output.recommendations == []
        assert output.issues == []


class TestExecuteKBAnalyze:
    """Tests for execute_kb_analyze function."""

    @pytest.mark.asyncio
    async def test_analyze_existing_dir_with_files(self) -> None:
        """Test analyzing existing directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create test files
            (kb_dir / "doc1.md").write_text("# Doc 1")
            (kb_dir / "doc2.md").write_text("# Doc 2")
            (kb_dir / "data.json").write_text("{}")

            input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
            result = await execute_kb_analyze(input_data)

            assert result.stats.total_documents == 3
            assert ".md" in result.stats.file_types
            assert result.stats.file_types[".md"] == 2
            assert result.stats.file_types[".json"] == 1
            assert result.health.status == "degraded"  # No index
            assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_analyze_with_index(self) -> None:
        """Test analyzing directory with existing index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            (kb_dir / "doc.md").write_text("content")

            # Create index directory
            index_dir = Path(tmpdir) / ".rag_index"
            index_dir.mkdir()
            (index_dir / "vectors.bin").write_bytes(b"x" * 1024)

            input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
            result = await execute_kb_analyze(input_data)

            assert result.health.status == "healthy"
            assert result.health.coverage_percent == 100.0
            assert result.stats.index_size_mb > 0

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_dir(self) -> None:
        """Test analyzing non-existent directory."""
        input_data = KBAnalyzeInput(knowledge_dir="/nonexistent/path")
        result = await execute_kb_analyze(input_data)

        assert result.health.status == "unhealthy"
        assert len(result.issues) > 0
        assert any(i.severity == "high" for i in result.issues)

    @pytest.mark.asyncio
    async def test_analyze_empty_dir(self) -> None:
        """Test analyzing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
            result = await execute_kb_analyze(input_data)

            assert result.stats.total_documents == 0
            assert result.stats.file_types == {}

    @pytest.mark.asyncio
    async def test_analyze_many_documents_recommendation(self) -> None:
        """Test recommendation for many documents without index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create more than 100 files
            for i in range(105):
                (kb_dir / f"doc{i}.md").write_text(f"content {i}")

            input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
            result = await execute_kb_analyze(input_data)

            assert result.stats.total_documents == 105
            assert any("indexed mode" in r.lower() for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_analyze_many_pdfs_recommendation(self) -> None:
        """Test recommendation for many PDF files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()

            # Create more than 10 PDF files
            for i in range(12):
                (kb_dir / f"doc{i}.pdf").write_bytes(b"pdf content")

            input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
            result = await execute_kb_analyze(input_data)

            assert result.stats.file_types[".pdf"] == 12
            assert any("pypdf" in r.lower() for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_analyze_with_subdirs(self) -> None:
        """Test analyzing directory with subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledge"
            kb_dir.mkdir()
            subdir = kb_dir / "subdir"
            subdir.mkdir()

            (kb_dir / "doc1.md").write_text("content")
            (subdir / "doc2.md").write_text("content")

            input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
            result = await execute_kb_analyze(input_data)

            # Should count files in subdirs
            assert result.stats.total_documents == 2


class TestKBAnalyzeSkill:
    """Tests for skill definition."""

    def test_skill_definition(self) -> None:
        """Test skill is properly defined."""
        assert kb_analyze_skill.name == "kb-analyze"
        assert kb_analyze_skill.input_schema == KBAnalyzeInput
        assert kb_analyze_skill.output_schema == KBAnalyzeOutput
        assert kb_analyze_skill.executor == execute_kb_analyze
        assert kb_analyze_skill.version == "1.0.0"
        assert kb_analyze_skill.user_invocable is True
        assert "Read" in kb_analyze_skill.allowed_tools
        assert "Glob" in kb_analyze_skill.allowed_tools
        assert len(kb_analyze_skill.hooks) == 3
