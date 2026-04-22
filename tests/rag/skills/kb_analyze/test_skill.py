"""Tests for kb-analyze skill."""

from __future__ import annotations

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
    async def test_existing_dir_with_files(self, write_knowledge_files) -> None:
        """Test analyzing existing directory with files."""
        kb_dir = write_knowledge_files(
            {
                "doc1.md": "# Doc 1",
                "doc2.md": "# Doc 2",
                "data.json": "{}",
            }
        )

        input_data = KBAnalyzeInput(knowledge_dir=str(kb_dir))
        result = await execute_kb_analyze(input_data)

        assert result.stats.total_documents == 3
        assert ".md" in result.stats.file_types
        assert result.stats.file_types[".md"] == 2
        assert result.stats.file_types[".json"] == 1
        assert result.health.status == "degraded"
        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_analyze_with_index(self, write_knowledge_files) -> None:
        """Test analyzing directory with existing index."""
        kb_dir = write_knowledge_files({"doc.md": "content"})
        index_dir = kb_dir / ".houyi"
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
    async def test_analyze_empty_dir(self, knowledge_dir: Path) -> None:
        """Test analyzing empty directory."""
        input_data = KBAnalyzeInput(knowledge_dir=str(knowledge_dir))
        result = await execute_kb_analyze(input_data)

        assert result.stats.total_documents == 0
        assert result.stats.file_types == {}

    @pytest.mark.asyncio
    async def test_analyze_many_documents_recommendation(self, knowledge_dir: Path) -> None:
        """Test recommendation for many documents without index."""
        for i in range(105):
            (knowledge_dir / f"doc{i}.md").write_text(f"content {i}")

        input_data = KBAnalyzeInput(knowledge_dir=str(knowledge_dir))
        result = await execute_kb_analyze(input_data)

        assert result.stats.total_documents == 105
        assert any("indexed mode" in r.lower() for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_analyze_many_pdfs_recommendation(self, knowledge_dir: Path) -> None:
        """Test recommendation for many PDF files."""
        for i in range(12):
            (knowledge_dir / f"doc{i}.pdf").write_bytes(b"pdf content")

        input_data = KBAnalyzeInput(knowledge_dir=str(knowledge_dir))
        result = await execute_kb_analyze(input_data)

        assert result.stats.file_types[".pdf"] == 12
        assert any("pypdf" in r.lower() for r in result.recommendations)

    @pytest.mark.asyncio
    async def test_analyze_with_subdirs(self, knowledge_dir: Path) -> None:
        """Test analyzing directory with subdirectories."""
        subdir = knowledge_dir / "subdir"
        subdir.mkdir()

        (knowledge_dir / "doc1.md").write_text("content")
        (subdir / "doc2.md").write_text("content")

        input_data = KBAnalyzeInput(knowledge_dir=str(knowledge_dir))
        result = await execute_kb_analyze(input_data)

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
