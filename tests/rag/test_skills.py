"""Tests for RAG skills."""


from houyi.rag.skills.kb_analyze.skill import (
    KBAnalyzeInput,
    KBAnalyzeOutput,
    kb_analyze_skill,
)
from houyi.rag.skills.kb_graph.skill import (
    KBGraphInput,
    KBGraphOutput,
    kb_graph_skill,
)
from houyi.rag.skills.kb_ingest.skill import (
    KBIngestInput,
    KBIngestOutput,
    kb_ingest_skill,
)
from houyi.rag.skills.kb_search.skill import (
    KBSearchInput,
    KBSearchOutput,
    Source,
    kb_search_skill,
)


class TestKBSearchInput:
    """Tests for KBSearchInput schema."""

    def test_default_values(self) -> None:
        """Test default input values."""
        input_data = KBSearchInput(query="test query")
        assert input_data.query == "test query"
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.mode == "auto"
        assert input_data.max_rounds == 5

    def test_custom_values(self) -> None:
        """Test custom input values."""
        input_data = KBSearchInput(
            query="custom query",
            knowledge_dir="/custom/path",
            mode="agentic",
            max_rounds=3,
        )
        assert input_data.query == "custom query"
        assert input_data.knowledge_dir == "/custom/path"
        assert input_data.mode == "agentic"
        assert input_data.max_rounds == 3


class TestKBSearchOutput:
    """Tests for KBSearchOutput schema."""

    def test_output_creation(self) -> None:
        """Test creating output."""
        output = KBSearchOutput(
            answer="Test answer",
            sources=[
                Source(file_path="/path/to/file.md", location="line 10"),
            ],
            confidence=0.85,
        )
        assert output.answer == "Test answer"
        assert len(output.sources) == 1
        assert output.confidence == 0.85

    def test_output_default_sources(self) -> None:
        """Test output with default empty sources."""
        output = KBSearchOutput(answer="Answer")
        assert output.sources == []
        assert output.confidence == 0.0


class TestKBSearchSkill:
    """Tests for kb_search_skill definition."""

    def test_skill_name(self) -> None:
        """Test skill name."""
        assert kb_search_skill.name == "kb-search"

    def test_skill_description(self) -> None:
        """Test skill has description."""
        assert kb_search_skill.description is not None
        assert len(kb_search_skill.description) > 0

    def test_skill_schemas(self) -> None:
        """Test skill has input/output schemas."""
        assert kb_search_skill.input_schema == KBSearchInput
        assert kb_search_skill.output_schema == KBSearchOutput

    def test_skill_executor(self) -> None:
        """Test skill has executor."""
        assert kb_search_skill.executor is not None

    def test_skill_metadata(self) -> None:
        """Test skill metadata and version."""
        assert kb_search_skill.metadata.get("category") == "rag"
        # Version is now a top-level field per SimpleSkill v0.1
        assert kb_search_skill.version == "1.0.0"

    def test_skill_hooks(self) -> None:
        """Test skill has hooks configured."""
        assert len(kb_search_skill.hooks) == 3
        # Check hook events
        hook_events = [h.event.value for h in kb_search_skill.hooks]
        assert "PreToolUse" in hook_events
        assert "PostToolUse" in hook_events
        assert "Stop" in hook_events

    def test_skill_allowed_tools(self) -> None:
        """Test skill has allowed tools."""
        assert "Read" in kb_search_skill.allowed_tools
        assert "Glob" in kb_search_skill.allowed_tools
        assert "Grep" in kb_search_skill.allowed_tools


class TestKBIngestSkill:
    """Tests for kb_ingest_skill definition."""

    def test_skill_name(self) -> None:
        """Test skill name."""
        assert kb_ingest_skill.name == "kb-ingest"

    def test_skill_version(self) -> None:
        """Test skill version."""
        assert kb_ingest_skill.version == "1.0.0"

    def test_skill_schemas(self) -> None:
        """Test skill has input/output schemas."""
        assert kb_ingest_skill.input_schema == KBIngestInput
        assert kb_ingest_skill.output_schema == KBIngestOutput

    def test_skill_hooks(self) -> None:
        """Test skill has hooks configured."""
        assert len(kb_ingest_skill.hooks) == 3

    def test_input_defaults(self) -> None:
        """Test input default values."""
        input_data = KBIngestInput(paths=["/test/path"])
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.mode == "incremental"
        assert input_data.build_graph is False
        assert input_data.chunk_size == 512


class TestKBGraphSkill:
    """Tests for kb_graph_skill definition."""

    def test_skill_name(self) -> None:
        """Test skill name."""
        assert kb_graph_skill.name == "kb-graph"

    def test_skill_version(self) -> None:
        """Test skill version."""
        assert kb_graph_skill.version == "1.0.0"

    def test_skill_schemas(self) -> None:
        """Test skill has input/output schemas."""
        assert kb_graph_skill.input_schema == KBGraphInput
        assert kb_graph_skill.output_schema == KBGraphOutput

    def test_skill_hooks(self) -> None:
        """Test skill has hooks configured."""
        assert len(kb_graph_skill.hooks) == 3

    def test_input_defaults(self) -> None:
        """Test input default values."""
        input_data = KBGraphInput(query="test query")
        assert input_data.max_hops == 3
        assert input_data.top_k == 10
        assert input_data.use_ppr is True


class TestKBAnalyzeSkill:
    """Tests for kb_analyze_skill definition."""

    def test_skill_name(self) -> None:
        """Test skill name."""
        assert kb_analyze_skill.name == "kb-analyze"

    def test_skill_version(self) -> None:
        """Test skill version."""
        assert kb_analyze_skill.version == "1.0.0"

    def test_skill_schemas(self) -> None:
        """Test skill has input/output schemas."""
        assert kb_analyze_skill.input_schema == KBAnalyzeInput
        assert kb_analyze_skill.output_schema == KBAnalyzeOutput

    def test_skill_hooks(self) -> None:
        """Test skill has hooks configured."""
        assert len(kb_analyze_skill.hooks) == 3

    def test_input_defaults(self) -> None:
        """Test input default values."""
        input_data = KBAnalyzeInput()
        assert input_data.knowledge_dir == "knowledge/"
        assert input_data.analysis_type == "full"
        assert input_data.include_content is False


class TestSkillsExport:
    """Tests for skills module export."""

    def test_all_skills_exported(self) -> None:
        """Test all skills are exported from skills module."""
        from houyi.rag.skills import (
            kb_analyze_skill,
            kb_graph_skill,
            kb_ingest_skill,
            kb_search_skill,
        )

        assert kb_search_skill.name == "kb-search"
        assert kb_ingest_skill.name == "kb-ingest"
        assert kb_graph_skill.name == "kb-graph"
        assert kb_analyze_skill.name == "kb-analyze"
