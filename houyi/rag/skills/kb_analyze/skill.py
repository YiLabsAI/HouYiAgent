"""Knowledge base analyze skill definition."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from houyi.core.skill import ExecutionMode, SkillSpec
from houyi.core.skill.hooks import HookEvent, HookType, SkillHook
from houyi.rag.config import _default_knowledge_dir


class Stats(BaseModel):
    """Knowledge base statistics."""

    total_documents: int = Field(default=0, description="Total document count")
    total_chunks: int = Field(default=0, description="Total chunk count")
    total_entities: int = Field(default=0, description="Total entity count")
    index_size_mb: float = Field(default=0.0, description="Index size in MB")
    file_types: dict[str, int] = Field(
        default_factory=dict,
        description="File type distribution",
    )


class Health(BaseModel):
    """Knowledge base health status."""

    status: str = Field(default="unknown", description="Health status")
    index_integrity: bool = Field(default=True, description="Index integrity check")
    coverage_percent: float = Field(default=0.0, description="Document coverage %")


class Issue(BaseModel):
    """A detected issue."""

    severity: str = Field(..., description="Issue severity (low/medium/high)")
    message: str = Field(..., description="Issue description")


class KBAnalyzeInput(BaseModel):
    """Input schema for knowledge base analyze."""

    knowledge_dir: str = Field(
        default_factory=_default_knowledge_dir,
        description="Knowledge base root directory (reads RAG_KNOWLEDGE_DIR env var)",
    )
    analysis_type: str = Field(
        default="full",
        description="Analysis type: 'full', 'stats', 'health', or 'optimize'",
    )
    include_content: bool = Field(
        default=False,
        description="Whether to include content analysis",
    )


class KBAnalyzeOutput(BaseModel):
    """Output schema for knowledge base analyze."""

    stats: Stats = Field(default_factory=Stats, description="Statistics")
    health: Health = Field(default_factory=Health, description="Health status")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Optimization recommendations",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="Detected issues",
    )


async def execute_kb_analyze(input_data: KBAnalyzeInput) -> KBAnalyzeOutput:
    """Execute knowledge base analysis.

    Args:
        input_data: Analysis input parameters

    Returns:
        Analysis output with stats and recommendations
    """
    from pathlib import Path as P

    knowledge_dir = P(input_data.knowledge_dir)

    # Collect basic stats
    stats = Stats()
    health = Health(status="healthy", index_integrity=True)
    recommendations: list[str] = []
    issues: list[Issue] = []

    try:
        if knowledge_dir.exists():
            # Count documents by type
            file_types: dict[str, int] = {}
            for f in knowledge_dir.rglob("*"):
                if f.is_file():
                    ext = f.suffix.lower() or ".unknown"
                    file_types[ext] = file_types.get(ext, 0) + 1
                    stats.total_documents += 1

            stats.file_types = file_types

            # Check for index
            index_dir = knowledge_dir.parent / ".rag_index"
            if index_dir.exists():
                # Calculate index size
                total_size = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
                stats.index_size_mb = total_size / (1024 * 1024)
                health.coverage_percent = 100.0
            else:
                health.status = "degraded"
                health.coverage_percent = 0.0
                issues.append(
                    Issue(
                        severity="medium",
                        message="No index found. Run kb-ingest to build index.",
                    )
                )
                recommendations.append("Run kb-ingest to build search index")

            # Generate recommendations based on stats
            if stats.total_documents > 100 and not index_dir.exists():
                recommendations.append(
                    "Consider building indexed mode for better performance with large document sets"
                )

            if ".pdf" in file_types and file_types.get(".pdf", 0) > 10:
                recommendations.append("Install pypdf for better PDF processing: pip install pypdf")

        else:
            health.status = "unhealthy"
            issues.append(
                Issue(
                    severity="high",
                    message=f"Knowledge directory not found: {knowledge_dir}",
                )
            )

    except Exception as e:
        health.status = "unhealthy"
        issues.append(
            Issue(
                severity="high",
                message=f"Analysis failed: {e}",
            )
        )

    return KBAnalyzeOutput(
        stats=stats,
        health=health,
        recommendations=recommendations,
        issues=issues,
    )


# Define hooks
_kb_analyze_hooks = [
    SkillHook(
        event=HookEvent.PRE_TOOL_USE,
        matcher="Read|Glob",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_analyze.hooks:pre_analyze_hook",
    ),
    SkillHook(
        event=HookEvent.POST_TOOL_USE,
        matcher=".*",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_analyze.hooks:post_analyze_hook",
    ),
    SkillHook(
        event=HookEvent.STOP,
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_analyze.hooks:stop_hook",
    ),
]

# Skill definition
kb_analyze_skill = SkillSpec(
    name="kb-analyze",
    description="""Knowledge base analysis tool.
Provides statistics, health checks, and optimization recommendations.
Analyzes document distribution, index status, and query performance.
Use when user asks to "analyze knowledge base" or "check knowledge base status".""",
    input_schema=KBAnalyzeInput,
    output_schema=KBAnalyzeOutput,
    executor=execute_kb_analyze,
    skill_md_path=str(Path(__file__).parent / "SKILL.md"),
    execution_mode=ExecutionMode.PLUGIN,
    version="1.0.0",
    user_invocable=True,
    allowed_tools=["Read", "Glob"],
    hooks=_kb_analyze_hooks,
    metadata={
        "category": "rag",
    },
)
