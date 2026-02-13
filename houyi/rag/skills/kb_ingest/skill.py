"""Knowledge base ingest skill definition."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from houyi.core.skill import ExecutionMode, SkillSpec
from houyi.core.skill.hooks import HookEvent, HookType, SkillHook
from houyi.rag.config import _default_knowledge_dir


class KBIngestInput(BaseModel):
    """Input schema for knowledge base ingest."""

    paths: list[str] = Field(..., description="Files or directories to ingest")
    knowledge_dir: str = Field(
        default_factory=_default_knowledge_dir,
        description="Knowledge base root directory (reads RAG_KNOWLEDGE_DIR env var)",
    )
    mode: str = Field(
        default="incremental",
        description="Ingest mode: 'incremental' or 'full'",
    )
    build_graph: bool = Field(
        default=False,
        description="Whether to build knowledge graph",
    )
    chunk_size: int = Field(
        default=512,
        description="Document chunk size",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Chunk overlap size",
    )


class KBIngestOutput(BaseModel):
    """Output schema for knowledge base ingest."""

    success: bool = Field(..., description="Whether ingest succeeded")
    documents_processed: int = Field(
        default=0,
        description="Number of documents processed",
    )
    chunks_created: int = Field(
        default=0,
        description="Number of chunks created",
    )
    index_path: str = Field(
        default="",
        description="Path to index storage",
    )
    message: str = Field(
        default="",
        description="Status message",
    )


async def execute_kb_ingest(input_data: KBIngestInput) -> KBIngestOutput:
    """Execute knowledge base ingest.

    Args:
        input_data: Ingest input parameters

    Returns:
        Ingest output with statistics
    """
    from houyi.rag import RAG

    try:
        # Create RAG
        rag = RAG(
            mode="indexed",
            knowledge_dir=input_data.knowledge_dir,
        )

        # Execute index
        result = await rag.index(
            paths=input_data.paths,
            build_graph=input_data.build_graph,
            chunk_size=input_data.chunk_size,
            chunk_overlap=input_data.chunk_overlap,
        )

        return KBIngestOutput(
            success=True,
            documents_processed=result.documents_processed,
            chunks_created=result.chunks_created,
            index_path=result.index_path,
            message=f"Successfully ingested {result.documents_processed} documents",
        )
    except Exception as e:
        return KBIngestOutput(
            success=False,
            message=f"Ingest failed: {e}",
        )


# Define hooks
_kb_ingest_hooks = [
    SkillHook(
        event=HookEvent.PRE_TOOL_USE,
        matcher="Write",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_ingest.hooks:pre_ingest_hook",
    ),
    SkillHook(
        event=HookEvent.POST_TOOL_USE,
        matcher=".*",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_ingest.hooks:post_ingest_hook",
    ),
    SkillHook(
        event=HookEvent.STOP,
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_ingest.hooks:stop_hook",
    ),
]

# Skill definition
kb_ingest_skill = SkillSpec(
    name="kb-ingest",
    description="""Knowledge base document ingest tool.
Loads documents, chunks them, generates embeddings, and builds search indexes.
Supports Markdown, TXT, PDF, and HTML formats.
Use when user wants to "add documents to knowledge base" or "build index".""",
    input_schema=KBIngestInput,
    output_schema=KBIngestOutput,
    executor=execute_kb_ingest,
    skill_md_path=str(Path(__file__).parent / "SKILL.md"),
    execution_mode=ExecutionMode.PLUGIN,
    version="1.0.0",
    user_invocable=True,
    allowed_tools=["Read", "Write", "Glob"],
    hooks=_kb_ingest_hooks,
    metadata={
        "category": "rag",
    },
)
