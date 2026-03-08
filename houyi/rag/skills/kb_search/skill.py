"""Knowledge base search skill definition."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook
from houyi.domain.skill.policy import ExecutionMode
from houyi.domain.skill.spec import SkillSpec
from houyi.rag.config import _default_knowledge_dir
from houyi.rag.skills._rag_runtime import build_skill_rag


class KBSearchInput(BaseModel):
    """Input schema for knowledge base search."""

    query: str = Field(..., description="User query to search in knowledge base")
    knowledge_dir: str = Field(
        default_factory=_default_knowledge_dir,
        description="Root directory of the knowledge base (reads RAG_KNOWLEDGE_DIR env var)",
    )
    mode: str = Field(
        default="auto",
        description="Search mode: 'auto' (default), 'agentic', or 'indexed'",
    )
    max_rounds: int = Field(
        default=5,
        description="Maximum search iterations for agentic mode",
    )
    llm_provider: str = Field(
        default="",
        description="LLM provider for enhanced search (openai, anthropic)",
    )
    llm_model: str = Field(
        default="",
        description="LLM model name",
    )


class Source(BaseModel):
    """A citation source reference."""

    file_path: str = Field(..., description="Source file path")
    location: str = Field(default="", description="Location within file")
    snippet: str = Field(default="", description="Relevant text snippet")


class KBSearchOutput(BaseModel):
    """Output schema for knowledge base search."""

    answer: str = Field(..., description="Generated answer based on retrieved content")
    sources: list[Source] = Field(
        default_factory=list,
        description="List of citation sources used in the answer",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score for the answer (0-1)",
    )


async def execute_kb_search(input_data: KBSearchInput) -> KBSearchOutput:
    """Execute knowledge base search.

    This is the skill executor function that integrates with Houyi Agent.

    Args:
        input_data: Search input parameters

    Returns:
        Search output with answer and sources
    """
    rag = build_skill_rag(
        mode=input_data.mode,
        knowledge_dir=input_data.knowledge_dir,
        llm_provider=input_data.llm_provider or None,
        llm_model=input_data.llm_model or None,
    )

    # Execute search
    result = await rag.query(
        query=input_data.query,
        max_rounds=input_data.max_rounds,
    )

    # Convert to output schema
    sources = [
        Source(
            file_path=s.file_path,
            location=s.location,
            snippet=s.snippet,
        )
        for s in result.sources
    ]

    return KBSearchOutput(
        answer=result.answer,
        sources=sources,
        confidence=result.confidence,
    )


# Define hooks for SimpleSkill v0.1
_kb_search_hooks = [
    SkillHook(
        event=HookEvent.PRE_TOOL_USE,
        matcher="Read|Glob|Grep",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_search.hooks:pre_search_hook",
    ),
    SkillHook(
        event=HookEvent.POST_TOOL_USE,
        matcher=".*",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_search.hooks:post_search_hook",
    ),
    SkillHook(
        event=HookEvent.STOP,
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_search.hooks:stop_hook",
    ),
]

# Skill definition
kb_search_skill = SkillSpec(
    name="kb-search",
    description="""Knowledge base intelligent search assistant.
Automatically selects search mode based on query complexity:
- Small/simple queries: Agentic mode (grep + read_file)
- Large/complex queries: Indexed mode (Vector + Graph + BM25)
Supports progressive search with up to 5 iterations to find the most relevant information.
Use when the user asks to "search knowledge base", "find in documents", or "look up information".""",
    input_schema=KBSearchInput,
    output_schema=KBSearchOutput,
    executor=execute_kb_search,
    skill_md_path=str(Path(__file__).parent / "SKILL.md"),
    execution_mode=ExecutionMode.PLUGIN,
    version="1.0.0",
    user_invocable=True,
    allowed_tools=["Read", "Glob", "Grep"],
    hooks=_kb_search_hooks,
    metadata={
        "category": "rag",
    },
)
