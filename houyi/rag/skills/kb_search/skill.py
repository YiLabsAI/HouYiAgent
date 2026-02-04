"""Knowledge base search skill definition."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from houyi.core.skill import ExecutionMode, SkillSpec


class KBSearchInput(BaseModel):
    """Input schema for knowledge base search."""

    query: str = Field(..., description="User query to search in knowledge base")
    knowledge_dir: str = Field(
        default="knowledge/",
        description="Root directory of the knowledge base",
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
    from houyi.rag import RAGService

    # Create RAG service
    service = RAGService(
        mode=input_data.mode,
        knowledge_dir=input_data.knowledge_dir,
        llm_provider=input_data.llm_provider or None,
        llm_model=input_data.llm_model or None,
    )

    # Execute search
    result = await service.query(
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
    metadata={
        "category": "rag",
        "version": "1.0.0",
    },
)
