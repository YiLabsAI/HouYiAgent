"""Knowledge graph query skill definition."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook
from houyi.domain.skill.policy import ExecutionMode
from houyi.domain.skill.spec import SkillSpec
from houyi.rag.config import _default_knowledge_dir
from houyi.rag.skills._rag_runtime import build_skill_rag


class Entity(BaseModel):
    """A knowledge graph entity."""

    id: str = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    type: str = Field(default="", description="Entity type")
    score: float = Field(default=0.0, description="Relevance score")


class Relation(BaseModel):
    """A knowledge graph relation."""

    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relation type")


class KBGraphInput(BaseModel):
    """Input schema for knowledge graph query."""

    query: str = Field(..., description="Query for graph traversal")
    knowledge_dir: str = Field(
        default_factory=_default_knowledge_dir,
        description="Knowledge base root directory (reads RAG_KNOWLEDGE_DIR env var)",
    )
    start_entities: list[str] = Field(
        default_factory=list,
        description="Starting entity IDs",
    )
    max_hops: int = Field(
        default=3,
        description="Maximum traversal hops",
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return",
    )
    use_ppr: bool = Field(
        default=True,
        description="Whether to use PPR for ranking",
    )


class KBGraphOutput(BaseModel):
    """Output schema for knowledge graph query."""

    entities: list[Entity] = Field(
        default_factory=list,
        description="Found entities",
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description="Found relations",
    )
    paths: list[list[str]] = Field(
        default_factory=list,
        description="Paths between entities",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score (0-1)",
    )


async def execute_kb_graph(input_data: KBGraphInput) -> KBGraphOutput:
    """Execute knowledge graph query.

    This skill uses the same runtime RAG builder contract as the other runtime
    RAG skills so knowledge storage selection stays consistent across search,
    ingest, and graph-oriented executors.

    Args:
        input_data: Graph query input parameters

    Returns:
        Graph query output with entities and relations
    """
    try:
        rag = build_skill_rag(
            mode="indexed",
            knowledge_dir=input_data.knowledge_dir,
        )

        # Execute graph query
        result = await rag.graph_query(
            query=input_data.query,
            start_entities=input_data.start_entities,
            max_hops=input_data.max_hops,
            top_k=input_data.top_k,
            use_ppr=input_data.use_ppr,
        )

        # Convert to output schema
        entities = [
            Entity(
                id=e.id,
                name=e.name,
                type=e.type,
                score=e.score,
            )
            for e in result.entities
        ]
        relations = [
            Relation(
                source=r.source,
                target=r.target,
                type=r.type,
            )
            for r in result.relations
        ]

        return KBGraphOutput(
            entities=entities,
            relations=relations,
            paths=result.paths,
            confidence=result.confidence,
        )
    except Exception:
        return KBGraphOutput(
            confidence=0.0,
        )


# Define hooks
_kb_graph_hooks = [
    SkillHook(
        event=HookEvent.PRE_TOOL_USE,
        matcher="Read|Grep",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_graph.hooks:pre_graph_hook",
    ),
    SkillHook(
        event=HookEvent.POST_TOOL_USE,
        matcher=".*",
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_graph.hooks:post_graph_hook",
    ),
    SkillHook(
        event=HookEvent.STOP,
        hook_type=HookType.HANDLER,
        handler_path="houyi.rag.skills.kb_graph.hooks:stop_hook",
    ),
]

# Skill definition
kb_graph_skill = SkillSpec(
    name="kb-graph",
    description="""Knowledge graph query tool.
Supports entity retrieval, relation traversal, and multi-hop reasoning.
Uses PPR (Personalized PageRank) for relevance ranking.
Use when user asks about "related entities" or "relationships between concepts".""",
    input_schema=KBGraphInput,
    output_schema=KBGraphOutput,
    executor=execute_kb_graph,
    skill_md_path=str(Path(__file__).parent / "SKILL.md"),
    execution_mode=ExecutionMode.PLUGIN,
    version="1.0.0",
    user_invocable=True,
    allowed_tools=["Read", "Grep"],
    hooks=_kb_graph_hooks,
    metadata={
        "category": "rag",
    },
)
