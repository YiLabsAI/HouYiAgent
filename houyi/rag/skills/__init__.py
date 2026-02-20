"""Skills for RAG integration with Agents.

Provides SimpleSkill v0.1 compliant skills for knowledge base operations:
- kb_search_skill: Knowledge base search with dual-mode (agentic/indexed)
- kb_ingest_skill: Document ingest and index building
- kb_graph_skill: Knowledge graph query and traversal
- kb_analyze_skill: Knowledge base statistics and health checks
"""

from houyi.rag.skills.kb_analyze.skill import kb_analyze_skill
from houyi.rag.skills.kb_graph.skill import kb_graph_skill
from houyi.rag.skills.kb_ingest.skill import kb_ingest_skill
from houyi.rag.skills.kb_search.skill import kb_search_skill

__all__ = [
    "kb_analyze_skill",
    "kb_graph_skill",
    "kb_ingest_skill",
    "kb_search_skill",
]
