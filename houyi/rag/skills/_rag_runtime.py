from __future__ import annotations

from houyi import rag as rag_module


def build_skill_rag(
    *,
    mode: str,
    knowledge_dir: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
):
    return rag_module.RAG(
        mode=mode,
        knowledge_dir=knowledge_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
