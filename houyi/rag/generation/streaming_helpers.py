from __future__ import annotations

from houyi.rag.types import SearchResult

RAG_ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

Guidelines:
- Answer ONLY based on the provided context
- Cite sources using [1], [2], etc. markers
- If the context doesn't contain enough information, say so clearly
- Be concise but complete
- If sources conflict, mention the discrepancy
- Use the same language as the user's question"""


def build_answer_prompt(query: str, results: list[SearchResult]) -> str:
    context = format_stream_context(results)
    return f"""Context:
{context}

Question: {query}

Please answer the question based on the context above."""


def build_stream_sources(results: list[SearchResult]) -> list[dict[str, str | int]]:
    sources: list[dict[str, str | int]] = []
    for index, result in enumerate(results[:10], 1):
        sources.append(
            {
                "index": index,
                "file_path": result.source.file_path if result.source else "",
                "snippet": result.content[:200] if result.content else "",
            }
        )
    return sources


def format_stream_context(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results[:10], 1):
        source_info = ""
        if result.source:
            source_info = f" (from: {result.source.file_path})"
        block = f"[{index}]{source_info}:\n{result.content.strip()}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def estimate_stream_confidence(
    answer: str,
    results: list[SearchResult],
    uncertainty_phrases: list[str] | None = None,
) -> float:
    confidence = 0.5
    citation_count = sum(1 for index in range(1, 11) if f"[{index}]" in answer)
    if citation_count > 0:
        confidence += 0.1 * min(citation_count, 3)
    high_score_count = sum(1 for result in results if result.score > 0.7)
    if high_score_count > 0:
        confidence += 0.1 * min(high_score_count, 3)
    phrases = uncertainty_phrases or [
        "not enough information",
        "cannot find",
        "no relevant",
    ]
    if any(phrase in answer.lower() for phrase in phrases):
        confidence -= 0.2
    return max(0.0, min(1.0, confidence))
