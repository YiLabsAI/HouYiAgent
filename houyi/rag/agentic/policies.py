"""Heuristic policy helpers for agentic round orchestration.

This module owns the lightweight, local rules used by AgenticMode when no
LLM call is needed: early termination, keyword fallback/refinement, entity
harvesting from retrieved snippets, result deduplication, and simple answer
assembly.

These helpers are intentionally shallow policies rather than reusable generic
RAG primitives. They encode the current agentic search strategy and can evolve
with the round orchestration without coupling AgenticMode to large in-method
 heuristic blocks.
"""

from __future__ import annotations

import re
from enum import Enum

from houyi.rag.types import SearchResult

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
}


def should_terminate(results: list[SearchResult], current_round: str | Enum) -> bool:
    """Decide whether later rounds can be skipped based on accumulated results."""
    round_value = current_round.value if isinstance(current_round, Enum) else current_round
    if round_value == "broad":
        return False
    high_score = [result for result in results if result.score > 0.7]
    if len(high_score) >= 5:
        return True
    return len(results) >= 10


def get_top_files(results: list[SearchResult], limit: int = 5) -> list[str]:
    """Select the highest-scoring source files seen so far for focused rounds."""
    file_scores: dict[str, float] = {}
    for result in results:
        if result.source and result.source.file_path:
            path = result.source.file_path
            file_scores[path] = max(file_scores.get(path, 0), result.score)
    sorted_files = sorted(file_scores.items(), key=lambda item: item[1], reverse=True)
    return [path for path, _ in sorted_files[:limit]]


def extract_entities(results: list[SearchResult]) -> list[str]:
    """Harvest simple entity candidates from retrieved snippets for cross-reference rounds."""
    entities: set[str] = set()
    for result in results:
        if not result.content:
            continue
        for word in result.content.split():
            clean = re.sub(r"[^\w]", "", word)
            if clean and clean[0].isupper() and len(clean) > 2:
                entities.add(clean)
        entities.update(re.findall(r'"([^"]+)"', result.content))
    return list(entities)[:10]


def refine_keywords(query: str, results: list[SearchResult]) -> list[str]:
    """Keep overlap terms between the query and top results for verification rounds."""
    query_words = set(query.lower().split())
    result_words: set[str] = set()
    for result in results:
        if result.content:
            result_words.update(result.content.lower().split())
    common = query_words & result_words
    refined = [word for word in common if len(word) > 2]
    return refined[:5] if refined else []


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-scoring representative for near-duplicate result content."""
    seen_content: set[str] = set()
    unique: list[SearchResult] = []
    for result in sorted(results, key=lambda item: item.score, reverse=True):
        content_key = result.content[:100] if result.content else result.chunk_id or ""
        if content_key and content_key not in seen_content:
            seen_content.add(content_key)
            unique.append(result)
    return unique


def extract_keywords_simple(query: str) -> list[str]:
    """Fallback keyword extraction used when no LLM keyword collaborator is available."""
    words = query.replace("?", "").split()
    return [word for word in words if word.lower() not in _STOP_WORDS and len(word) > 1]


def build_answer_simple(results: list[SearchResult]) -> str:
    """Fallback answer assembly for local-only agentic retrieval responses."""
    contents = [result.content.strip() for result in results[:5] if result.content]
    if not contents:
        return "No relevant information found."
    return "\n\n---\n\n".join(contents)
