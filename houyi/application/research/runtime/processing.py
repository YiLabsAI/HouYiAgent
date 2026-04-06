from __future__ import annotations

import json
from typing import Any

from houyi.application.research.types import SearchResult, SearchRound, SourceReference


def process_sub_question_execution_order(questions: list[Any]) -> list[Any]:
    by_id = {q.question_id: q for q in questions}
    visited: set[str] = set()
    result: list[Any] = []

    def _visit(question_id: str) -> None:
        if question_id in visited:
            return
        visited.add(question_id)
        question = by_id.get(question_id)
        if question is None:
            return
        for dependency in question.depends_on:
            _visit(dependency)
        result.append(question)

    for question in sorted(questions, key=lambda item: item.priority, reverse=True):
        _visit(question.question_id)
    return result


def process_agent_search_output(sub_question: Any, output: Any) -> SearchResult:
    raw = str(output or "")
    sources: list[SourceReference] = []
    summary = ""
    queries_used: list[str] = []

    try:
        text = raw.strip()
        if text.startswith("```"):
            first_newline = text.index("\n")
            last_fence = text.rfind("```")
            text = text[first_newline + 1 : last_fence].strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        for source in data.get("sources", []):
            sources.append(
                SourceReference(
                    url=source.get("url", ""),
                    title=source.get("title", ""),
                    snippet=source.get("snippet", source.get("content_summary", "")),
                    source_type="web",
                    reliability_score=0.5,
                )
            )
        summary = data.get("summary", "")
        queries_used = data.get("queries_used", [])
    except (json.JSONDecodeError, ValueError, KeyError):
        summary = raw[:500] if raw else "No results"

    coverage = min(1.0, len(sources) / max(sub_question.expected_sources, 1))
    return SearchResult(
        question_id=sub_question.question_id,
        rounds=[
            SearchRound(
                round_index=0,
                queries=queries_used,
                hits=[],
                sufficient=bool(sources),
                rationale=summary[:200],
            )
        ],
        sources=sources,
        summary=summary,
        coverage_score=coverage,
        exhausted=not bool(sources),
    )
