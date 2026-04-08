from __future__ import annotations

import json
import re

_MAX_QUERY_LENGTH = 380


def _parse_query_list(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _normalize_queries([str(q) for q in parsed[:5]])
    except json.JSONDecodeError:
        pass

    numbered = _extract_numbered_queries(text)
    if numbered:
        return _normalize_queries(numbered[:5])

    return _normalize_queries([text]) if text else []


def _extract_numbered_queries(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    queries: list[str] = []
    for line in lines:
        line = line.lstrip("-• ")
        match = re.match(
            r"(?:\*\*)?query\s*\d+\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.+)",
            line,
            re.I,
        )
        if match:
            candidate = match.group(1).strip()
            if candidate:
                queries.append(candidate)
    return queries


def _normalize_queries(queries: list[str]) -> list[str]:
    normalized: list[str] = []
    for query in queries:
        candidate = " ".join(query.split()).strip()
        if not candidate:
            continue
        candidate = candidate[:_MAX_QUERY_LENGTH].rstrip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _canonical_query(query: str) -> str:
    normalized = _normalize_queries([query])
    if not normalized:
        return ""
    return normalized[0].lower()


def _parse_sufficiency(content: str) -> tuple[bool, str]:
    text = content.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_fence = text.rfind("```")
        text = text[first_nl + 1 : last_fence].strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return False, "Failed to parse sufficiency evaluation"
        return bool(data.get("sufficient", False)), str(data.get("rationale", ""))
    except json.JSONDecodeError:
        return False, "Failed to parse sufficiency evaluation"
