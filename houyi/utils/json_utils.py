from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def parse_embedded_json(content: str) -> Any:
    normalized = _strip_markdown_fence(content.strip())
    try:
        return json.loads(normalized)
    except JSONDecodeError:
        pass

    extracted = _extract_top_level_json_block(normalized)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except JSONDecodeError:
            pass

    # Try each fenced code block independently (LLM may self-correct
    # and produce a second valid block after a first malformed one).
    for block in _extract_fenced_blocks(content.strip()):
        try:
            return json.loads(block)
        except JSONDecodeError:
            inner = _extract_top_level_json_block(block)
            if inner is not None:
                try:
                    return json.loads(inner)
                except JSONDecodeError:
                    pass

    # Last resort: truncation repair — find the longest balanced prefix.
    repaired = _truncation_repair(normalized)
    if repaired is not None:
        return json.loads(repaired)

    raise JSONDecodeError("No valid JSON found", content, 0)


def _strip_markdown_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    body = [line for line in lines if not line.startswith("```")]
    return "\n".join(body).strip()


def _extract_top_level_json_block(content: str) -> str | None:
    start = _find_first_json_start(content)
    if start < 0:
        return None
    opener = content[start]
    closer = "}" if opener == "{" else "]"
    return _extract_balanced_block(content, start, opener, closer)


def _find_first_json_start(content: str) -> int:
    object_start = content.find("{")
    array_start = content.find("[")
    if object_start < 0:
        return array_start
    if array_start < 0:
        return object_start
    return min(object_start, array_start)


def _extract_fenced_blocks(content: str) -> list[str]:
    """Extract the body of each markdown code fence as a separate string."""
    blocks: list[str] = []
    lines = content.splitlines()
    inside = False
    buf: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if inside:
                blocks.append("\n".join(buf).strip())
                buf = []
                inside = False
            else:
                inside = True
                buf = []
        elif inside:
            buf.append(line)
    return blocks


def _truncation_repair(content: str) -> str | None:
    """Find the longest balanced JSON object by scanning closing braces from the end."""
    start = _find_first_json_start(content)
    if start < 0 or content[start] != "{":
        return None
    # Collect positions of every top-level-candidate closing brace from the end.
    last = content.rfind("}")
    while last > start:
        candidate = content[start : last + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return candidate
        except JSONDecodeError:
            pass
        last = content.rfind("}", start, last)
    return None


def _extract_balanced_block(content: str, start: int, opener: str, closer: str) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
            continue
        if char != closer:
            continue
        depth -= 1
        if depth == 0:
            return content[start : index + 1]

    return None
