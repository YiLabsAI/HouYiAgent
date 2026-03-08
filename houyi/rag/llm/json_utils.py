from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def parse_embedded_json(content: str) -> Any:
    normalized = _strip_markdown_fence(content.strip())
    try:
        return json.loads(normalized)
    except JSONDecodeError as exc:
        extracted = _extract_top_level_json_block(normalized)
        if extracted is None:
            raise exc
        return json.loads(extracted)


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
