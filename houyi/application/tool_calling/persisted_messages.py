from __future__ import annotations

import json
from typing import Any

from houyi.application.context.token_estimator import TokenEstimator
from houyi.application.tool_calling.tool_results import ToolResultBuilder


def collect_persisted_tool_message_payloads(
    *,
    intermediate_messages: list[dict[str, Any]],
    model: str | None = None,
    tool_result_max_tokens: int | None = None,
    per_tool_quota: dict[str, int] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    persisted_messages: list[dict[str, Any]] = []
    estimator = (
        TokenEstimator(model=model)
        if isinstance(model, str)
        and model.strip()
        and (tool_result_max_tokens is not None or per_tool_quota)
        else None
    )
    tool_trace_by_call_id = _index_tool_trace(tool_trace)
    for intermediate in intermediate_messages:
        role = str(intermediate.get("role") or "")
        if role == "assistant" and intermediate.get("tool_calls"):
            persisted_messages.append(_build_assistant_payload(intermediate))
            continue
        if role == "tool":
            persisted_messages.append(
                _build_tool_payload(
                    intermediate,
                    tool_trace_by_call_id=tool_trace_by_call_id,
                    estimator=estimator,
                    tool_result_max_tokens=tool_result_max_tokens,
                    per_tool_quota=per_tool_quota,
                )
            )
    return persisted_messages


def compress_tool_result_payload(
    message: dict[str, Any],
    *,
    estimator: TokenEstimator | None,
    tool_result_max_tokens: int | None,
    per_tool_quota: dict[str, int] | None,
) -> dict[str, Any]:
    if estimator is None:
        return message
    original_tokens = max(0, int(estimator.count_message(_to_llm_message(message)) or 0))
    category = _classify_tool_category(_message_name(message))
    quota_value = (
        per_tool_quota.get(category) if category and isinstance(per_tool_quota, dict) else None
    )
    quota = int(quota_value) if isinstance(quota_value, int) else None
    compressed_payload, strategy, overflow_detected = _build_compressed_payload(
        message=message,
        category=category,
        quota=quota,
    )
    needs_compression = overflow_detected or (
        tool_result_max_tokens is not None
        and tool_result_max_tokens > 0
        and original_tokens > tool_result_max_tokens
    )
    if not needs_compression:
        return message
    content = _json_dump_payload(compressed_payload)
    compressed_tokens = max(0, int(estimator.count_text(content) or 0))
    max_tokens = (
        int(tool_result_max_tokens)
        if isinstance(tool_result_max_tokens, int) and tool_result_max_tokens > 0
        else None
    )
    if max_tokens is not None and compressed_tokens > max_tokens:
        fallback_payload = {
            "summary": summarize_tool_content(message, max_chars=220),
            "tool_category": category or "generic",
            "compression_strategy": f"{strategy}_fallback",
            "truncated": True,
        }
        references = _extract_references(
            _parse_tool_payload(_message_content(message)), category=category, quota=3
        )
        if references:
            fallback_payload["references"] = references
        content = _json_dump_payload(fallback_payload)
        compressed_tokens = max(0, int(estimator.count_text(content) or 0))
    metadata = dict(_message_metadata(message))
    metadata["tool_result_profile"] = {
        "compressed": True,
        "tool_category": category or "generic",
        "compression_strategy": strategy,
        "tokens_before": original_tokens,
        "tokens_after": compressed_tokens,
        "tool_result_max_tokens": max_tokens,
        "per_tool_quota": quota,
        "summary": summarize_tool_content(message, max_chars=160),
    }
    return {
        **message,
        "content": content,
        "metadata": metadata,
    }


def summarize_tool_content(message: dict[str, Any], *, max_chars: int) -> str:
    tool_name = _message_name(message) or "tool"
    content = _message_content(message).strip()
    if not content:
        return f"{tool_name} returned empty result"
    payload = _parse_tool_payload(content)
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            pattern = data.get("pattern")
            matches = data.get("matches")
            if isinstance(pattern, str) and isinstance(matches, list):
                return f"{tool_name} search '{pattern}' returned {len(matches)} match(es)"
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return f"{tool_name} error: {_collapse_text(error, max_chars)}"
        return f"{tool_name} returned structured result"
    if isinstance(payload, list):
        return f"{tool_name} returned {len(payload)} item(s)"
    return f"{tool_name}: {_collapse_text(content, max_chars)}"


def _build_compressed_payload(
    *,
    message: dict[str, Any],
    category: str | None,
    quota: int | None,
) -> tuple[dict[str, Any], str, bool]:
    payload = _parse_tool_payload(_message_content(message))
    summary = summarize_tool_content(message, max_chars=220)
    if category == "search":
        results, overflow = _extract_search_results(payload, quota=quota)
        return (
            {
                "summary": summary,
                "tool_category": "search",
                "results": results,
                "truncated": overflow,
            },
            "search_topk_summary",
            overflow,
        )
    if category == "read":
        source = _extract_source_reference(payload)
        excerpt, overflow = _extract_read_excerpt(payload, quota=quota)
        compressed = {
            "summary": summary,
            "tool_category": "read",
            "excerpt": excerpt,
            "truncated": overflow,
        }
        if source:
            compressed["source"] = source
            compressed["references"] = [source]
        return compressed, "read_excerpt", overflow
    if category == "exec":
        exec_payload, overflow = _extract_exec_payload(payload, quota=quota)
        exec_payload["summary"] = summary
        exec_payload["tool_category"] = "exec"
        exec_payload["truncated"] = overflow
        return exec_payload, "exec_io_trim", overflow
    if category == "table":
        table_payload, overflow = _extract_table_payload(payload, quota=quota)
        table_payload["summary"] = summary
        table_payload["tool_category"] = "table"
        table_payload["truncated"] = overflow
        return table_payload, "table_schema_summary", overflow
    content = _message_content(message)
    return (
        {
            "summary": summary,
            "tool_category": "generic",
            "excerpt": _collapse_text(str(content or ""), 800),
            "truncated": len(str(content or "")) > 800,
        },
        "generic_summary",
        len(str(content or "")) > 800,
    )


def _classify_tool_category(tool_name: str | None) -> str | None:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    lowered = tool_name.strip().lower()
    if any(token in lowered for token in ("search", "find", "grep")):
        return "search"
    if any(token in lowered for token in ("read", "fetch", "scrape", "url")):
        return "read"
    if any(token in lowered for token in ("exec", "shell", "command", "python")):
        return "exec"
    if any(token in lowered for token in ("table", "csv", "sql", "dataframe", "sheet")):
        return "table"
    return None


def _parse_tool_payload(content: str) -> Any:
    try:
        return json.loads(str(content or "").strip())
    except Exception:
        return None


def _extract_search_results(payload: Any, *, quota: int | None) -> tuple[list[Any], bool]:
    items: list[Any] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("matches", "results", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    items = value
                    break
        if not items:
            for key in ("matches", "results", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    items = value
                    break
    elif isinstance(payload, list):
        items = payload
    limit = quota if isinstance(quota, int) and quota > 0 else len(items)
    trimmed = [_normalize_reference(item) for item in items[:limit]]
    return trimmed, len(items) > len(trimmed)


def _extract_read_excerpt(payload: Any, *, quota: int | None) -> tuple[str, bool]:
    content = ""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("content", "text", "body", "stdout"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    content = value
                    break
        if not content:
            for key in ("content", "text", "body"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    content = value
                    break
    if not content:
        content = str(payload or "")
    segments = [
        segment.strip() for segment in content.replace("\r\n", "\n").split("\n") if segment.strip()
    ]
    limit = quota if isinstance(quota, int) and quota > 0 else len(segments)
    excerpt = "\n".join(segments[:limit])
    return _collapse_text(excerpt, 4000), len(segments) > limit or len(excerpt) > 4000


def _extract_exec_payload(payload: Any, *, quota: int | None) -> tuple[dict[str, Any], bool]:
    source: dict[str, Any] = payload if isinstance(payload, dict) else {}
    source_data = source.get("data")
    data: dict[str, Any] = source_data if isinstance(source_data, dict) else {}
    limit = quota if isinstance(quota, int) and quota > 0 else 20
    stdout = _truncate_lines(str(data.get("stdout") or source.get("stdout") or ""), limit)
    stderr = _truncate_lines(str(data.get("stderr") or source.get("stderr") or ""), limit)
    code = _collapse_text(str(data.get("code") or source.get("code") or ""), 1200)
    overflow = stdout[1] or stderr[1]
    result: dict[str, Any] = {"stdout": stdout[0], "stderr": stderr[0]}
    if code:
        result["code"] = code
    exit_code = data.get("exit_code", source.get("exit_code"))
    if isinstance(exit_code, int):
        result["exit_code"] = exit_code
    return result, overflow


def _extract_table_payload(payload: Any, *, quota: int | None) -> tuple[dict[str, Any], bool]:
    source: dict[str, Any] = payload if isinstance(payload, dict) else {}
    source_data = source.get("data")
    data: dict[str, Any] = source_data if isinstance(source_data, dict) else {}
    limit = quota if isinstance(quota, int) and quota > 0 else 100
    columns = data.get("columns") or source.get("columns")
    rows = data.get("rows") or source.get("rows")
    schema: dict[str, Any] = {
        "columns": columns[:limit] if isinstance(columns, list) else columns,
        "row_count": data.get(
            "row_count", source.get("row_count", len(rows) if isinstance(rows, list) else None)
        ),
    }
    if isinstance(data.get("count"), int):
        schema["count"] = data.get("count")
    overflow = isinstance(columns, list) and len(columns) > limit
    return schema, overflow


def _extract_source_reference(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    source: dict[str, Any] = payload
    source_data = source.get("data")
    data: dict[str, Any] = source_data if isinstance(source_data, dict) else {}
    for key in ("url", "source", "path", "file_path", "root_path"):
        value = data.get(key) if key in data else source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_references(payload: Any, *, category: str | None, quota: int) -> list[Any]:
    if category == "search":
        results, _ = _extract_search_results(payload, quota=quota)
        return results
    reference = _extract_source_reference(payload)
    return [reference] if reference else []


def _normalize_reference(item: Any) -> Any:
    if isinstance(item, dict):
        compact: dict[str, Any] = {}
        for key in ("title", "snippet", "url", "path", "name", "score"):
            value = item.get(key)
            if value in (None, ""):
                continue
            compact[key] = value
        if compact:
            return compact
    if isinstance(item, str):
        return _collapse_text(item, 240)
    return item


def _truncate_lines(text: str, limit: int) -> tuple[str, bool]:
    if not text:
        return "", False
    lines = text.replace("\r\n", "\n").split("\n")
    trimmed = "\n".join(lines[:limit])
    return _collapse_text(trimmed, 2000), len(lines) > limit or len(trimmed) > 2000


def _collapse_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:max_chars]


def _json_dump_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _index_tool_trace(tool_trace: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    tool_trace_by_call_id: dict[str, dict[str, Any]] = {}
    for entry in tool_trace or []:
        if not isinstance(entry, dict):
            continue
        call_id = entry.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            tool_trace_by_call_id[call_id] = entry
    return tool_trace_by_call_id


def _build_assistant_payload(intermediate: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": str(intermediate.get("content") or ""),
        "reasoning_content": (
            str(intermediate.get("reasoning_content"))
            if isinstance(intermediate.get("reasoning_content"), str)
            else None
        ),
        "tool_calls": intermediate.get("tool_calls"),
    }


def _build_tool_payload(
    intermediate: dict[str, Any],
    *,
    tool_trace_by_call_id: dict[str, dict[str, Any]],
    estimator: TokenEstimator | None,
    tool_result_max_tokens: int | None,
    per_tool_quota: dict[str, int] | None,
) -> dict[str, Any]:
    tool_call_id = (
        str(intermediate.get("tool_call_id")) if intermediate.get("tool_call_id") else None
    )
    trace_meta = tool_trace_by_call_id.get(tool_call_id or "") if tool_call_id is not None else None
    merged_metadata = _merge_trace_metadata(intermediate.get("metadata"), trace_meta)
    payload = {
        "role": "tool",
        "content": _resolve_tool_payload_content(intermediate, trace_meta),
        "tool_call_id": tool_call_id,
        "name": (str(intermediate.get("name")) if intermediate.get("name") else None),
        "metadata": merged_metadata,
    }
    return compress_tool_result_payload(
        payload,
        estimator=estimator,
        tool_result_max_tokens=tool_result_max_tokens,
        per_tool_quota=per_tool_quota,
    )


def _merge_trace_metadata(metadata: Any, trace_meta: dict[str, Any] | None) -> dict[str, Any]:
    merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    if not isinstance(trace_meta, dict):
        return merged_metadata
    if trace_meta.get("round_index") is not None:
        merged_metadata.setdefault("round_index", trace_meta.get("round_index"))
    if trace_meta.get("parallel_group_id") is not None:
        merged_metadata.setdefault("parallel_group_id", trace_meta.get("parallel_group_id"))
    if trace_meta.get("duration_ms") is not None:
        merged_metadata.setdefault("duration_ms", trace_meta.get("duration_ms"))
    if isinstance(trace_meta.get("args"), dict):
        merged_metadata.setdefault("tool_args", trace_meta.get("args"))
    return merged_metadata


def _resolve_tool_payload_content(
    intermediate: dict[str, Any],
    trace_meta: dict[str, Any] | None,
) -> str:
    if isinstance(trace_meta, dict):
        result_payload = trace_meta.get("result")
        if isinstance(result_payload, dict) and "raw" in result_payload:
            return ToolResultBuilder.serialize(result_payload.get("raw"))
    return str(intermediate.get("content") or "")


def _message_name(message: dict[str, Any]) -> str | None:
    name = message.get("name")
    return str(name) if isinstance(name, str) and name else None


def _message_content(message: dict[str, Any]) -> str:
    return str(message.get("content") or "")


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _to_llm_message(message: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": str(message.get("role") or "tool"),
        "content": _message_content(message),
    }
    if isinstance(message.get("tool_call_id"), str) and message.get("tool_call_id"):
        payload["tool_call_id"] = message.get("tool_call_id")
    if isinstance(message.get("name"), str) and message.get("name"):
        payload["name"] = message.get("name")
    return payload
