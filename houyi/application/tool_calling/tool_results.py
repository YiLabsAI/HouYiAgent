"""Tool result building, formatting, and error detection utilities."""

from __future__ import annotations

import json
import string
from typing import Any


class ToolResultBuilder:
    """Construct and format tool-call result payloads."""

    @staticmethod
    def build(
        raw: Any,
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a normalised tool-result dict from *raw* output."""
        if isinstance(raw, dict):
            raw_payload = raw
        elif hasattr(raw, "model_dump"):
            raw_payload = raw.model_dump()
        else:
            raw_payload = {"result": raw}
        is_error = isinstance(raw_payload, dict) and "error" in raw_payload

        return {
            "call_id": call_id,
            "raw": raw_payload,
            "content": ToolResultBuilder.serialize(raw_payload),
            "is_error": is_error,
            "metadata": metadata or {},
        }

    @staticmethod
    def format(result: dict[str, Any]) -> str:
        """Extract or serialize the displayable content of a result."""
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        return ToolResultBuilder.serialize(result)

    @staticmethod
    def content_length(result: Any) -> int:
        """Return the serialized content length for a result or raw payload."""
        return len(
            ToolResultBuilder.format(result) if isinstance(result, dict) else str(result or "")
        )

    @staticmethod
    def serialize(payload: Any) -> str:
        """JSON-serialize a payload for tool-message content."""
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return json.dumps({"result": str(payload)}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def is_error(result: Any) -> bool:
        """Return True if *result* represents a tool execution error."""
        if not isinstance(result, dict):
            return False
        if result.get("is_error"):
            return True
        raw = result.get("raw")
        return isinstance(raw, dict) and "error" in raw

    @staticmethod
    def coerce_payload(raw: Any) -> dict[str, Any]:
        """Coerce arbitrary output into a dict payload."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            return {"result": f"<{type(raw).__name__}:{len(raw)} bytes>"}
        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        return {"result": raw}

    @staticmethod
    def extract_error_detail(result: Any) -> str | None:
        """Extract the most actionable error detail from a result payload."""
        payload = result.get("raw") if isinstance(result, dict) and "raw" in result else result
        if not isinstance(payload, dict):
            return None
        for key in ("message", "cause", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def is_binary_like_content(result: Any) -> bool:
        """Detect payloads that should not be expanded verbatim into tool messages."""
        payload = result.get("raw") if isinstance(result, dict) and "raw" in result else result
        if isinstance(payload, (bytes, bytearray)):
            return True
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, (bytes, bytearray)):
                    return True
        text = ToolResultBuilder.format(result) if isinstance(result, dict) else str(payload or "")
        if not text:
            return False
        non_printable = sum(1 for ch in text if ch not in string.printable and ch not in "\n\r\t")
        return non_printable > 0 and non_printable / max(len(text), 1) > 0.1

    @staticmethod
    def parse_arguments(raw_args: Any) -> dict[str, Any]:
        """Parse tool arguments from the model response (string or dict)."""
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
        return {}


__all__ = ["ToolResultBuilder"]
