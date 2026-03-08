"""Cache management for LLM responses and tool results."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any

from houyi.application.tool_calling.tool_results import ToolResultBuilder

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage LLM and tool caches with stable keys and metadata enrichment."""

    def __init__(
        self,
        llm_cache: dict[str, Any] | None = None,
        tool_cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.llm_cache = llm_cache
        self.tool_cache = tool_cache

    def get_llm_cached(self, cache_key: str) -> Any | None:
        if self.llm_cache is None or not cache_key:
            return None
        return self.llm_cache.get(cache_key)

    def set_llm_cached(self, cache_key: str, response: Any) -> None:
        if self.llm_cache is None or not cache_key:
            return
        self.llm_cache[cache_key] = self._clone_response(response)

    def build_llm_cache_key(
        self,
        adapter: Any,
        messages: list[Any],
        tools: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
    ) -> str | None:
        if self.llm_cache is None:
            return None
        try:
            key_parts = {
                "adapter": adapter.__class__.__name__,
                "messages": self._normalize_messages(messages),
                "tools": tools,
                "model": chat_kwargs.get("model"),
                "temperature": chat_kwargs.get("temperature"),
                "max_tokens": chat_kwargs.get("max_tokens"),
            }
            key_json = json.dumps(key_parts, sort_keys=True, default=str)
            return hashlib.sha256(key_json.encode()).hexdigest()
        except Exception:
            logger.debug("Failed to build LLM cache key", exc_info=True)
            return None

    def get_tool_cached(self, cache_key: str) -> dict[str, Any] | None:
        if self.tool_cache is None or not cache_key:
            return None
        cached_result = self.tool_cache.get(cache_key)
        if cached_result is None:
            return None

        result = dict(cached_result)
        metadata = dict(result.get("metadata") or {})
        metadata["cache_hit"] = True
        metadata["cache_key"] = cache_key
        result["metadata"] = metadata

        raw_payload = result.get("raw")
        if isinstance(raw_payload, dict):
            raw_meta = dict(raw_payload.get("metadata") or {})
            raw_meta["cache_hit"] = True
            raw_meta["cache_key"] = cache_key
            raw_payload["metadata"] = raw_meta
            result["raw"] = raw_payload

        return result

    def set_tool_cached(self, cache_key: str, result: dict[str, Any]) -> None:
        if self.tool_cache is None or not cache_key:
            return
        if ToolResultBuilder.is_error(result):
            return
        self.tool_cache[cache_key] = result

    def build_tool_cache_key(
        self,
        tool_name: str,
        args: dict[str, Any],
        skill: Any | None,
    ) -> str | None:
        if self.tool_cache is None:
            return None
        try:
            key_parts = {
                "tool_name": tool_name,
                "skill_version": getattr(skill, "version", None) if skill else None,
                "args": args,
            }
            key_json = json.dumps(key_parts, sort_keys=True, default=str)
            return hashlib.sha256(key_json.encode()).hexdigest()
        except Exception:
            logger.debug("Failed to build tool cache key", exc_info=True)
            return None

    def enrich_result_with_cache_metadata(
        self,
        result: dict[str, Any],
        cache_hit: bool,
        cache_key: str | None,
        tool_reported_cache_hit: bool,
    ) -> dict[str, Any]:
        cache_hit_for_reporting = cache_hit or tool_reported_cache_hit
        if not cache_hit_for_reporting:
            return result

        result_meta = dict(result.get("metadata") or {})
        result_meta["cache_hit"] = True

        raw_result = result.get("raw")
        raw_metadata = raw_result.get("metadata") if isinstance(raw_result, dict) else None
        existing_cache_key = result_meta.get("cache_key")
        raw_cache_key = raw_metadata.get("cache_key") if isinstance(raw_metadata, dict) else None

        if cache_hit and cache_key:
            result_meta["cache_key"] = cache_key
        elif existing_cache_key:
            result_meta["cache_key"] = existing_cache_key
        elif raw_cache_key:
            result_meta["cache_key"] = raw_cache_key

        result["metadata"] = result_meta
        return result

    @staticmethod
    def _clone_response(response: Any) -> Any:
        if hasattr(response, "model_copy"):
            return response.model_copy(deep=True)
        return copy.deepcopy(response)

    @staticmethod
    def _normalize_messages(messages: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append(msg)
            elif hasattr(msg, "model_dump"):
                normalized.append(msg.model_dump())
            elif hasattr(msg, "dict"):
                normalized.append(msg.dict())
            else:
                normalized.append({"content": str(msg)})
        return normalized


__all__ = ["CacheManager"]
