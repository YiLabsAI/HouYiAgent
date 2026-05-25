"""Runtime tools used during research execution.

Each tool is a callable with a name attribute and an OpenAI-format
schema dict.  AgentRunner picks these up automatically via
_get_tool_schemas() and _execute_tool().
"""

from __future__ import annotations

import json
import logging
from typing import Any

from houyi.skills.web_search.service import WebSearchService

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Wraps WebSearchService as a callable runtime tool for AgentRunner.

    The LLM sees this as web_search(query, max_results?, include_content?)
    and the runner dispatches the call here.
    """

    name = "web_search"

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information. Returns a list of results "
                "with title, URL, snippet, and optionally full page content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Whether to fetch full page content.",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        },
    }

    def __init__(self, web_search_service: WebSearchService) -> None:
        self._service = web_search_service

    async def __call__(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = True,
    ) -> str:
        """Execute web search and return JSON-serialized results."""
        try:
            response = await self._service.search(
                query,
                max_results=max_results,
                include_content=include_content,
            )
            results = [
                {
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "content": (r.content or "")[:2000] if include_content else "",
                }
                for r in response.results
            ]
            return json.dumps({"results": results, "provider": response.provider})
        except Exception as exc:
            logger.warning("web_search tool failed for query=%r: %s", query, exc)
            return json.dumps({"error": str(exc), "results": []})
