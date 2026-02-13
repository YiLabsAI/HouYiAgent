"""Google Vertex Gemini adapter using google-genai SDK (REST-based).

Uses the google-genai SDK which communicates via REST API,
making it compatible with HTTP proxies (unlike the gRPC-based vertexai SDK).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.llm.base import LLMAdapter, LLMMessage, LLMResponse


class GoogleVertexGeminiAdapter(LLMAdapter):
    """Adapter for Google Vertex AI Gemini via google-genai SDK (REST)."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        credentials_path: str | None = None,
    ) -> None:
        self.project = project
        self.location = location
        self.model = model
        self.credentials_path = credentials_path
        try:
            from google import genai  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        if self.credentials_path:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self.credentials_path)

        self._client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

    @classmethod
    def from_env(cls) -> GoogleVertexGeminiAdapter:
        project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = (
            os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )
        model = os.getenv("VERTEX_GEMINI_MODEL") or "gemini-2.5-pro"
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not project:
            raise ValueError("VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT must be set")
        return cls(
            project=project, location=location, model=model, credentials_path=credentials_path
        )

    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        normalized_messages = self._normalize_messages(messages)

        system_instruction = None
        contents: list[types.Content] = []
        for message in normalized_messages:
            if message["role"] == "system":
                system_instruction = message["content"]
                continue
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.get("content", ""))],
                )
            )

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            config_kwargs["tool_config"] = self._convert_tool_choice(tool_choice)

        config = types.GenerateContentConfig(**config_kwargs)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return self._normalize_response(response)

    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        try:
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK not installed. Install with: pip install google-genai"
            ) from exc

        normalized_messages = self._normalize_messages(messages)

        system_instruction = None
        contents: list[types.Content] = []
        for message in normalized_messages:
            if message["role"] == "system":
                system_instruction = message["content"]
                continue
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.get("content", ""))],
                )
            )

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        stream = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _normalize_response(self, response: Any) -> LLMResponse:
        content = ""
        tool_calls: list[dict[str, Any]] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                if part.function_call:
                    fc = part.function_call
                    args = fc.args
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append(
                        {
                            "id": f"call_{fc.name}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": args,
                            },
                        }
                    )

        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=usage,
            model=self.model,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool schema to Vertex function declarations."""
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            function_payload = tool.get("function") if isinstance(tool, dict) else None
            if not function_payload:
                continue
            declarations.append(
                {
                    "name": function_payload.get("name"),
                    "description": function_payload.get("description"),
                    "parameters": function_payload.get("parameters"),
                }
            )
        if not declarations:
            return []
        return [{"function_declarations": declarations}]

    def _convert_tool_choice(self, tool_choice: Any) -> dict[str, Any]:
        """Map OpenAI tool_choice to Vertex function calling config."""
        mode = "AUTO"
        if tool_choice in {"required", "any"}:
            mode = "ANY"
        elif tool_choice in {"none"}:
            mode = "NONE"
        return {"function_calling_config": {"mode": mode}}
