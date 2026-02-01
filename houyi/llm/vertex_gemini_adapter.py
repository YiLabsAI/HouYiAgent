"""Google Vertex Gemini adapter (OpenAI tool-call compatible output)."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from houyi.llm.base import LLMAdapter, LLMMessage, LLMResponse


class GoogleVertexGeminiAdapter(LLMAdapter):
    """Adapter for Google Vertex AI Gemini (generate_content)."""

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
            import vertexai  # type: ignore
            from google.cloud import aiplatform  # type: ignore
            from vertexai.generative_models import GenerativeModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Google Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform"
            ) from exc

        if self.credentials_path:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self.credentials_path)
        aiplatform.init(project=self.project, location=self.location)
        vertexai.init(project=self.project, location=self.location)
        self._model = GenerativeModel(self.model)

    @classmethod
    def from_env(cls) -> GoogleVertexGeminiAdapter:
        project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = (
            os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )
        model = os.getenv("VERTEX_GEMINI_MODEL") or "gemini-1.5-pro"
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
        normalized_messages = self._normalize_messages(messages)
        system_message = None
        contents: list[dict[str, Any]] = []
        for message in normalized_messages:
            if message["role"] == "system":
                system_message = message["content"]
                continue
            contents.append(
                {
                    "role": "user" if message["role"] == "user" else "model",
                    "parts": [{"text": message.get("content", "")}],
                }
            )

        config: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if system_message:
            config["system_instruction"] = {"parts": [{"text": system_message}]}
        if tools:
            config["tools"] = self._convert_tools_to_vertex(tools)
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            config["tool_config"] = self._convert_tool_choice(tool_choice)

        response = await self._model.generate_content_async(
            contents, generation_config=config, **kwargs
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
        response = await self.chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if response.content:
            yield response.content

    def _normalize_response(self, response: Any) -> LLMResponse:
        content = ""
        tool_calls: list[dict[str, Any]] = []
        parts = getattr(response, "candidates", [])
        if parts:
            content_parts = parts[0].content.parts
            for part in content_parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
                if hasattr(part, "function_call") and part.function_call:
                    function_call = part.function_call
                    args = getattr(function_call, "args", None)
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append(
                        {
                            "id": f"call_{function_call.name}",
                            "type": "function",
                            "function": {
                                "name": function_call.name,
                                "arguments": args,
                            },
                        }
                    )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage={},
            model=self.model,
        )

    def _convert_tools_to_vertex(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool schema to Vertex function declarations."""

        declarations: list[dict[str, Any]] = []
        for tool in tools:
            function_payload = tool.get("function") if isinstance(tool, dict) else None
            if not function_payload:
                continue
            declaration = {
                "name": function_payload.get("name"),
                "description": function_payload.get("description"),
                "parameters": function_payload.get("parameters"),
            }
            declarations.append(declaration)
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
