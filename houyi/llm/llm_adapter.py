"""LLM Adapter for streaming output.

Supports:
- DeepSeek via SiliconFlow
- Google Gemini via Vertex AI
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Base class for LLM adapters."""

    @abstractmethod
    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from LLM.

        Args:
            prompt: Input prompt
            model: Model name (optional, uses default if not specified)
            **kwargs: Additional model-specific parameters

        Yields:
            Text chunks from the LLM
        """
        pass


class SiliconFlowAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs (SiliconFlow, DeepSeek, vLLM, Ollama, etc.).

    Strategy: openai SDK first (industry standard), httpx raw SSE as fallback.
    The openai SDK provides automatic retries, proper error types, streaming
    support, and token usage tracking — matching LangChain/LlamaIndex patterns.
    """

    _SDK_AVAILABLE: bool | None = None

    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3")
        self.last_usage: dict[str, int] | None = None
        self._sdk_client: object | None = None

        if not self.api_key:
            logger.warning("SILICONFLOW_API_KEY not set, will use mock responses")

        # Detect SDK availability once per class (not per instance)
        if SiliconFlowAdapter._SDK_AVAILABLE is None:
            try:
                import openai  # noqa: F401  # pylint: disable=unused-import

                SiliconFlowAdapter._SDK_AVAILABLE = True
                logger.info("openai SDK available — using SDK mode (recommended)")
            except ImportError:
                SiliconFlowAdapter._SDK_AVAILABLE = False
                logger.warning(
                    "openai SDK not installed — falling back to httpx raw SSE. "
                    "Install with: pip install 'houyi[model-adapters]' or pip install openai"
                )

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream completion from OpenAI-compatible API.

        Returns:
            AsyncIterator of (content, reasoning_content) tuples
        """
        model = model or self.default_model

        if not self.api_key:
            logger.info("Using mock streaming (no API key)")
            words = f"Mock response from {model}: {prompt[:50]}...".split()
            for word in words:
                yield (word + " ", None)
            return

        if SiliconFlowAdapter._SDK_AVAILABLE:
            async for chunk in self._stream_via_sdk(
                prompt, model, enable_reasoning, thinking_budget, **kwargs
            ):
                yield chunk
        else:
            async for chunk in self._stream_via_httpx(
                prompt, model, enable_reasoning, thinking_budget, **kwargs
            ):
                yield chunk

    async def _stream_via_sdk(
        self,
        prompt: str,
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream via openai SDK (preferred path)."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

        extra_body: dict[str, object] = {}
        if enable_reasoning and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
            logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

        sdk_kwargs: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            sdk_kwargs["extra_body"] = extra_body
        for k, v in kwargs.items():
            if v is not None and k not in sdk_kwargs:
                sdk_kwargs[k] = v

        logger.info("SDK streaming to %s model=%s", self.base_url, model)
        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None

        try:
            stream = await client.chat.completions.create(**sdk_kwargs)
            async for chunk in stream:
                # Capture usage from final chunk
                if chunk.usage:
                    self.last_usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = delta.content if delta else None
                # reasoning_content is a DeepSeek extension, access via getattr
                reasoning = getattr(delta, "reasoning_content", None)

                if isinstance(content, str) and content:
                    chunk_count += 1
                if isinstance(reasoning, str) and reasoning:
                    reasoning_count += 1

                if (isinstance(content, str) and content) or (
                    isinstance(reasoning, str) and reasoning
                ):
                    yield (content or "", reasoning if isinstance(reasoning, str) else None)

            logger.info(
                "SDK stream completed: %d content, %d reasoning chunks, usage=%s",
                chunk_count,
                reasoning_count,
                self.last_usage,
            )
        finally:
            await client.close()

    async def _stream_via_httpx(
        self,
        prompt: str,
        model: str,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Fallback: stream via httpx raw SSE when openai SDK is not installed."""
        import httpx

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        extra_body: dict[str, object] = {}
        if enable_reasoning and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
            logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

        logger.info("httpx fallback streaming to %s model=%s", self.base_url, model)

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            payload["extra_body"] = extra_body
        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        chunk_count = 0
        reasoning_count = 0
        self.last_usage = None

        try:
            async with http_client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except Exception:
                        logger.debug("Failed to decode SSE chunk: %r", data)
                        continue

                    if isinstance(event, dict) and "usage" in event:
                        usage_data = event["usage"]
                        if isinstance(usage_data, dict):
                            self.last_usage = {
                                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                                "completion_tokens": usage_data.get("completion_tokens", 0),
                                "total_tokens": usage_data.get("total_tokens", 0),
                            }

                    choices = event.get("choices") if isinstance(event, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue

                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if not isinstance(delta, dict):
                        continue

                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")

                    if isinstance(content, str) and content:
                        chunk_count += 1
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_count += 1

                    if (isinstance(content, str) and content) or (
                        isinstance(reasoning, str) and reasoning
                    ):
                        yield (content or "", reasoning if isinstance(reasoning, str) else None)

            logger.info(
                "httpx stream completed: %d content, %d reasoning chunks, usage=%s",
                chunk_count,
                reasoning_count,
                self.last_usage,
            )
        except Exception as e:
            logger.error("httpx fallback API error: %s", e, exc_info=True)
            raise
        finally:
            await http_client.aclose()


class VertexAIAdapter(LLMAdapter):
    """Adapter for Google Gemini via Vertex AI."""

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")
        self.location = os.getenv("GOOGLE_LOCATION", "us-central1")
        self.default_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

        if not self.project_id:
            logger.warning("GOOGLE_PROJECT_ID not set, will use mock responses")

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from Gemini via Vertex AI."""
        model = model or self.default_model

        if not self.project_id:
            # Mock streaming for testing
            logger.info("Using mock streaming (no project ID)")
            words = f"Mock response from {model}: {prompt[:50]}...".split()
            for word in words:
                yield word + " "
            return

        try:
            from google.cloud import aiplatform
            from vertexai.preview.generative_models import GenerativeModel

            aiplatform.init(project=self.project_id, location=self.location)
            model_instance = GenerativeModel(model)

            response = await model_instance.generate_content_async(
                prompt,
                stream=True,
                **kwargs,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error("Vertex AI error: %s", e, exc_info=True)
            yield f"[Error: {str(e)}]"


class LLMAdapterFactory:
    """Factory for creating LLM adapters."""

    @staticmethod
    def create(provider: str | None = None) -> LLMAdapter:
        """Create an LLM adapter.

        Args:
            provider: Provider name (siliconflow, vertex, or None for default)

        Returns:
            LLM adapter instance
        """
        provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "siliconflow")

        if provider == "siliconflow":
            return SiliconFlowAdapter()
        elif provider == "vertex":
            return VertexAIAdapter()
        else:
            logger.warning("Unknown provider %s, using SiliconFlow", provider)
            return SiliconFlowAdapter()
