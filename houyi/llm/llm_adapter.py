"""LLM Adapter for streaming output.

Supports:
- DeepSeek via SiliconFlow
- Google Gemini via Vertex AI
"""

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
    """Adapter for DeepSeek via SiliconFlow API."""

    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3")

        if not self.api_key:
            logger.warning("SILICONFLOW_API_KEY not set, will use mock responses")

    async def stream_completion(
        self,
        prompt: str,
        model: str | None = None,
        enable_reasoning: bool = False,
        thinking_budget: int | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Stream completion from DeepSeek via SiliconFlow.

        Returns:
            AsyncIterator of (content, reasoning_content) tuples
        """
        model = model or self.default_model

        if not self.api_key:
            # Mock streaming for testing
            logger.info("Using mock streaming (no API key)")
            words = f"Mock response from {model}: {prompt[:50]}...".split()
            for word in words:
                yield (word + " ", None)
            return

        try:
            # Use OpenAI-compatible API with proper timeout
            import httpx
            import openai

            # Create httpx client with timeout
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
            )

            # Prepare extra_body for reasoning models
            extra_body = {}
            if enable_reasoning and thinking_budget:
                extra_body["thinking_budget"] = thinking_budget
                logger.info("Reasoning enabled with thinking_budget=%d", thinking_budget)

            logger.info("Creating streaming completion request to %s", self.base_url)

            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body=extra_body if extra_body else None,
                **kwargs,
            )

            logger.info("Stream created, starting to receive chunks")
            chunk_count = 0
            reasoning_count = 0

            async for chunk in stream:
                content = None
                reasoning = None

                if chunk.choices[0].delta.content:
                    chunk_count += 1
                    content = chunk.choices[0].delta.content

                # Handle reasoning content for reasoning models
                if (
                    hasattr(chunk.choices[0].delta, "reasoning_content")
                    and chunk.choices[0].delta.reasoning_content
                ):
                    reasoning_count += 1
                    reasoning = chunk.choices[0].delta.reasoning_content

                if content or reasoning:
                    yield (content or "", reasoning)

            logger.info(
                "Stream completed: %d content chunks, %d reasoning chunks",
                chunk_count,
                reasoning_count,
            )
            await http_client.aclose()

        except Exception as e:
            logger.error("SiliconFlow API error: %s", e, exc_info=True)
            yield (f"[Error: {str(e)}]", None)


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
