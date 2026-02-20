"""LLM adapters for different providers.

Public API:
    - ``LLMAdapter``        — abstract base class
    - ``LLMMessage``        — conversation message model
    - ``LLMResponse``       — non-streaming response model
    - ``LLMAdapterFactory`` — factory for creating adapters by provider name
    - ``SiliconFlowAdapter``— OpenAI-compatible adapter (SiliconFlow, DeepSeek, etc.)
    - ``VertexAIAdapter``   — Vertex AI httpx JWT fallback adapter
"""

from houyi.llm.base import LLMAdapter, LLMMessage, LLMResponse, StreamResponse
from houyi.llm.factory import LLMAdapterFactory
from houyi.llm.models import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_RESERVE,
    MODEL_CONTEXT_WINDOWS,
)
from houyi.llm.siliconflow_adapter import SiliconFlowAdapter
from houyi.llm.vertex_httpx_adapter import VertexAIAdapter

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_RESERVE",
    "MODEL_CONTEXT_WINDOWS",
    "LLMAdapter",
    "LLMAdapterFactory",
    "LLMMessage",
    "LLMResponse",
    "SiliconFlowAdapter",
    "StreamResponse",
    "VertexAIAdapter",
]
