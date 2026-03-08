from houyi.adapters.llm import (
    DEFAULT_MODEL,
    LLMAdapter,
    LLMAdapterFactory,
    LLMMessage,
    LLMResponse,
    OpenAIAdapter,
    SiliconFlowAdapter,
    StreamResponse,
    VertexAIAdapter,
    create_vertex_adapter,
)
from houyi.adapters.llm.base import LLMAdapter as ExportedLLMAdapter
from houyi.adapters.llm.base import LLMMessage as ExportedLLMMessage
from houyi.adapters.llm.base import LLMResponse as ExportedLLMResponse
from houyi.adapters.llm.base import StreamResponse as ExportedStreamResponse
from houyi.adapters.llm.factory import LLMAdapterFactory as ExportedLLMAdapterFactory
from houyi.adapters.llm.factory import _create_vertex_adapter
from houyi.adapters.llm.models import DEFAULT_MODEL as ExportedDefaultModel
from houyi.adapters.llm.openai_adapter import OpenAIAdapter as ExportedOpenAIAdapter
from houyi.adapters.llm.siliconflow_adapter import SiliconFlowAdapter as ExportedSiliconFlowAdapter
from houyi.adapters.llm.vertex_httpx_adapter import VertexAIAdapter as ExportedVertexAIAdapter


def test_llm_adapter_exports_canonical_symbols() -> None:
    assert ExportedDefaultModel == DEFAULT_MODEL
    assert LLMAdapter is ExportedLLMAdapter
    assert LLMAdapterFactory is ExportedLLMAdapterFactory
    assert LLMMessage is ExportedLLMMessage
    assert LLMResponse is ExportedLLMResponse
    assert OpenAIAdapter is ExportedOpenAIAdapter
    assert StreamResponse is ExportedStreamResponse
    assert SiliconFlowAdapter is ExportedSiliconFlowAdapter
    assert VertexAIAdapter is ExportedVertexAIAdapter
    assert create_vertex_adapter is _create_vertex_adapter
