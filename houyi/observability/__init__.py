"""Observability module with OpenTelemetry integration.

Provides:
- Span/TraceManager: Core tracing primitives
- TraceContext: Async-safe context propagation
- Instrumentation decorators: @instrument_llm, @instrument_tool, @instrument_retriever
- Manual span contexts: LLMSpanContext, ToolSpanContext
- Privacy-first configuration: ObservabilityConfig, PrivacyConfig
"""

from houyi.observability.config import (
    ObservabilityConfig,
    PrivacyConfig,
    get_config,
    reset_config,
    set_config,
)
from houyi.observability.content_store import (
    ContentRef,
    ContentStore,
    ContentStoreConfig,
    ContentType,
    FileContentStore,
    get_content_store,
    reset_content_store,
    set_content_store,
)
from houyi.observability.context import (
    TraceContext,
    get_current_span,
    set_current_span,
)
from houyi.observability.instrumentation import (
    LLMSpanContext,
    ToolSpanContext,
    instrument_llm,
    instrument_retriever,
    instrument_tool,
)
from houyi.observability.query import (
    ObservabilityQuery,
    QueryResult,
    SpanWithContent,
    TraceView,
    get_query,
    reset_query,
)
from houyi.observability.storage import (
    SpanFilter,
    SpanStorage,
    SpanStorageConfig,
    SQLiteSpanStorage,
    get_storage,
    reset_storage,
    set_storage,
)
from houyi.observability.trace_manager import Span, TraceManager
from houyi.observability.types import (
    CostInfo,
    SpanEvent,
    SpanSchema,
    SpanStatus,
    SpanType,
    TokenUsage,
)

__all__ = [
    # Core
    "CostInfo",
    "Span",
    "SpanEvent",
    "SpanSchema",
    "SpanStatus",
    "SpanType",
    "TokenUsage",
    "TraceManager",
    # Context propagation
    "TraceContext",
    "get_current_span",
    "set_current_span",
    # Instrumentation decorators
    "instrument_llm",
    "instrument_tool",
    "instrument_retriever",
    # Manual span contexts
    "LLMSpanContext",
    "ToolSpanContext",
    # Configuration
    "ObservabilityConfig",
    "PrivacyConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Storage
    "SpanFilter",
    "SpanStorage",
    "SpanStorageConfig",
    "SQLiteSpanStorage",
    "get_storage",
    "set_storage",
    "reset_storage",
    # Content store
    "ContentRef",
    "ContentStore",
    "ContentStoreConfig",
    "ContentType",
    "FileContentStore",
    "get_content_store",
    "set_content_store",
    "reset_content_store",
    # Query interface
    "ObservabilityQuery",
    "QueryResult",
    "SpanWithContent",
    "TraceView",
    "get_query",
    "reset_query",
]
