"""Observability module with OpenTelemetry integration.

Provides:
- Span/TraceManager: Core tracing primitives
- TraceContext: Async-safe context propagation
- Instrumentation decorators: @instrument_llm, @instrument_tool, @instrument_retriever
- Manual span contexts: LLMSpanContext, ToolSpanContext
- Privacy-first configuration: ObservabilityConfig, PrivacyConfig
"""

from houyi.infrastructure.observability.config import (
    ObservabilityConfig,
    PrivacyConfig,
    get_config,
    reset_config,
    set_config,
)
from houyi.infrastructure.observability.content_store import (
    ContentRef,
    ContentStore,
    ContentStoreConfig,
    ContentType,
    FileContentStore,
    get_content_store,
    reset_content_store,
    set_content_store,
)
from houyi.infrastructure.observability.context import (
    TraceContext,
    get_current_span,
    set_current_span,
)
from houyi.infrastructure.observability.instrumentation import (
    LLMSpanContext,
    ToolSpanContext,
    instrument_llm,
    instrument_retriever,
    instrument_tool,
)
from houyi.infrastructure.observability.query import (
    ObservabilityQuery,
    QueryResult,
    SpanWithContent,
    TraceView,
    get_query,
    reset_query,
)
from houyi.infrastructure.observability.storage import (
    SpanFilter,
    SpanStorage,
    SpanStorageConfig,
    SQLiteSpanStorage,
    get_storage,
    reset_storage,
    set_storage,
)
from houyi.infrastructure.observability.trace_manager import Span, TraceManager
from houyi.infrastructure.observability.types import (
    CostInfo,
    SpanEvent,
    SpanSchema,
    SpanStatus,
    SpanType,
    TokenUsage,
)

__all__ = [
    # Content store
    "ContentRef",
    "ContentStore",
    "ContentStoreConfig",
    "ContentType",
    # Core
    "CostInfo",
    "FileContentStore",
    # Manual span contexts
    "LLMSpanContext",
    # Configuration
    "ObservabilityConfig",
    # Query interface
    "ObservabilityQuery",
    "PrivacyConfig",
    "QueryResult",
    "SQLiteSpanStorage",
    "Span",
    "SpanEvent",
    # Storage
    "SpanFilter",
    "SpanSchema",
    "SpanStatus",
    "SpanStorage",
    "SpanStorageConfig",
    "SpanType",
    "SpanWithContent",
    "TokenUsage",
    "ToolSpanContext",
    # Context propagation
    "TraceContext",
    "TraceManager",
    "TraceView",
    "get_config",
    "get_content_store",
    "get_current_span",
    "get_query",
    "get_storage",
    # Instrumentation decorators
    "instrument_llm",
    "instrument_retriever",
    "instrument_tool",
    "reset_config",
    "reset_content_store",
    "reset_query",
    "reset_storage",
    "set_config",
    "set_content_store",
    "set_current_span",
    "set_storage",
]
