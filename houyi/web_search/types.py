from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""
    content: str | None = None
    score: float | None = None
    source: str | None = None
    published_at: str | None = None
    citations: list[str] | None = None


@dataclass(slots=True)
class WebSearchMetadata:
    cached: bool
    cache_hit: bool
    latency_ms: int | None
    provider: str
    provider_latency_ms: int | None = None
    extraction_latency_ms: int | None = None
    extraction_provider: str | None = None
    error_count: int = 0
    rate_limit_count: int = 0
    errors: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class WebSearchResponse:
    query: str
    provider: str
    results: list[WebSearchResult]
    metadata: WebSearchMetadata
    raw: dict[str, Any] | None = None
