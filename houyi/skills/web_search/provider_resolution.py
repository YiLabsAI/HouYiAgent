from __future__ import annotations

from collections.abc import Iterable

_SUPPORTED_WEB_SEARCH_PROVIDERS = frozenset({"ddg", "searxng", "serper", "tavily", "bocha"})


def normalize_web_search_provider_name(value: str | None) -> str:
    return (value or "").strip().lower()


def resolve_supported_web_search_provider(
    requested_provider: str | None,
    *,
    configured_provider: str | None = None,
    fallback_provider: str | None = None,
    allowed_providers: Iterable[str] | None = None,
) -> str:
    allowed = frozenset(allowed_providers or _SUPPORTED_WEB_SEARCH_PROVIDERS)

    normalized_requested = normalize_web_search_provider_name(requested_provider)
    if normalized_requested in allowed:
        return normalized_requested

    normalized_configured = normalize_web_search_provider_name(configured_provider)
    if normalized_configured in allowed:
        return normalized_configured

    normalized_fallback = normalize_web_search_provider_name(fallback_provider)
    if normalized_fallback in allowed:
        return normalized_fallback

    if "serper" in allowed:
        return "serper"
    if "ddg" in allowed:
        return "ddg"
    return sorted(allowed)[0]


__all__ = [
    "_SUPPORTED_WEB_SEARCH_PROVIDERS",
    "normalize_web_search_provider_name",
    "resolve_supported_web_search_provider",
]
