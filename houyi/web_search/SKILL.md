---
name: web_search
version: 0.3.0
description: Multi-provider web search with caching, retry, and content extraction
author: Houyi Team
invocationPolicy:
  sideEffect: network
  modelAutoInvoke: allow_with_consent
permissions:
  network:
    enabled: true
hooks:
  - event: PreToolUse
    matcher: web_search
  - event: PostToolUse
    matcher: web_search
---

# Web Search Skill

Multi-provider web search with automatic fallback, caching, and content extraction.

## Providers

| Provider | Type | API Key | Install |
|----------|------|---------|---------|
| Serper | Remote API | `SERPER_API_KEY` | Built-in |
| Tavily | Remote API | `TAVILY_API_KEY` | `pip install 'houyi[websearch-tavily]'` |
| DuckDuckGo | HTML search | None | `pip install 'houyi[websearch-ddg]'` |
| Bocha | Remote API | `BOCHA_API_KEY` | Built-in |
| SearxNG | Self-hosted | `SEARXNG_BASE_URL` | Built-in |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_SEARCH_PROVIDER` | auto-detect | Default provider name |
| `WEB_SEARCH_TIMEOUT` | `10` | Request timeout (seconds) |
| `WEB_SEARCH_CACHE_ENABLED` | `true` | Enable result caching |
| `WEB_SEARCH_CACHE_TTL` | `300` | Cache TTL (seconds) |
| `WEB_SEARCH_CACHE_MAX_SIZE` | `256` | Max cached entries |
| `WEB_SEARCH_PROXY_ENABLED` | `false` | Enable proxy for requests |

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | str | required | Search query |
| `max_results` | int | 3 | Maximum results to return |
| `provider` | str | auto | Override default provider |
| `mode` | str | — | `"browse"` to fetch full page content |
| `use_cache` | bool | true | Override cache behavior |
