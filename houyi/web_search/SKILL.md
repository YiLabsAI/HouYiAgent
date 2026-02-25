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

## Runtime behavior (important)

### Provider selection and fallback

1. If `provider` is explicitly passed in tool input, that provider is used as primary.
2. Else if `WEB_SEARCH_PROVIDER` is set, that provider is used as primary.
3. Else provider is auto-detected from available credentials (`serper -> tavily -> bocha -> searxng`), and falls back to `ddg` when none are configured.

Fallback chain is only built in step 3 (auto-detected mode). If provider is explicitly set (tool input or env), fallback is not auto-expanded.

### Proxy behavior

Proxy is enabled only when `WEB_SEARCH_PROXY_ENABLED=true`.

When enabled, proxy URL resolution order is:
1. `HOUYI_PROXY_URL` (explicit override)
2. System HTTPS proxy
3. System HTTP proxy

When disabled (default), providers are created with `proxy_url=None`.

### `mode="browse"` behavior

- `mode="browse"` sets `include_content=true` and attempts content extraction after search.
- Extraction first tries Jina, then readability fallback.
- If extraction does not return better page text, `content` may remain equal or very close to provider-side summary/snippet.
- This is expected and does not necessarily indicate a bug.

## Troubleshooting quick checks

1. **Unsure whether proxy is active**
   - Check `WEB_SEARCH_PROXY_ENABLED` and `HOUYI_PROXY_URL` in runtime env.
   - If enabled, verify system proxy is also correct.
2. **No fallback observed**
   - Confirm `provider` is not set in tool input and `WEB_SEARCH_PROVIDER` is unset.
3. **`snippet` and `content` look similar in browse mode**
   - Check metadata/extraction provider and network reachability for content fetch.
4. **Bocha relevance is weak for constrained queries**
   - Add explicit query constraints (e.g., `site:infoq.cn` and time qualifiers in query text).
