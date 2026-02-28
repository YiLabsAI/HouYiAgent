"""Provider Service: health checks and model discovery for LLM providers.

Abstracts provider-specific logic (auth, endpoints, model listing) behind
a uniform interface so that chat_api.py stays clean and new providers can
be added by implementing a single class.

Architecture:
    ProviderProbe (ABC)
      ├── VertexAIProbe      — Google Vertex AI (JWT auth via VertexAIAdapter)
      └── OpenAICompatProbe  — Any OpenAI-compatible provider (SiliconFlow, Ollama, etc.)

Each probe exposes two async methods:
    test_connection()  → { ok, message, latency_ms }
    fetch_models()     → { models: [{ id, owned_by }], error: str|None }

Usage from chat_api.py:
    probe = get_probe(provider_id, base_url)
    result = await probe.test_connection(base_url, api_key)
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VERTEX_DOMAINS = ("aiplatform.googleapis.com", "generativelanguage.googleapis.com")


def _is_vertex_provider(provider_id: str, base_url: str = "") -> bool:
    """Detect Vertex AI / Google AI providers by ID prefix or base_url domain."""
    if provider_id.startswith("vertex"):
        return True
    return any(domain in base_url for domain in _VERTEX_DOMAINS)


def sanitize_error(text: str, max_len: int = 200) -> str:
    """Strip HTML tags and truncate error messages for clean API responses.

    Google Cloud 404 pages return full HTML with <style> blocks containing
    CSS rules like ``*{margin:0;padding:0}`` that leak into the UI if only
    tags are stripped.  We therefore remove <style>/<script> blocks (with
    their content) first, then strip remaining tags.
    """
    # Remove <style>...</style> and <script>...</script> blocks (incl. content)
    clean = re.sub(r"<(style|script)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining HTML tags
    clean = re.sub(r"<[^>]+>", "", clean).strip()
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean)
    if len(clean) > max_len:
        clean = clean[:max_len] + "..."
    return clean


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ProviderProbe(ABC):
    """Base class for provider health checks and model discovery."""

    @abstractmethod
    async def test_connection(self, base_url: str = "", api_key: str = "") -> dict[str, Any]:
        """Test connectivity to the provider.

        Returns: { "ok": bool, "message": str, "latency_ms": int }
        """

    @abstractmethod
    async def fetch_models(
        self,
        base_url: str = "",
        api_key: str = "",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch available models from the provider.

        Returns: { "models": [{ "id": str, "owned_by": str }], "error": str|None }
        """


_FETCH_MODELS_CACHE_TTL_SECONDS = 300
_fetch_models_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}


def _build_models_cache_key(provider_kind: str, base_url: str = "") -> str:
    normalized_base = (base_url or "").strip().rstrip("/").lower()
    return f"{provider_kind}:{normalized_base}"


def _get_cached_models(cache_key: str) -> list[dict[str, str]] | None:
    cached = _fetch_models_cache.get(cache_key)
    if not cached:
        return None
    expires_at, models = cached
    if time.time() >= expires_at:
        _fetch_models_cache.pop(cache_key, None)
        return None
    return list(models)


def _set_cached_models(cache_key: str, models: list[dict[str, str]]) -> None:
    _fetch_models_cache[cache_key] = (
        time.time() + _FETCH_MODELS_CACHE_TTL_SECONDS,
        list(models),
    )


# ---------------------------------------------------------------------------
# Vertex AI
# ---------------------------------------------------------------------------

# Well-known Gemini models available on Vertex AI.
# Returned by fetch_models() since Vertex AI does not have a simple
# public REST endpoint to list publisher models without extra permissions.
_VERTEX_KNOWN_MODELS = [
    {"id": "gemini-3.1-pro-preview", "owned_by": "google"},
    {"id": "gemini-3-pro-preview", "owned_by": "google"},
    {"id": "gemini-3-flash-preview", "owned_by": "google"},
    {"id": "gemini-2.5-pro", "owned_by": "google"},
    {"id": "gemini-2.5-flash", "owned_by": "google"},
    {"id": "gemini-2.0-flash", "owned_by": "google"},
    {"id": "gemini-2.0-flash-lite", "owned_by": "google"},
    {"id": "gemini-1.5-pro", "owned_by": "google"},
    {"id": "gemini-1.5-flash", "owned_by": "google"},
]


class VertexAIProbe(ProviderProbe):
    """Health check and model discovery for Google Vertex AI.

    Reuses VertexAIAdapter's JWT-based auth (zero external deps beyond
    openssl).  The same auth path is used for actual LLM calls, so a
    successful health check guarantees the adapter will work.
    """

    def _get_adapter(self):
        """Lazy-import and instantiate VertexAIAdapter."""
        from houyi.llm.vertex_httpx_adapter import VertexAIAdapter

        return VertexAIAdapter()

    async def test_connection(self, base_url: str = "", api_key: str = "") -> dict[str, Any]:
        """Test Vertex AI connectivity via a lightweight chat/completions call.

        Sends ``max_tokens=1`` to minimize cost.  Uses the same OpenAI-compatible
        endpoint that VertexAIAdapter.stream_chat() uses, so success here
        guarantees real calls will also authenticate correctly.
        """
        t0 = time.monotonic()
        try:
            adapter = self._get_adapter()
            if not adapter.project_id or not adapter._sa:
                return {
                    "ok": False,
                    "message": "Vertex AI: GOOGLE_APPLICATION_CREDENTIALS not configured or missing project_id",
                    "latency_ms": 0,
                }
            token = await adapter._get_access_token()
            if not token:
                return {
                    "ok": False,
                    "message": "Vertex AI: failed to obtain access token (check service account)",
                    "latency_ms": 0,
                }

            openai_base = adapter._get_openai_base_url()
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{openai_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "google/gemini-2.0-flash",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )

            latency = int((time.monotonic() - t0) * 1000)
            if resp.status_code == 200:
                return {
                    "ok": True,
                    "message": f"Vertex AI connected (project: {adapter.project_id})",
                    "latency_ms": latency,
                }
            return {
                "ok": False,
                "message": f"HTTP {resp.status_code}: {sanitize_error(resp.text)}",
                "latency_ms": latency,
            }
        except Exception as e:
            latency = int((time.monotonic() - t0) * 1000)
            return {"ok": False, "message": f"Vertex AI: {str(e)[:200]}", "latency_ms": latency}

    async def fetch_models(
        self,
        base_url: str = "",
        api_key: str = "",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Return well-known Gemini models available on Vertex AI.

        Vertex AI's publisher models REST API requires specific IAM permissions
        and returns 404 for many project configurations.  Instead, we return a
        curated list of generally-available Gemini models.
        """
        cache_key = _build_models_cache_key("vertex", base_url)
        if not force_refresh:
            cached = _get_cached_models(cache_key)
            if cached is not None:
                return {"models": cached, "error": None}

        models = list(_VERTEX_KNOWN_MODELS)
        _set_cached_models(cache_key, models)
        return {"models": models, "error": None}


# ---------------------------------------------------------------------------
# OpenAI-compatible (SiliconFlow, Ollama, vLLM, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatProbe(ProviderProbe):
    """Health check and model discovery for OpenAI-compatible providers.

    Works with any provider that exposes ``GET /models`` returning
    ``{ "data": [{ "id": "...", "owned_by": "..." }] }``.
    """

    async def test_connection(self, base_url: str = "", api_key: str = "") -> dict[str, Any]:
        if not base_url:
            return {"ok": False, "message": "base_url is required", "latency_ms": 0}

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                )
            latency = int((time.monotonic() - t0) * 1000)
            if resp.status_code == 200:
                return {"ok": True, "message": "Connection successful", "latency_ms": latency}
            return {
                "ok": False,
                "message": f"HTTP {resp.status_code}: {sanitize_error(resp.text)}",
                "latency_ms": latency,
            }
        except Exception as e:
            latency = int((time.monotonic() - t0) * 1000)
            return {"ok": False, "message": str(e)[:300], "latency_ms": latency}

    async def fetch_models(
        self,
        base_url: str = "",
        api_key: str = "",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if not base_url:
            return {"models": [], "error": "base_url is required"}

        cache_key = _build_models_cache_key("openai_compat", base_url)
        if not force_refresh:
            cached = _get_cached_models(cache_key)
            if cached is not None:
                return {"models": cached, "error": None}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                )
            if resp.status_code != 200:
                return {
                    "models": [],
                    "error": f"HTTP {resp.status_code}: {sanitize_error(resp.text)}",
                }

            data = resp.json()
            # OpenAI-compatible format: { "data": [{ "id": "...", "owned_by": "..." }] }
            raw_models = data.get("data", [])
            models = [
                {"id": m.get("id", ""), "owned_by": m.get("owned_by", "")}
                for m in raw_models
                if m.get("id")
            ]
            models.sort(key=lambda x: x["id"])
            _set_cached_models(cache_key, models)
            return {"models": models, "error": None}
        except Exception as e:
            return {"models": [], "error": str(e)[:300]}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Singleton probes (stateless, safe to reuse)
_vertex_probe = VertexAIProbe()
_openai_probe = OpenAICompatProbe()


def get_probe(provider_id: str, base_url: str = "") -> ProviderProbe:
    """Return the appropriate probe for a provider.

    Routing logic:
    - provider_id starts with "vertex" → VertexAIProbe
    - base_url contains a known Google domain → VertexAIProbe
    - Everything else → OpenAICompatProbe
    """
    if _is_vertex_provider(provider_id, base_url):
        return _vertex_probe
    return _openai_probe
