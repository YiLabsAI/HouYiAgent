"""Tests for provider_service.py — ProviderProbe abstraction.

Covers:
- sanitize_error: HTML stripping, truncation
- _is_vertex_provider: detection by ID and domain
- get_probe: factory routing
- VertexAIProbe.test_connection: success, missing credentials, token failure, HTTP errors
- VertexAIProbe.fetch_models: returns well-known models list
- OpenAICompatProbe.test_connection: success, HTTP errors, missing base_url
- OpenAICompatProbe.fetch_models: success, HTTP errors, missing base_url, parse errors
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from houyi_studio.server.chat.provider_service import (
    _VERTEX_KNOWN_MODELS,
    OpenAICompatProbe,
    VertexAIProbe,
    _is_vertex_provider,
    get_probe,
    sanitize_error,
)

# ---------------------------------------------------------------------------
# sanitize_error
# ---------------------------------------------------------------------------


class TestSanitizeError:
    def test_strips_html_tags(self):
        assert sanitize_error("<b>Error</b>") == "Error"

    def test_strips_style_blocks(self):
        html = "<style>*{margin:0}</style><p>Not Found</p>"
        result = sanitize_error(html)
        assert "{margin" not in result
        assert "Not Found" in result

    def test_strips_script_blocks(self):
        html = "<script>alert('x')</script>Hello"
        assert sanitize_error(html) == "Hello"

    def test_truncates_long_text(self):
        long_text = "x" * 500
        result = sanitize_error(long_text, max_len=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_collapses_whitespace(self):
        assert sanitize_error("a   b\n\nc") == "a b c"

    def test_google_404_page(self):
        """Real Google Cloud 404 HTML should be cleaned to readable text."""
        html = (
            "<!DOCTYPE html><html><head><style>*{margin:0;padding:0}"
            "body{font:14px sans-serif}</style></head>"
            "<body>Error 404 (Not Found)!!1 <p>404. That's an error.</p>"
            "<p>The requested URL was not found.</p></body></html>"
        )
        result = sanitize_error(html)
        assert "<" not in result
        assert "{margin" not in result
        assert "404" in result


# ---------------------------------------------------------------------------
# _is_vertex_provider
# ---------------------------------------------------------------------------


class TestIsVertexProvider:
    def test_vertex_prefix(self):
        assert _is_vertex_provider("vertex-gemini") is True
        assert _is_vertex_provider("vertex") is True

    def test_aiplatform_domain(self):
        assert _is_vertex_provider("custom", "https://aiplatform.googleapis.com") is True

    def test_generativelanguage_domain(self):
        assert _is_vertex_provider("custom", "https://generativelanguage.googleapis.com") is True

    def test_non_vertex(self):
        assert _is_vertex_provider("openai", "https://api.openai.com/v1") is False
        assert _is_vertex_provider("siliconflow", "") is False

    def test_empty(self):
        assert _is_vertex_provider("", "") is False


# ---------------------------------------------------------------------------
# get_probe
# ---------------------------------------------------------------------------


class TestGetProbe:
    def test_vertex_by_id(self):
        probe = get_probe("vertex-gemini", "")
        assert isinstance(probe, VertexAIProbe)

    def test_vertex_by_domain(self):
        probe = get_probe("custom", "https://aiplatform.googleapis.com")
        assert isinstance(probe, VertexAIProbe)

    def test_openai_compat(self):
        probe = get_probe("openai", "https://api.openai.com/v1")
        assert isinstance(probe, OpenAICompatProbe)

    def test_default_is_openai(self):
        probe = get_probe("", "")
        assert isinstance(probe, OpenAICompatProbe)


# ---------------------------------------------------------------------------
# VertexAIProbe
# ---------------------------------------------------------------------------


class TestVertexAIProbe:
    @pytest.fixture
    def probe(self):
        return VertexAIProbe()

    @pytest.mark.asyncio
    async def test_test_connection_missing_credentials(self, probe):
        """Should return friendly error when no service account configured."""
        mock_adapter = MagicMock()
        mock_adapter.project_id = None
        mock_adapter._sa = None

        with patch.object(probe, "_get_adapter", return_value=mock_adapter):
            result = await probe.test_connection()

        assert result["ok"] is False
        assert "GOOGLE_APPLICATION_CREDENTIALS" in result["message"]
        assert result["latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_test_connection_token_failure(self, probe):
        """Should return error when token acquisition fails."""
        mock_adapter = MagicMock()
        mock_adapter.project_id = "test-project"
        mock_adapter._sa = {"client_email": "test@test.iam.gserviceaccount.com"}
        mock_adapter._get_access_token = AsyncMock(return_value=None)

        with patch.object(probe, "_get_adapter", return_value=mock_adapter):
            result = await probe.test_connection()

        assert result["ok"] is False
        assert "access token" in result["message"]

    @pytest.mark.asyncio
    async def test_test_connection_success(self, probe):
        """Should return ok=True when chat/completions returns 200."""
        mock_adapter = MagicMock()
        mock_adapter.project_id = "test-project"
        mock_adapter._sa = {"client_email": "test@test.iam.gserviceaccount.com"}
        mock_adapter._get_access_token = AsyncMock(return_value="fake-token")
        mock_adapter._get_openai_base_url.return_value = (
            "https://fake.googleapis.com/v1beta1/projects/test/locations/global/endpoints/openapi"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch.object(probe, "_get_adapter", return_value=mock_adapter),
            patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await probe.test_connection()

        assert result["ok"] is True
        assert "test-project" in result["message"]
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_test_connection_http_error(self, probe):
        """Should return sanitized error on non-200 response."""
        mock_adapter = MagicMock()
        mock_adapter.project_id = "test-project"
        mock_adapter._sa = {"client_email": "test@test.iam.gserviceaccount.com"}
        mock_adapter._get_access_token = AsyncMock(return_value="fake-token")
        mock_adapter._get_openai_base_url.return_value = "https://fake.googleapis.com"

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "<html><body>Unauthorized</body></html>"

        with (
            patch.object(probe, "_get_adapter", return_value=mock_adapter),
            patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await probe.test_connection()

        assert result["ok"] is False
        assert "401" in result["message"]
        assert "<" not in result["message"]

    @pytest.mark.asyncio
    async def test_test_connection_exception(self, probe):
        """Should catch exceptions and return friendly error."""
        mock_adapter = MagicMock()
        mock_adapter.project_id = "test-project"
        mock_adapter._sa = {"client_email": "test@test.iam.gserviceaccount.com"}
        mock_adapter._get_access_token = AsyncMock(side_effect=RuntimeError("openssl not found"))

        with patch.object(probe, "_get_adapter", return_value=mock_adapter):
            result = await probe.test_connection()

        assert result["ok"] is False
        assert "Vertex AI" in result["message"]
        assert "openssl" in result["message"]

    @pytest.mark.asyncio
    async def test_fetch_models_returns_known_list(self, probe):
        """Should return curated Gemini models without making any HTTP calls."""
        result = await probe.fetch_models()

        assert result["error"] is None
        assert len(result["models"]) == len(_VERTEX_KNOWN_MODELS)
        model_ids = [m["id"] for m in result["models"]]
        assert "gemini-2.5-pro" in model_ids
        assert "gemini-2.0-flash" in model_ids
        for m in result["models"]:
            assert m["owned_by"] == "google"


# ---------------------------------------------------------------------------
# OpenAICompatProbe
# ---------------------------------------------------------------------------


class TestOpenAICompatProbe:
    @pytest.fixture
    def probe(self):
        return OpenAICompatProbe()

    @pytest.mark.asyncio
    async def test_test_connection_missing_base_url(self, probe):
        result = await probe.test_connection("", "key")
        assert result["ok"] is False
        assert "base_url" in result["message"]

    @pytest.mark.asyncio
    async def test_test_connection_success(self, probe):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            result = await probe.test_connection("https://api.example.com/v1", "key")

        assert result["ok"] is True
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_test_connection_http_error(self, probe):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            result = await probe.test_connection("https://api.example.com/v1", "key")

        assert result["ok"] is False
        assert "403" in result["message"]

    @pytest.mark.asyncio
    async def test_test_connection_network_error(self, probe):
        with patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_cls.return_value = mock_client

            result = await probe.test_connection("https://api.example.com/v1", "key")

        assert result["ok"] is False
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_fetch_models_missing_base_url(self, probe):
        result = await probe.fetch_models("", "key")
        assert result["models"] == []
        assert "base_url" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_models_success(self, probe):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "owned_by": "openai"},
            ]
        }

        with patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            result = await probe.fetch_models("https://api.openai.com/v1", "key")

        assert result["error"] is None
        assert len(result["models"]) == 2
        # Should be sorted by id
        assert result["models"][0]["id"] == "gpt-3.5-turbo"
        assert result["models"][1]["id"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_fetch_models_http_error(self, probe):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            result = await probe.fetch_models("https://api.example.com/v1", "key")

        assert result["models"] == []
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_models_filters_empty_ids(self, probe):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "", "owned_by": "openai"},  # empty id should be filtered
                {"owned_by": "openai"},  # missing id should be filtered
            ]
        }

        with patch("houyi_studio.server.chat.provider_service.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            result = await probe.fetch_models("https://api.openai.com/v1", "key")

        assert len(result["models"]) == 1
        assert result["models"][0]["id"] == "gpt-4o"
