"""Unit tests for houyi.llm.vertex_httpx_adapter — VertexAIAdapter.

Tests cover:
- Mock mode (no SA / project)
- Service account loading and project_id resolution
- OpenAI-compatible URL construction (global vs regional)
- max_tokens clamping
- Retry / backoff (429, 400, 401, persistent 500)
- Access token caching
- Model name formatting
- Reasoning (reasoning_effort, reasoning_content parsing)
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.config.env_config import EnvConfig
from houyi.llm.vertex_httpx_adapter import VertexAIAdapter


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Reset EnvConfig singleton before/after each test so env patches take effect."""
    EnvConfig._reset()
    yield
    EnvConfig._reset()


def _make_sa_file(tmp_path, project_id="test-project-123"):
    """Create a minimal service account JSON file."""
    sa = {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": "key123",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n",
        "client_email": "test@test-project-123.iam.gserviceaccount.com",
        "client_id": "123456",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    sa_path = tmp_path / "sa.json"
    sa_path.write_text(json.dumps(sa))
    return str(sa_path), sa


class TestVertexAIAdapterMock:
    """Test VertexAIAdapter mock mode (no project ID or service account)."""

    @pytest.mark.asyncio
    async def test_mock_streaming_no_sa(self):
        """No GOOGLE_APPLICATION_CREDENTIALS -> mock mode."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("GOOGLE_CLOUD_PROJECT", None)
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            with patch.dict(os.environ, env, clear=True):
                adapter = VertexAIAdapter()
                assert adapter.project_id is None
                assert adapter._sa is None

                chunks = []
                async for content, _reasoning in adapter.stream_completion("Hello world"):
                    chunks.append(content)

                full = "".join(chunks)
                assert "Mock response" in full
                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_mock_streaming_via_stream_chat(self):
        """stream_chat also falls back to mock when no SA."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("GOOGLE_CLOUD_PROJECT", None)
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            with patch.dict(os.environ, env, clear=True):
                adapter = VertexAIAdapter()
                chunks = []
                async for content, reasoning in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}]
                ):
                    chunks.append((content, reasoning))
                assert len(chunks) > 0
                assert all(r is None for _, r in chunks)


class TestVertexAIAdapterServiceAccount:
    """Test VertexAIAdapter service account loading and project_id resolution."""

    def test_loads_project_id_from_sa_file(self, tmp_path):
        """project_id should come from SA file, not env."""
        sa_path, _ = _make_sa_file(tmp_path, "sa-project")
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_PROJECT": "env-project",
            },
        ):
            adapter = VertexAIAdapter()
            assert adapter.project_id == "sa-project"

    def test_falls_back_to_env_project_id(self, tmp_path):
        """If SA file has no project_id, fall back to env."""
        sa = {
            "type": "service_account",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n",
            "client_email": "test@test.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        sa_path = tmp_path / "sa.json"
        sa_path.write_text(json.dumps(sa))
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": str(sa_path),
                "GOOGLE_CLOUD_PROJECT": "env-project",
            },
        ):
            adapter = VertexAIAdapter()
            assert adapter.project_id == "env-project"

    def test_invalid_sa_file_path(self):
        """Non-existent SA file -> no SA loaded."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/sa.json",
            },
        ):
            adapter = VertexAIAdapter()
            assert adapter._sa is None


class TestVertexAIAdapterURL:
    """Test Vertex AI OpenAI-compatible URL construction."""

    def test_global_endpoint(self, tmp_path):
        sa_path, _ = _make_sa_file(tmp_path)
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            url = adapter._get_openai_base_url()
            assert "aiplatform.googleapis.com/v1beta1" in url
            assert "locations/global" in url
            assert not url.startswith("https://global-")

    def test_regional_endpoint(self, tmp_path):
        sa_path, _ = _make_sa_file(tmp_path)
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            },
        ):
            adapter = VertexAIAdapter()
            url = adapter._get_openai_base_url()
            assert url.startswith("https://us-central1-aiplatform.googleapis.com/v1beta1")
            assert "locations/us-central1" in url

    def test_default_location(self, tmp_path):
        sa_path, _ = _make_sa_file(tmp_path)
        env = os.environ.copy()
        env.pop("GOOGLE_CLOUD_LOCATION", None)
        env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        with patch.dict(os.environ, env, clear=True):
            adapter = VertexAIAdapter()
            assert adapter.location == "us-central1"


class TestVertexAIAdapterMaxTokensClamp:
    """Test max_tokens clamping for Vertex AI."""

    @pytest.mark.asyncio
    async def test_clamps_large_max_tokens(self, tmp_path):
        """max_tokens > 65536 should be clamped."""
        sa_path, _ = _make_sa_file(tmp_path)

        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def aread(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            last_body = None

            def stream(self, method, url, **kwargs):
                MockHttpxClient.last_body = kwargs.get("json")
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for c, _r in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=111111,
                ):
                    chunks.append(c)

        assert MockHttpxClient.last_body["max_tokens"] == 65536
        assert chunks == ["OK"]

    @pytest.mark.asyncio
    async def test_normal_max_tokens_not_clamped(self, tmp_path):
        """max_tokens within range should pass through unchanged."""
        sa_path, _ = _make_sa_file(tmp_path)

        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def aread(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            last_body = None

            def stream(self, method, url, **kwargs):
                MockHttpxClient.last_body = kwargs.get("json")
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                async for _ in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=4096,
                ):
                    pass

        assert MockHttpxClient.last_body["max_tokens"] == 4096


class TestVertexAIAdapterRetry:
    """Test retry / backoff behavior for transient errors."""

    @pytest.mark.asyncio
    async def test_retries_on_429(self, tmp_path):
        """429 should trigger retry with backoff."""
        sa_path, _ = _make_sa_file(tmp_path)
        attempt_count = 0

        class MockResponse:
            def __init__(self, status, body=""):
                self.status_code = status
                self._body = body

            async def aread(self):
                return self._body.encode()

            async def aiter_lines(self):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]})
                yield "data: [DONE]"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, method, url, **kwargs):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count <= 2:
                    return MockResponse(429, '{"error": "rate limited"}')
                return MockResponse(200)

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            async def noop_backoff(*args, **kwargs):
                pass

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch(
                    "houyi.llm.vertex_httpx_adapter.exponential_backoff",
                    side_effect=noop_backoff,
                ),
            ):
                chunks = []
                async for c, _r in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}],
                ):
                    chunks.append(c)

        assert attempt_count == 3
        assert chunks == ["OK"]

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self, tmp_path):
        """400 should fail immediately without retry."""
        sa_path, _ = _make_sa_file(tmp_path)
        attempt_count = 0

        class MockResponse:
            status_code = 400

            async def aread(self):
                return b'{"error": "bad request"}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, method, url, **kwargs):
                nonlocal attempt_count
                attempt_count += 1
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                with pytest.raises(Exception, match="400"):
                    async for _ in adapter.stream_chat(
                        [{"role": "user", "content": "hi"}],
                    ):
                        pass

        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_401_invalidates_token(self, tmp_path):
        """401 should invalidate cached token."""
        sa_path, _ = _make_sa_file(tmp_path)

        class MockResponse:
            status_code = 401

            async def aread(self):
                return b'{"error": "unauthorized"}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, method, url, **kwargs):
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "old-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                with pytest.raises(Exception, match="401"):
                    async for _ in adapter.stream_chat(
                        [{"role": "user", "content": "hi"}],
                    ):
                        pass

        assert adapter._access_token is None
        assert adapter._token_expiry == 0

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_500(self, tmp_path):
        """Persistent 500 should exhaust all retries then raise."""
        sa_path, _ = _make_sa_file(tmp_path)
        attempt_count = 0

        class MockResponse:
            status_code = 500

            async def aread(self):
                return b'{"error": "internal"}'

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, method, url, **kwargs):
                nonlocal attempt_count
                attempt_count += 1
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            async def noop_backoff(*args, **kwargs):
                pass

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch(
                    "houyi.llm.vertex_httpx_adapter.exponential_backoff",
                    side_effect=noop_backoff,
                ),
            ):
                with pytest.raises(Exception, match="500"):
                    async for _ in adapter.stream_chat(
                        [{"role": "user", "content": "hi"}],
                    ):
                        pass

        assert attempt_count == 4


class TestVertexAIAdapterTokenCache:
    """Test access token caching behavior."""

    @pytest.mark.asyncio
    async def test_cached_token_reused(self, tmp_path):
        """Token within expiry window should be reused without re-signing."""
        import time

        sa_path, _ = _make_sa_file(tmp_path)

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "cached-token"
            adapter._token_expiry = time.time() + 600

            token = await adapter._get_access_token()
            assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_expired_token_refreshed(self, tmp_path):
        """Expired token should trigger JWT signing + exchange."""
        import time

        sa_path, _ = _make_sa_file(tmp_path)

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "old-token"
            adapter._token_expiry = time.time() - 100

            mock_httpx_resp = MagicMock()
            mock_httpx_resp.json.return_value = {
                "access_token": "new-token",
                "expires_in": 3600,
            }
            mock_httpx_resp.raise_for_status = MagicMock()

            mock_httpx_client = AsyncMock()
            mock_httpx_client.post = AsyncMock(return_value=mock_httpx_resp)
            mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
            mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

            with (
                patch.object(adapter, "_sign_jwt_with_openssl", return_value="fake-jwt"),
                patch("httpx.AsyncClient", return_value=mock_httpx_client),
            ):
                token = await adapter._get_access_token()

            assert token == "new-token"
            assert adapter._access_token == "new-token"


class TestVertexAIAdapterModelFormat:
    """Test model name formatting in request body."""

    @pytest.mark.asyncio
    async def test_model_has_google_prefix(self, tmp_path):
        """Model in request body should have google/ prefix."""
        sa_path, _ = _make_sa_file(tmp_path)

        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def aread(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            last_body = None

            def stream(self, method, url, **kwargs):
                MockHttpxClient.last_body = kwargs.get("json")
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                async for _ in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    model="gemini-3-pro-preview",
                ):
                    pass

        assert MockHttpxClient.last_body["model"] == "google/gemini-3-pro-preview"


class TestVertexAIAdapterReasoning:
    """Test Gemini reasoning_effort and reasoning_content parsing."""

    @pytest.mark.asyncio
    async def test_enable_reasoning_sends_reasoning_effort(self, tmp_path):
        """enable_reasoning=True should add reasoning_effort='high' to request body."""
        sa_path, _ = _make_sa_file(tmp_path)

        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def aread(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            last_body = None

            def stream(self, method, url, **kwargs):
                MockHttpxClient.last_body = kwargs.get("json")
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                async for _ in adapter.stream_chat(
                    [{"role": "user", "content": "think hard"}],
                    enable_reasoning=True,
                ):
                    pass

        assert MockHttpxClient.last_body["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_no_reasoning_effort_by_default(self, tmp_path):
        """Without enable_reasoning, reasoning_effort should NOT be in body."""
        sa_path, _ = _make_sa_file(tmp_path)

        sse_lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def aread(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            last_body = None

            def stream(self, method, url, **kwargs):
                MockHttpxClient.last_body = kwargs.get("json")
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                async for _ in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}],
                ):
                    pass

        assert "reasoning_effort" not in MockHttpxClient.last_body

    @pytest.mark.asyncio
    async def test_parses_reasoning_content_from_delta(self, tmp_path):
        """reasoning_content in SSE delta should be yielded as second tuple element."""
        sa_path, _ = _make_sa_file(tmp_path)

        sse_lines = [
            "data: "
            + json.dumps({"choices": [{"delta": {"reasoning_content": "Let me think..."}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "The answer is 42"}}]}),
            "data: "
            + json.dumps({"choices": [{"delta": {"content": ".", "reasoning_content": "done"}}]}),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def aread(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, method, url, **kwargs):
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_CLOUD_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for content, reasoning in adapter.stream_chat(
                    [{"role": "user", "content": "think"}],
                    enable_reasoning=True,
                ):
                    chunks.append((content, reasoning))

        assert chunks[0] == ("", "Let me think...")
        assert chunks[1] == ("The answer is 42", None)
        assert chunks[2] == (".", "done")
