"""Unit tests for houyi.llm.llm_adapter — stream_chat and stream_completion.

Tests cover:
- LLMAdapter base default stream_chat (extracts last user → delegates to stream_completion)
- LLMAdapter base stream_chat with no user message
- SiliconFlowAdapter mock mode (no API key)
- SiliconFlowAdapter stream_chat mock path
- SiliconFlowAdapter stream_completion delegates to stream_chat
- LLMAdapterFactory creation
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from houyi.config.env_config import EnvConfig
from houyi.llm.llm_adapter import (
    LLMAdapter,
    LLMAdapterFactory,
    SiliconFlowAdapter,
    VertexAIAdapter,
)


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Reset EnvConfig singleton before/after each test so env patches take effect."""
    EnvConfig._reset()
    yield
    EnvConfig._reset()


# --- Concrete subclass for testing base class behavior ---


class StubAdapter(LLMAdapter):
    """Minimal concrete adapter for testing base class stream_chat default."""

    def __init__(self):
        self.received_prompts: list[str] = []

    async def stream_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        self.received_prompts.append(prompt)
        for word in prompt.split():
            yield word


class TestLLMAdapterBaseStreamChat:
    """Test the base class default stream_chat implementation."""

    @pytest.mark.asyncio
    async def test_extracts_last_user_message(self):
        adapter = StubAdapter()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        chunks = []
        async for content, reasoning in adapter.stream_chat(messages):
            chunks.append((content, reasoning))

        # Should have extracted "Second question" (last user message)
        assert adapter.received_prompts == ["Second question"]
        # Each word yields (word, None)
        assert all(r is None for _, r in chunks)
        assert len(chunks) == 2  # "Second" and "question"

    @pytest.mark.asyncio
    async def test_no_user_message_sends_empty(self):
        adapter = StubAdapter()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "assistant", "content": "Hello"},
        ]
        chunks = []
        async for content, _reasoning in adapter.stream_chat(messages):
            chunks.append(content)

        assert adapter.received_prompts == [""]
        assert chunks == []  # empty string splits to nothing

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        adapter = StubAdapter()
        chunks = []
        async for content, _reasoning in adapter.stream_chat([]):
            chunks.append(content)

        assert adapter.received_prompts == [""]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_completion_tuple_passthrough(self):
        """If stream_completion yields tuples, base stream_chat passes them through."""

        class TupleAdapter(LLMAdapter):
            async def stream_completion(self, prompt, model=None, **kwargs):
                yield ("content", "reasoning")
                yield ("more", None)

        adapter = TupleAdapter()
        chunks = []
        async for c, r in adapter.stream_chat([{"role": "user", "content": "hi"}]):
            chunks.append((c, r))

        assert chunks == [("content", "reasoning"), ("more", None)]


class TestSiliconFlowAdapterMockMode:
    """Test SiliconFlowAdapter when no API key is set (mock mode)."""

    @pytest.mark.asyncio
    async def test_mock_streaming_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            # Ensure no API key
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                # Reset SDK detection to avoid stale state
                SiliconFlowAdapter._SDK_AVAILABLE = None
                adapter = SiliconFlowAdapter()
                assert adapter.api_key is None

                messages = [{"role": "user", "content": "Hello world"}]
                chunks = []
                async for content, reasoning in adapter.stream_chat(messages):
                    chunks.append((content, reasoning))

                # Mock mode yields words from "Mock response from {model}: {content}..."
                assert len(chunks) > 0
                full_content = "".join(c for c, _ in chunks)
                assert "Mock response" in full_content
                assert all(r is None for _, r in chunks)

    @pytest.mark.asyncio
    async def test_stream_completion_delegates_to_stream_chat(self):
        """stream_completion should delegate to stream_chat."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._SDK_AVAILABLE = None
                adapter = SiliconFlowAdapter()

                chunks = []
                async for chunk in adapter.stream_completion("Test prompt"):
                    chunks.append(chunk)

                # stream_completion yields tuples from stream_chat
                assert len(chunks) > 0
                full = "".join(c for c, _ in chunks)
                assert "Mock response" in full

    @pytest.mark.asyncio
    async def test_mock_extracts_last_user_content(self):
        """Mock mode should use last user message content."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("SILICONFLOW_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                SiliconFlowAdapter._SDK_AVAILABLE = None
                adapter = SiliconFlowAdapter()

                messages = [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Tell me about Python"},
                ]
                chunks = []
                async for content, _ in adapter.stream_chat(messages):
                    chunks.append(content)

                full = "".join(chunks)
                assert "Tell me about Python" in full


class TestSiliconFlowAdapterSDKPath:
    """Test SiliconFlowAdapter SDK path with mocked openai client."""

    @pytest.mark.asyncio
    async def test_sdk_stream_chat(self):
        """Mock the openai SDK path to verify stream_chat behavior."""

        # Create mock chunks that simulate openai SDK response
        class MockDelta:
            def __init__(self, content=None, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content

        class MockChoice:
            def __init__(self, delta):
                self.delta = delta

        class MockChunk:
            def __init__(self, choices=None, usage=None):
                self.choices = choices or []
                self.usage = usage

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(MockDelta(content="Hello"))])
            yield MockChunk(choices=[MockChoice(MockDelta(content=" world"))])
            yield MockChunk(choices=[MockChoice(MockDelta(reasoning_content="thinking"))])
            yield MockChunk(choices=[], usage=MockUsage())

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_client.close = AsyncMock()

        # Ensure openai module exists (may not be installed)
        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = True
            adapter = SiliconFlowAdapter()

            with patch.dict(sys.modules, {"openai": fake_openai}):
                chunks = []
                async for content, reasoning in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}]
                ):
                    chunks.append((content, reasoning))

        assert len(chunks) == 3
        assert chunks[0] == ("Hello", None)
        assert chunks[1] == (" world", None)
        assert chunks[2] == ("", "thinking")
        assert adapter.last_usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class TestSiliconFlowAdapterHttpxPath:
    """Test SiliconFlowAdapter httpx fallback path."""

    @pytest.mark.asyncio
    async def test_httpx_stream_chat(self):
        """Mock the httpx path to verify stream_chat behavior."""
        import json

        sse_lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "Hi"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": " there"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"reasoning_content": "let me think"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
                    "choices": [],
                }
            ),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, *args, **kwargs):
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for content, reasoning in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}]
                ):
                    chunks.append((content, reasoning))

        assert len(chunks) == 3
        assert chunks[0] == ("Hi", None)
        assert chunks[1] == (" there", None)
        assert chunks[2] == ("", "let me think")
        assert adapter.last_usage == {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        }


class TestSiliconFlowAdapterSDKReasoning:
    """Test SDK path with reasoning enabled (covers extra_body and kwargs branches)."""

    @pytest.mark.asyncio
    async def test_sdk_with_reasoning_and_kwargs(self):
        """Cover enable_reasoning + thinking_budget + extra kwargs."""

        class MockDelta:
            def __init__(self, content=None, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content

        class MockChoice:
            def __init__(self, delta):
                self.delta = delta

        class MockChunk:
            def __init__(self, choices=None, usage=None):
                self.choices = choices or []
                self.usage = usage

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(MockDelta(content="A"))])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_client.close = AsyncMock()

        fake_openai = ModuleType("openai")
        fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = True
            adapter = SiliconFlowAdapter()

            with patch.dict(sys.modules, {"openai": fake_openai}):
                chunks = []
                async for c, r in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_reasoning=True,
                    thinking_budget=1024,
                    temperature=0.7,
                ):
                    chunks.append((c, r))

        assert chunks == [("A", None)]
        # Verify extra_body was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"] == {"thinking_budget": 1024}


class TestSiliconFlowHttpxEdgeCases:
    """Test httpx path edge cases: empty lines, bad JSON, missing delta."""

    @pytest.mark.asyncio
    async def test_httpx_edge_cases(self):
        import json as json_mod

        sse_lines = [
            "",  # empty line
            "event: ping",  # non-data line
            "data: ",  # empty data
            "data: not-json",  # bad JSON
            "data: " + json_mod.dumps({"choices": []}),  # empty choices
            "data: " + json_mod.dumps({"choices": [{"delta": None}]}),  # null delta
            "data: " + json_mod.dumps({"choices": [{"delta": {"content": "OK"}}]}),
            "data: "
            + json_mod.dumps(
                {
                    "choices": [{"delta": {"reasoning_content": "think"}}],
                }
            ),
            "data: [DONE]",
        ]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockHttpxClient:
            def stream(self, *args, **kwargs):
                return MockResponse()

            async def aclose(self):
                pass

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}):
            EnvConfig._reset()
            SiliconFlowAdapter._SDK_AVAILABLE = False
            adapter = SiliconFlowAdapter()

            with patch("httpx.AsyncClient", return_value=MockHttpxClient()):
                chunks = []
                async for c, r in adapter.stream_chat(
                    [{"role": "user", "content": "Hi"}],
                    enable_reasoning=True,
                    thinking_budget=512,
                ):
                    chunks.append((c, r))

        assert len(chunks) == 2
        assert chunks[0] == ("OK", None)
        assert chunks[1] == ("", "think")


class TestVertexAIAdapterMock:
    """Test VertexAIAdapter mock mode (no project ID or service account)."""

    @pytest.mark.asyncio
    async def test_mock_streaming_no_sa(self):
        """No GOOGLE_APPLICATION_CREDENTIALS → mock mode."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("GOOGLE_PROJECT_ID", None)
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            with patch.dict(os.environ, env, clear=True):
                adapter = VertexAIAdapter()
                assert adapter.project_id is None
                assert adapter._sa is None

                chunks = []
                async for chunk in adapter.stream_completion("Hello world"):
                    chunks.append(chunk)

                full = "".join(chunks)
                assert "Mock response" in full
                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_mock_streaming_via_stream_chat(self):
        """stream_chat also falls back to mock when no SA."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("GOOGLE_PROJECT_ID", None)
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

    def _make_sa_file(self, tmp_path, project_id="test-project-123"):
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

    def test_loads_project_id_from_sa_file(self, tmp_path):
        """project_id should come from SA file, not env."""
        sa_path, _ = self._make_sa_file(tmp_path, "sa-project")
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_PROJECT_ID": "env-project",
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
                "GOOGLE_PROJECT_ID": "env-project",
            },
        ):
            adapter = VertexAIAdapter()
            assert adapter.project_id == "env-project"

    def test_invalid_sa_file_path(self):
        """Non-existent SA file → no SA loaded."""
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
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            url = adapter._get_openai_base_url()
            assert "aiplatform.googleapis.com/v1beta1" in url
            assert "locations/global" in url
            assert not url.startswith("https://global-")

    def test_regional_endpoint(self, tmp_path):
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)
        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
                "GOOGLE_LOCATION": "us-central1",
            },
        ):
            adapter = VertexAIAdapter()
            url = adapter._get_openai_base_url()
            assert url.startswith("https://us-central1-aiplatform.googleapis.com/v1beta1")
            assert "locations/us-central1" in url

    def test_default_location(self, tmp_path):
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)
        env = os.environ.copy()
        env.pop("GOOGLE_LOCATION", None)
        env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        with patch.dict(os.environ, env, clear=True):
            adapter = VertexAIAdapter()
            assert adapter.location == "us-central1"


class TestVertexAIAdapterMaxTokensClamp:
    """Test max_tokens clamping for Vertex AI."""

    @pytest.mark.asyncio
    async def test_clamps_large_max_tokens(self, tmp_path):
        """max_tokens > 65536 should be clamped."""
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            # Manually set token to skip real auth
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
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
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
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)
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
                "GOOGLE_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            async def noop_backoff(*args, **kwargs):
                pass

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch("houyi.llm.llm_adapter._exponential_backoff", side_effect=noop_backoff),
            ):
                chunks = []
                async for c, _r in adapter.stream_chat(
                    [{"role": "user", "content": "hi"}],
                ):
                    chunks.append(c)

        assert attempt_count == 3  # 2 retries + 1 success
        assert chunks == ["OK"]

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self, tmp_path):
        """400 should fail immediately without retry."""
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)
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
                "GOOGLE_LOCATION": "global",
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

        assert attempt_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_401_invalidates_token(self, tmp_path):
        """401 should invalidate cached token."""
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
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

        # Token should be invalidated after 401
        assert adapter._access_token is None
        assert adapter._token_expiry == 0

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_500(self, tmp_path):
        """Persistent 500 should exhaust all retries then raise."""
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)
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
                "GOOGLE_LOCATION": "global",
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "fake-token"
            adapter._token_expiry = 9999999999.0

            async def noop_backoff(*args, **kwargs):
                pass

            with (
                patch("httpx.AsyncClient", return_value=MockHttpxClient()),
                patch("houyi.llm.llm_adapter._exponential_backoff", side_effect=noop_backoff),
            ):
                with pytest.raises(Exception, match="500"):
                    async for _ in adapter.stream_chat(
                        [{"role": "user", "content": "hi"}],
                    ):
                        pass

        assert attempt_count == 4  # 1 initial + 3 retries


class TestVertexAIAdapterTokenCache:
    """Test access token caching behavior."""

    @pytest.mark.asyncio
    async def test_cached_token_reused(self, tmp_path):
        """Token within expiry window should be reused without re-signing."""
        import time

        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "cached-token"
            adapter._token_expiry = time.time() + 600  # 10 min from now

            token = await adapter._get_access_token()
            assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_expired_token_refreshed(self, tmp_path):
        """Expired token should trigger JWT signing + exchange."""
        import time

        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": sa_path,
            },
        ):
            adapter = VertexAIAdapter()
            adapter._access_token = "old-token"
            adapter._token_expiry = time.time() - 100  # expired

            with (
                patch.object(adapter, "_sign_jwt_with_openssl", return_value="fake-jwt"),
                patch("urllib.request.urlopen") as mock_urlopen,
            ):
                mock_resp = MagicMock()
                mock_resp.read.return_value = json.dumps(
                    {
                        "access_token": "new-token",
                        "expires_in": 3600,
                    }
                ).encode()
                mock_urlopen.return_value.__enter__.return_value = mock_resp

                token = await adapter._get_access_token()

            assert token == "new-token"
            assert adapter._access_token == "new-token"


class TestVertexAIAdapterModelFormat:
    """Test model name formatting in request body."""

    @pytest.mark.asyncio
    async def test_model_has_google_prefix(self, tmp_path):
        """Model in request body should have google/ prefix."""
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
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
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
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
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
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
        sa_path, _ = TestVertexAIAdapterServiceAccount()._make_sa_file(tmp_path)

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
                "GOOGLE_LOCATION": "global",
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

        # First chunk: only reasoning
        assert chunks[0] == ("", "Let me think...")
        # Second chunk: only content
        assert chunks[1] == ("The answer is 42", None)
        # Third chunk: both content and reasoning
        assert chunks[2] == (".", "done")


class TestRetryHelpers:
    """Test shared retry/backoff helper functions."""

    def test_retryable_status_codes(self):
        from houyi.llm.llm_adapter import _is_retryable_status

        assert _is_retryable_status(429)
        assert _is_retryable_status(500)
        assert _is_retryable_status(502)
        assert _is_retryable_status(503)
        assert _is_retryable_status(504)
        assert not _is_retryable_status(400)
        assert not _is_retryable_status(401)
        assert not _is_retryable_status(403)
        assert not _is_retryable_status(404)
        assert not _is_retryable_status(200)

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleeps(self):
        from houyi.llm.llm_adapter import _exponential_backoff

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _exponential_backoff(0, base=1.0, cap=10.0)
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert 0 <= delay <= 1.0  # attempt 0: max delay = min(1*2^0, 10) = 1.0

    @pytest.mark.asyncio
    async def test_exponential_backoff_caps(self):
        from houyi.llm.llm_adapter import _exponential_backoff

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _exponential_backoff(10, base=1.0, cap=5.0)
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert 0 <= delay <= 5.0  # capped at 5.0


class TestLLMAdapterFactory:
    """Test LLMAdapterFactory.create."""

    def test_default_siliconflow(self):
        with patch.dict(os.environ, {"DEFAULT_LLM_PROVIDER": "siliconflow"}, clear=False):
            adapter = LLMAdapterFactory.create()
            assert isinstance(adapter, SiliconFlowAdapter)

    def test_vertex(self):
        adapter = LLMAdapterFactory.create("vertex")
        assert isinstance(adapter, VertexAIAdapter)

    def test_unknown_falls_back(self):
        adapter = LLMAdapterFactory.create("unknown_provider")
        assert isinstance(adapter, SiliconFlowAdapter)

    def test_none_uses_env(self):
        with patch.dict(os.environ, {"DEFAULT_LLM_PROVIDER": "vertex"}):
            adapter = LLMAdapterFactory.create()
            assert isinstance(adapter, VertexAIAdapter)
