"""Tests for DryRunValidator (schema + policy + LLM verification)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from houyi_studio.server.skill.dry_run import (
    DryRunValidator,
    _build_tool_definitions,
    _parse_llm_response,
    _simplify_schema,
)

from houyi.core.skill_registry import SkillRegistry

# ── Fakes ─────────────────────────────────────────────────────────────


class _Schema:
    def __init__(self, required=None):
        self._req = set(required or [])

    def model_validate(self, data):
        for f in self._req:
            if f not in data:
                raise ValueError(f"Missing: {f}")

    def model_json_schema(self):
        return {"type": "object", "required": list(self._req)}


class _Tool:
    def __init__(self, name, schema=None, description=""):
        self.name = name
        self.input_schema = schema
        self.description = description


class _Skill:
    def __init__(self, name="s", tools=None, perms=None, policy=None, schema=None, description=""):
        self.name = name
        self.tools = tools or []
        self.permissions = perms
        self.invocation_policy = policy
        self.input_schema = schema
        self.description = description
        self.provider = ""
        self.qualified_name = name
        self.executor = None


class _PermKind:
    def __init__(self, enabled=False, write=False, delete=False):
        self.enabled = enabled
        self.write = write
        self.delete = delete


class _Perms:
    def __init__(self, exec_=None, network=None, filesystem=None):
        self.exec = exec_ or _PermKind()
        self.network = network or _PermKind()
        self.filesystem = filesystem or _PermKind()


class _PolicyResult:
    def __init__(self, val="allow"):
        self.action = MagicMock(value=val)


# ── DryRunValidator.validate ──────────────────────────────────────────


class TestValidate:
    @pytest.fixture
    def registry(self):
        reg = SkillRegistry()
        return reg

    def _make_validator(self, reg, enforcer=None):
        return DryRunValidator(reg, policy_enforcer=enforcer)

    @pytest.mark.asyncio
    async def test_skill_not_found(self, registry):
        v = self._make_validator(registry)
        r = await v.validate("nope", "t", {})
        assert r["valid"] is False
        assert "not found" in r["schema_errors"][0]

    @pytest.mark.asyncio
    async def test_valid_empty_input(self, registry):
        registry.register(_Skill(name="s", tools=[_Tool("t")]), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {})
        assert r["valid"] is True

    @pytest.mark.asyncio
    async def test_schema_error(self, registry):
        schema = _Schema(required=["query"])
        registry.register(
            _Skill(name="s", tools=[_Tool("t", schema=schema)]),
            overwrite=True,
        )
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {"wrong": "x"})
        assert r["valid"] is False
        assert len(r["schema_errors"]) > 0

    @pytest.mark.asyncio
    async def test_schema_pass(self, registry):
        schema = _Schema(required=["query"])
        registry.register(
            _Skill(name="s", tools=[_Tool("t", schema=schema)]),
            overwrite=True,
        )
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {"query": "hello"})
        assert r["valid"] is True

    @pytest.mark.asyncio
    async def test_policy_deny_via_skill_policy(self, registry):
        """When InvocationPolicy is set on the skill, _check_policy reads it directly."""
        from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

        ip = InvocationPolicy()
        ip.model_auto_invoke = ModelAutoInvoke.DENY
        registry.register(_Skill(name="s", policy=ip), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {})
        assert r["valid"] is False
        assert r["policy_result"] == "deny"

    @pytest.mark.asyncio
    async def test_policy_allow_with_consent_via_skill_policy(self, registry):
        from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

        ip = InvocationPolicy()
        ip.model_auto_invoke = ModelAutoInvoke.ALLOW_WITH_CONSENT
        registry.register(_Skill(name="s", policy=ip), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {})
        assert r["valid"] is True
        assert r["policy_result"] == "allow_with_consent"

    @pytest.mark.asyncio
    async def test_policy_deny_blocks_live_execution(self, registry):
        """When policy is deny, live=True should NOT call the LLM."""
        from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

        ip = InvocationPolicy()
        ip.model_auto_invoke = ModelAutoInvoke.DENY
        registry.register(_Skill(name="s", policy=ip), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {}, live=True)
        assert r["valid"] is False
        assert r["policy_result"] == "deny"
        assert r["llm_verification"]["success"] is False
        assert "static validation failed" in r["llm_verification"]["message"]
        assert r["llm_verification"]["phases"] == []

    @pytest.mark.asyncio
    async def test_policy_deny_via_enforcer_fallback(self, registry):
        """When no InvocationPolicy on skill, falls back to PolicyEnforcer."""
        registry.register(_Skill(name="s"), overwrite=True)
        enforcer = MagicMock()
        enforcer.check_invocation.return_value = MagicMock(allowed=False)
        v = self._make_validator(registry, enforcer=enforcer)
        r = await v.validate("s", "t", {})
        assert r["valid"] is False
        assert r["policy_result"] == "deny"

    @pytest.mark.asyncio
    async def test_side_effects_detected(self, registry):
        perms = _Perms(network=_PermKind(enabled=True))
        registry.register(_Skill(name="s", perms=perms), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {})
        assert "network" in r["estimated_side_effects"]

    @pytest.mark.asyncio
    async def test_live_false_no_llm(self, registry):
        registry.register(_Skill(name="s"), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {}, live=False)
        assert "llm_verification" not in r

    @pytest.mark.asyncio
    async def test_live_true_includes_verification(self, registry):
        registry.register(_Skill(name="s", tools=[_Tool("t")]), overwrite=True)
        v = self._make_validator(registry)
        with patch(
            "houyi_studio.server.skill.dry_run._live_verify",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = {"success": True, "message": "ok"}
            r = await v.validate("s", "t", {}, live=True)
        assert r["llm_verification"]["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_to_skill_schema(self, registry):
        """When tool has no schema, fall back to skill-level input_schema."""
        skill_schema = _Schema(required=["q"])
        registry.register(
            _Skill(name="s", tools=[_Tool("t")], schema=skill_schema),
            overwrite=True,
        )
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {"wrong": 1})
        assert r["valid"] is False


# ── _build_tool_definitions ───────────────────────────────────────────


class TestBuildToolDefs:
    def test_no_tools_uses_skill_itself(self):
        """When skill has no .tools, build def from the skill's own schema."""
        schema = _Schema(required=["q"])
        skill = _Skill(name="calc", schema=schema, description="A calculator")
        defs = _build_tool_definitions(skill)
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "calc"
        assert defs[0]["function"]["description"] == "A calculator"
        assert "required" in defs[0]["function"]["parameters"]

    def test_no_tools_no_schema(self):
        """Skill with no tools and no schema still produces a definition."""
        skill = _Skill(name="ping")
        defs = _build_tool_definitions(skill)
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "ping"
        assert defs[0]["function"]["parameters"] == {}

    def test_with_tools(self):
        schema = _Schema(required=["a"])
        s = _Skill(tools=[_Tool("fn", schema=schema, description="d")])
        defs = _build_tool_definitions(s)
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "fn"
        assert defs[0]["function"]["description"] == "d"
        assert "required" in defs[0]["function"]["parameters"]


# ── _simplify_schema ─────────────────────────────────────────────────


class TestSimplifySchema:
    """Regression tests for stripping Pydantic metadata from JSON Schema."""

    def test_removes_top_level_title(self):
        schema = {"title": "WebSearchInput", "type": "object", "properties": {}}
        assert "title" not in _simplify_schema(schema)
        assert _simplify_schema(schema)["type"] == "object"

    def test_removes_property_titles(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"title": "Query", "type": "string"},
            },
        }
        result = _simplify_schema(schema)
        assert "title" not in result["properties"]["query"]
        assert result["properties"]["query"]["type"] == "string"

    def test_flattens_anyof_nullable(self):
        """anyOf: [{type: string}, {type: null}] -> type: string"""
        schema = {
            "type": "object",
            "properties": {
                "provider": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Provider",
                },
            },
        }
        result = _simplify_schema(schema)
        prop = result["properties"]["provider"]
        assert "anyOf" not in prop
        assert "title" not in prop
        assert prop["type"] == "string"
        assert prop["default"] is None

    def test_preserves_required_and_other_keys(self):
        schema = {
            "title": "Input",
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"title": "Q", "type": "string"}},
        }
        result = _simplify_schema(schema)
        assert result["required"] == ["query"]
        assert result["type"] == "object"

    def test_real_pydantic_schema(self):
        """Simulate the exact schema Pydantic generates for WebSearchInput."""
        pydantic_schema = {
            "title": "WebSearchInput",
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"title": "Query", "type": "string"},
                "max_results": {
                    "default": 3,
                    "minimum": 1,
                    "title": "Max Results",
                    "type": "integer",
                },
                "provider": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Provider",
                },
            },
        }
        result = _simplify_schema(pydantic_schema)

        assert "title" not in result
        assert result["required"] == ["query"]
        assert result["properties"]["query"] == {"type": "string"}
        assert result["properties"]["max_results"] == {
            "default": 3,
            "minimum": 1,
            "type": "integer",
        }
        assert result["properties"]["provider"] == {"default": None, "type": "string"}


class TestBuildToolDefsSchemaCleanup:
    """Verify _build_tool_definitions applies _simplify_schema."""

    def test_pydantic_titles_stripped(self):
        class _PydanticSchema:
            @staticmethod
            def model_json_schema():
                return {
                    "title": "MyInput",
                    "type": "object",
                    "properties": {
                        "q": {"title": "Q", "type": "string"},
                        "opt": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Opt",
                        },
                    },
                }

        skill = _Skill(name="s", schema=_PydanticSchema, description="d")
        defs = _build_tool_definitions(skill)
        params = defs[0]["function"]["parameters"]
        assert "title" not in params
        assert "title" not in params["properties"]["q"]
        assert params["properties"]["opt"]["type"] == "string"
        assert "anyOf" not in params["properties"]["opt"]


# ── _parse_llm_response ──────────────────────────────────────────────


class TestParseLlmResponse:
    def test_matched_object_style(self):
        """Handles raw OpenAI-style response objects."""

        class _TC:
            name = "fn"
            arguments = '{"q":"hi"}'

        class _Resp:
            tool_calls = [_TC()]

        r = _parse_llm_response(_Resp(), "fn")
        assert r["success"] is True
        assert r["tool_call"]["name"] == "fn"

    def test_matched_dict_style(self):
        """Handles dict-format tool_calls from LLMResponse."""
        resp = MagicMock()
        resp.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": '{"q":"hi"}'},
            }
        ]
        r = _parse_llm_response(resp, "fn")
        assert r["success"] is True
        assert r["tool_call"]["name"] == "fn"
        assert r["tool_call"]["arguments"] == {"q": "hi"}

    def test_mismatch(self):
        class _TC:
            name = "other"
            arguments = "{}"

        class _Resp:
            tool_calls = [_TC()]

        r = _parse_llm_response(_Resp(), "fn")
        assert r["success"] is False

    def test_no_tool_call(self):
        class _Resp:
            tool_calls = None
            content = "I cannot call tools"

        r = _parse_llm_response(_Resp(), "fn")
        assert r["success"] is False
        assert "did not produce" in r["message"]


# ── _live_verify integration ────────────────────────────────────────


_FACTORY_PATCH = "houyi.llm.factory.LLMAdapterFactory"


class TestLiveVerifyIntegration:
    """Tests for _live_verify with mocked LLM adapter across skill types."""

    @pytest.mark.asyncio
    async def test_single_tool_skill_with_schema(self):
        """Guidance/single-tool skill with input_schema can do live verify."""
        from houyi_studio.server.skill.dry_run import _live_verify

        schema = _Schema(required=["query"])
        skill = _Skill(name="web_search", schema=schema, description="Search")
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "web_search", "arguments": '{"query":"hi"}'}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "web_search", "web_search", {"query": "hi"})

        assert result["success"] is True
        assert "correctly called" in result["message"]

    @pytest.mark.asyncio
    async def test_single_tool_skill_no_schema(self):
        """Guidance skill without schema still generates tool definition."""
        from houyi_studio.server.skill.dry_run import _live_verify

        skill = _Skill(name="planning-with-files", description="Plan")
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "planning-with-files", "arguments": "{}"}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "planning-with-files", "planning-with-files", {})

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_multi_tool_skill(self):
        """Skill with .tools attribute uses tool list for definitions."""
        from houyi_studio.server.skill.dry_run import _live_verify

        schema = _Schema(required=["lat", "lon"])
        skill = _Skill(
            name="weather",
            tools=[
                _Tool("get_weather", schema=schema, description="Get weather"),
                _Tool("get_date", description="Get current date"),
            ],
        )
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"lat":30,"lon":120}'},
            }
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "weather", "get_weather", {"lat": 30, "lon": 120})

        assert result["success"] is True
        call_kwargs = mock_adapter.chat.call_args[1]
        assert len(call_kwargs["tools"]) == 2

    @pytest.mark.asyncio
    async def test_llm_adapter_not_installed(self):
        """Graceful failure when LLM adapter cannot be imported."""
        from houyi_studio.server.skill.dry_run import _live_verify

        skill = _Skill(name="test")
        with patch.dict("sys.modules", {"houyi.llm.factory": None}):
            result = await _live_verify(skill, "test", "test", {})
        assert result["success"] is False
        assert "not available" in result["message"]

    @pytest.mark.asyncio
    async def test_llm_call_exception(self):
        """Graceful failure when LLM call raises an exception."""
        from houyi_studio.server.skill.dry_run import _live_verify

        skill = _Skill(name="test", description="Test skill")
        mock_adapter = AsyncMock()
        mock_adapter.chat.side_effect = RuntimeError("API rate limit")

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "test", "test", {})

        assert result["success"] is False
        assert "API rate limit" in result["message"]

    @pytest.mark.asyncio
    async def test_llm_wrong_tool_called(self):
        """LLM calls a different tool than expected."""
        from houyi_studio.server.skill.dry_run import _live_verify

        skill = _Skill(name="calc", description="Calculator")
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "wrong_tool", "arguments": "{}"}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "calc", "calc", {})

        assert result["success"] is False
        assert "wrong_tool" in result["message"]

    @pytest.mark.asyncio
    async def test_llm_no_tool_call_in_response(self):
        """LLM returns text instead of a tool call."""
        from houyi_studio.server.skill.dry_run import _live_verify

        skill = _Skill(name="calc", description="Calculator")
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "I don't know how to use tools"

        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "calc", "calc", {})

        assert result["success"] is False
        assert "did not produce" in result["message"]

    @pytest.mark.asyncio
    async def test_tool_execution_preview_is_dict_not_string(self):
        """result_preview must be a dict to avoid double-escaping.

        If it were a pre-serialised JSON string, the outer WebSocket JSON
        serialisation would double-escape quotes and newlines — exactly the
        bug this regression test guards against.
        """
        from houyi_studio.server.skill.dry_run import _live_verify

        async def _executor(**kwargs):
            return {"title": "Test Title", "snippet": "line one\nline two"}

        skill = _Skill(name="search", description="Search")
        skill.executor = _executor
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "search", "arguments": '{"q":"hi"}'}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(skill, "search", "search", {"q": "hi"})

        assert result["success"] is True
        exec_phase = next(p for p in result["phases"] if p["name"] == "tool_execution")
        preview = exec_phase["data"]["result_preview"]
        assert isinstance(preview, dict), (
            "result_preview must be a dict to avoid double-escaping in JSON responses"
        )
        assert preview["title"] == "Test Title"
        assert preview["snippet"] == "line one\nline two"


# ── Full async validate + live ────────────────────────────────────────


class TestValidateWithLive:
    """End-to-end validate() tests including live LLM mode."""

    @pytest.fixture
    def registry(self):
        return SkillRegistry()

    @pytest.mark.asyncio
    async def test_guidance_skill_live_pass(self, registry):
        """Guidance skill (no tools, no schema) passes live verify."""
        skill = _Skill(name="planning", description="Plan")
        registry.register(skill, overwrite=True)
        validator = DryRunValidator(registry)

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "planning", "arguments": "{}"}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await validator.validate("planning", "planning", {}, live=True)

        assert result["valid"] is True
        assert result["llm_verification"]["success"] is True

    @pytest.mark.asyncio
    async def test_schema_skill_live_pass(self, registry):
        """Skill with schema passes both static and live validation."""
        schema = _Schema(required=["query"])
        skill = _Skill(name="search", schema=schema, description="Search")
        registry.register(skill, overwrite=True)
        validator = DryRunValidator(registry)

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "search", "arguments": '{"query":"test"}'}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await validator.validate("search", "search", {"query": "test"}, live=True)

        assert result["valid"] is True
        assert result["schema_errors"] == []
        assert result["llm_verification"]["success"] is True

    @pytest.mark.asyncio
    async def test_schema_fail_but_live_still_runs(self, registry):
        """Schema validation fails but live mode still runs."""
        schema = _Schema(required=["query"])
        skill = _Skill(name="search", schema=schema, description="Search")
        registry.register(skill, overwrite=True)
        validator = DryRunValidator(registry)

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"type": "function", "function": {"name": "search", "arguments": '{"query":"test"}'}}
        ]
        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await validator.validate("search", "search", {"wrong": "data"}, live=True)

        assert result["valid"] is False
        assert len(result["schema_errors"]) > 0
        assert "llm_verification" in result
