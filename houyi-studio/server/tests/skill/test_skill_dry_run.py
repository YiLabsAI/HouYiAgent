"""Tests for DryRunValidator (schema + policy + LLM verification)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from houyi_studio.server.skill import loader as skill_loader_module
from houyi_studio.server.skill.dry_run import (
    DryRunValidator,
    _build_natural_query,
    _build_tool_definitions,
    _derive_script_compat_executor,
    _parse_llm_response,
    _simplify_schema,
)

from houyi.domain.skill.registry import SkillRegistry

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
    def __init__(
        self,
        name="s",
        tools=None,
        perms=None,
        policy=None,
        schema=None,
        description="",
        instructions=None,
        extra_frontmatter=None,
    ):
        self.name = name
        self.tools = tools or []
        self.permissions = perms
        self.invocation_policy = policy
        self.input_schema = schema
        self.description = description
        self.instructions = instructions
        self.extra_frontmatter = extra_frontmatter or {}
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
    async def test_schema_error_is_user_friendly_for_numeric_bounds(self, registry):
        from pydantic import BaseModel, Field

        class _BoundsSchema(BaseModel):
            subtask_index: int = Field(ge=0)

        registry.register(
            _Skill(name="s", tools=[_Tool("t", schema=_BoundsSchema)]),
            overwrite=True,
        )
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {"subtask_index": -1})

        assert r["valid"] is False
        assert "subtask_index: must be >= 0" in r["schema_errors"]
        assert all("For further information visit" not in err for err in r["schema_errors"])

    @pytest.mark.asyncio
    async def test_policy_deny_via_skill_policy(self, registry):
        """When InvocationPolicy is set on the skill, _check_policy reads it directly."""
        from houyi.domain.skill.policy import InvocationPolicy, ModelAutoInvoke

        ip = InvocationPolicy()
        ip.model_auto_invoke = ModelAutoInvoke.DENY
        registry.register(_Skill(name="s", policy=ip), overwrite=True)
        v = self._make_validator(registry)
        r = await v.validate("s", "t", {})
        assert r["valid"] is False
        assert r["policy_result"] == "deny"

    @pytest.mark.asyncio
    async def test_policy_allow_with_consent_via_skill_policy(self, registry):
        from houyi.domain.skill.policy import InvocationPolicy, ModelAutoInvoke

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
        from houyi.domain.skill.policy import InvocationPolicy, ModelAutoInvoke

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

    @pytest.mark.asyncio
    async def test_validate_includes_available_workflows_from_instructions(self, registry):
        instructions = (
            "```bash\n"
            "python scripts/office/unpack.py document.docx unpacked/\n"
            "python scripts/office/soffice.py --headless --convert-to docx document.doc\n"
            "```"
        )
        registry.register(_Skill(name="docx", instructions=instructions), overwrite=True)

        v = self._make_validator(registry)
        r = await v.validate("docx", "docx", {}, live=False)

        workflows = r.get("available_workflows", [])
        assert len(workflows) == 2
        assert workflows[0]["source"] == "instructions"
        assert workflows[0]["command"].startswith("python ")
        assert workflows[0]["evidence"].startswith("python ")
        assert workflows[0]["confidence"] in {"medium", "high"}
        assert isinstance(workflows[0]["confidence_score"], float)
        assert workflows[0]["validation"]["status"] in {"pass", "warn"}
        assert "missing_dependencies" in workflows[0]["validation"]

    @pytest.mark.asyncio
    async def test_validate_prefers_frontmatter_workflows_over_instruction_templates(
        self, registry
    ):
        instructions = "```bash\npython scripts/office/unpack.py document.docx unpacked/\n```"
        extra_frontmatter = {
            "workflows": [
                {
                    "id": "read_docx",
                    "title": "Read docx",
                    "command": "python scripts/office/unpack.py document.docx unpacked/",
                    "params": ["input_path", "output_dir"],
                    "depends_on": ["pandoc"],
                }
            ]
        }
        registry.register(
            _Skill(name="docx", instructions=instructions, extra_frontmatter=extra_frontmatter),
            overwrite=True,
        )

        v = self._make_validator(registry)
        r = await v.validate("docx", "docx", {}, live=False)

        workflows = r.get("available_workflows", [])
        assert len(workflows) == 1
        assert workflows[0]["id"] == "read_docx"
        assert workflows[0]["source"] == "frontmatter"
        assert workflows[0]["depends_on"] == ["pandoc"]
        assert workflows[0]["evidence"] == "frontmatter.workflows[0]"
        assert workflows[0]["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_validate_filters_non_executable_noise_from_instructions(self, registry):
        instructions = (
            "```bash\n"
            "# this is just commentary\n"
            "python\n"
            "python scripts/run.py auth_manager.py status\n"
            "python scripts/run.py auth_manager.py status\n"
            "```"
        )
        registry.register(_Skill(name="notebooklm", instructions=instructions), overwrite=True)

        v = self._make_validator(registry)
        r = await v.validate("notebooklm", "notebooklm", {}, live=False)

        workflows = r.get("available_workflows", [])
        assert len(workflows) == 1
        assert workflows[0]["command"] == "python scripts/run.py auth_manager.py status"

    @pytest.mark.asyncio
    async def test_derived_script_compat_executor_falls_back_when_primary_template_missing_dependency(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            skill_dir = Path(td)
            office_dir = skill_dir / "scripts" / "office"
            office_dir.mkdir(parents=True)

            (office_dir / "soffice.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
            (office_dir / "unpack.py").write_text(
                "import json\nprint(json.dumps({'mode':'unpack'}))\n",
                encoding="utf-8",
            )

            instructions = (
                "```bash\n"
                "python scripts/office/soffice.py --headless --convert-to docx document.doc\n"
                "python scripts/office/unpack.py document.docx unpacked/\n"
                "```"
            )
            skill = _Skill(name="docx", instructions=instructions)
            skill.skill_dir = str(skill_dir)

            executor = _derive_script_compat_executor(skill)
            assert callable(executor)

            def _which(binary: str):
                if binary == "soffice":
                    return None
                return "/usr/bin/" + binary

            with patch.object(skill_loader_module.shutil, "which", side_effect=_which):
                result = await executor(convert_to="docx")

            assert result.get("success") is True
            command = result.get("command", [])
            assert any(str(token).endswith("scripts/office/unpack.py") for token in command)


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

    def test_preserves_empty_dict_arguments_in_dict_style(self):
        """Empty dict arguments are preserved as-is."""
        resp = MagicMock()
        resp.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {}},
            }
        ]
        r = _parse_llm_response(resp, "get_weather")
        assert r["success"] is True
        assert r["tool_call"]["arguments"] == {}

    def test_preserves_empty_dict_arguments_in_object_style(self):
        """Object-style tool call with {} args remains {}."""

        class _TC:
            name = "get_weather"
            arguments = {}

        class _Resp:
            tool_calls = [_TC()]

        r = _parse_llm_response(_Resp(), "get_weather")
        assert r["success"] is True
        assert r["tool_call"]["arguments"] == {}

    def test_parses_fenced_json_arguments_and_extracts_action(self):
        resp = MagicMock()
        resp.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "ext__planning-with-files",
                    "arguments": '```json\n{"action": "create"}\n```',
                },
            }
        ]
        r = _parse_llm_response(resp, "ext__planning-with-files")
        assert r["success"] is True
        assert r["tool_call"]["arguments"] == {"action": "create"}
        assert r["tool_call"]["action"] == "create"

    def test_parses_python_style_arguments_and_extracts_nested_action(self):
        resp = MagicMock()
        resp.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "ext__planning-with-files",
                    "arguments": "{'kwargs': {'Action': 'create'}}",
                },
            }
        ]
        r = _parse_llm_response(resp, "ext__planning-with-files")
        assert r["success"] is True
        assert r["tool_call"]["arguments"] == {"kwargs": {"Action": "create"}}
        assert r["tool_call"]["action"] == "create"

    def test_recovers_arguments_from_raw_content_when_tool_call_args_empty(self):
        resp = MagicMock()
        resp.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "ext__planning-with-files",
                    "arguments": {},
                },
            }
        ]
        resp.content = (
            '{"tool_calls":[{"function":{"name":"ext__planning-with-files",'
            '"arguments":{"action":"create","task":"demo"}}}]}'
        )

        r = _parse_llm_response(resp, "ext__planning-with-files")
        assert r["success"] is True
        assert r["tool_call"]["arguments"] == {"action": "create", "task": "demo"}
        assert r["tool_call"]["action"] == "create"

    def test_recovers_tool_call_from_raw_content_when_tool_calls_missing(self):
        class _Resp:
            tool_calls = []
            content = (
                '{"message":{"tool_call":{"name":"frontend-design","arguments":{"brief":"bold"}}}}'
            )

        r = _parse_llm_response(_Resp(), "frontend-design")
        assert r["success"] is True
        assert r["tool_call"]["name"] == "frontend-design"
        assert r["tool_call"]["arguments"] == {"brief": "bold"}


class TestBuildNaturalQuery:
    def test_keeps_original_query_text_in_prompt_context(self):
        """Dry-run probe keeps the original user query text."""
        original_query = "published articles on infoq in the last year"
        prompt = _build_natural_query(
            "web_search",
            "web_search",
            {"query": original_query, "provider": "ddg"},
        )
        assert original_query in prompt


# ── _live_verify integration ────────────────────────────────────────


_FACTORY_PATCH = "houyi.adapters.llm.LLMAdapterFactory"


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
        assert result["requested_input"] == {"query": "hi"}

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
        with patch.dict("sys.modules", {"houyi.adapters.llm": None}):
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
        assert result["requested_input"] == {}

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

    @pytest.mark.asyncio
    async def test_tool_execution_falls_back_to_requested_input_when_observed_args_missing(self):
        """If provider omits arguments, live dry-run should still replay with requested_input."""
        from houyi_studio.server.skill.dry_run import _live_verify

        async def _executor(**kwargs):
            return {"success": True, "received": kwargs}

        skill = _Skill(name="rag-skill", description="RAG")
        skill.executor = _executor

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "type": "function",
                "function": {"name": "rag-skill", "arguments": None},
            }
        ]

        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(
                skill,
                "rag-skill",
                "rag-skill",
                {"query": "what is rag", "knowledge_dir": "knowledge/"},
            )

        assert result["success"] is True
        assert result["tool_call"]["arguments_source"] == "requested_input_fallback"
        exec_phase = next(p for p in result["phases"] if p["name"] == "tool_execution")
        assert exec_phase["status"] == "pass"
        assert exec_phase["data"]["argument_source"] == "requested_input_fallback"

    @pytest.mark.asyncio
    async def test_tool_execution_falls_back_when_observed_args_are_empty_dict(self):
        """If provider returns empty arguments object, use requested_input as fallback evidence."""
        from houyi_studio.server.skill.dry_run import _live_verify

        async def _executor(**kwargs):
            return {"success": True, "received": kwargs}

        skill = _Skill(name="notebooklm", description="NotebookLM")
        skill.executor = _executor

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "type": "function",
                "function": {"name": "notebooklm", "arguments": "{}"},
            }
        ]

        mock_adapter = AsyncMock()
        mock_adapter.chat.return_value = mock_response

        requested_input = {
            "question": "Summarize architecture decisions",
            "notebook_url": "https://notebooklm.google.com/notebook/example",
        }

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(
                skill,
                "notebooklm",
                "notebooklm",
                requested_input,
            )

        assert result["success"] is True
        assert result["tool_call"]["arguments_source"] == "requested_input_fallback"
        assert result["tool_call"]["arguments"] == requested_input
        exec_phase = next(p for p in result["phases"] if p["name"] == "tool_execution")
        assert exec_phase["status"] == "pass"
        assert exec_phase["data"]["argument_source"] == "requested_input_fallback"

    @pytest.mark.asyncio
    async def test_live_verify_adds_final_response_after_tool_execution(self):
        from houyi_studio.server.skill.dry_run import _live_verify

        async def _executor(**kwargs):
            return {"success": True, "received": kwargs}

        skill = _Skill(name="rag-skill", description="RAG")
        skill.executor = _executor

        first = MagicMock()
        first.tool_calls = [
            {
                "type": "function",
                "function": {"name": "rag-skill", "arguments": '{"query":"what is rag"}'},
            }
        ]

        second = MagicMock()
        second.content = "RAG is retrieval-augmented generation."

        mock_adapter = AsyncMock()
        mock_adapter.chat.side_effect = [first, second]

        with patch(_FACTORY_PATCH) as factory:
            factory.create.return_value = mock_adapter
            result = await _live_verify(
                skill,
                "rag-skill",
                "rag-skill",
                {"query": "what is rag"},
            )

        final_phase = next(p for p in result["phases"] if p["name"] == "final_response")
        assert final_phase["status"] == "pass"
        assert "RAG is retrieval-augmented generation" in result.get("final_answer", "")

    @pytest.mark.asyncio
    async def test_planning_skill_live_verify_executes_bound_executor(self):
        """Planning skill should execute in live dry-run (not skip with missing executor)."""
        from houyi_studio.server.skill.dry_run import _live_verify

        from houyi.skills.planning import PlanningSkill

        with tempfile.TemporaryDirectory() as tmpdir:
            skill = PlanningSkill(workspace=Path(tmpdir)).to_spec()
            mock_response = MagicMock()
            mock_response.tool_calls = [
                {
                    "type": "function",
                    "function": {
                        "name": "planning-with-files",
                        "arguments": '{"action":"status"}',
                    },
                }
            ]
            mock_adapter = AsyncMock()
            mock_adapter.chat.return_value = mock_response

            with patch(_FACTORY_PATCH) as factory:
                factory.create.return_value = mock_adapter
                result = await _live_verify(
                    skill,
                    "planning-with-files",
                    "planning-with-files",
                    {"action": "status"},
                )

        assert result["success"] is True
        exec_phase = next(p for p in result["phases"] if p["name"] == "tool_execution")
        assert exec_phase["status"] != "skip"
        assert exec_phase["data"].get("reason") != "no executor available"

    @pytest.mark.asyncio
    async def test_planning_create_without_task_returns_validation_failure(self):
        """Planning create without task should fail gracefully without TypeError."""
        from houyi_studio.server.skill.dry_run import _live_verify

        from houyi.skills.planning import PlanningSkill

        with tempfile.TemporaryDirectory() as tmpdir:
            skill = PlanningSkill(workspace=Path(tmpdir)).to_spec()
            mock_response = MagicMock()
            mock_response.tool_calls = [
                {
                    "type": "function",
                    "function": {
                        "name": "planning-with-files",
                        "arguments": '{"action":"create"}',
                    },
                }
            ]
            mock_adapter = AsyncMock()
            mock_adapter.chat.return_value = mock_response

            with patch(_FACTORY_PATCH) as factory:
                factory.create.return_value = mock_adapter
                result = await _live_verify(
                    skill,
                    "planning-with-files",
                    "planning-with-files",
                    {"action": "create"},
                )

        exec_phase = next(p for p in result["phases"] if p["name"] == "tool_execution")
        assert exec_phase["status"] == "fail"
        preview = exec_phase["data"].get("result_preview")
        assert isinstance(preview, dict)
        assert preview.get("success") is False
        assert "Missing required field for create: task" in str(preview.get("message", ""))

    @pytest.mark.asyncio
    async def test_live_verify_derives_script_executor_from_instructions(self):
        """Instruction-driven script skills should execute in dry-run even without bound executor."""
        from houyi_studio.server.skill.dry_run import _live_verify

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "run_demo.py"
            script_path.write_text(
                "import json\nprint(json.dumps({'success': True, 'mode': 'script-compat'}))\n",
                encoding="utf-8",
            )

            skill = _Skill(name="script-skill", description="Script skill")
            skill.skill_dir = tmp_path
            skill.instructions = """
## Workflow Steps
```bash
python run_demo.py --action status
```
"""

            mock_response = MagicMock()
            mock_response.tool_calls = [
                {
                    "type": "function",
                    "function": {"name": "script-skill", "arguments": "{}"},
                }
            ]
            mock_adapter = AsyncMock()
            mock_adapter.chat.return_value = mock_response

            with patch(_FACTORY_PATCH) as factory:
                factory.create.return_value = mock_adapter
                result = await _live_verify(
                    skill,
                    "script-skill",
                    "script-skill",
                    {"action": "status"},
                )

        exec_phase = next(p for p in result["phases"] if p["name"] == "tool_execution")
        assert exec_phase["status"] != "skip"
        assert exec_phase["data"].get("argument_source") == "requested_input_fallback"


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
