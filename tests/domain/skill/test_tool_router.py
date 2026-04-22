"""Tests for the Tool Router (M9).

Covers:
  - ToolRouter initialization from skills
  - filter_tools whitelist enforcement
  - check() two-level routing: allowed-tools + InvocationPolicy
  - check_batch() multi-tool routing
  - Edge cases: no restrictions, empty skills, unknown tools
"""

from unittest.mock import MagicMock

from houyi.domain.skill.policy import (
    InvocationPolicy,
    ModelAutoInvoke,
    PolicyEnforcer,
)
from houyi.domain.skill.tool_router import ToolRouter


def _make_skill(name: str, allowed_tools: list[str] | None = None):
    """Create a minimal SkillSpec-like object for testing."""
    skill = MagicMock()
    skill.name = name
    skill.allowed_tools = allowed_tools or []
    return skill


def _make_tool(name: str) -> dict:
    """Create an OpenAI-style tool definition."""
    return {"type": "function", "function": {"name": name, "parameters": {}}}


class TestToolRouterInit:
    def test_no_skills(self):
        router = ToolRouter(skills=[])
        assert not router.has_restrictions

    def test_unrestricted_skill(self):
        router = ToolRouter(skills=[_make_skill("calculator")])
        assert not router.has_restrictions

    def test_restricted_skill(self):
        router = ToolRouter(skills=[_make_skill("web_search", allowed_tools=["search", "fetch"])])
        assert router.has_restrictions

    def test_mixed_skills(self):
        router = ToolRouter(
            skills=[
                _make_skill("web_search", allowed_tools=["search"]),
                _make_skill("calculator"),  # unrestricted
            ]
        )
        assert router.has_restrictions


class TestFilterTools:
    def test_no_restrictions_passes_all(self):
        router = ToolRouter(skills=[_make_skill("calc")])
        tools = [_make_tool("add"), _make_tool("multiply")]
        filtered = router.filter_tools(tools)
        assert len(filtered) == 2

    def test_whitelist_filters(self):
        router = ToolRouter(skills=[_make_skill("ws", allowed_tools=["search"])])
        tools = [_make_tool("search"), _make_tool("delete"), _make_tool("execute")]
        filtered = router.filter_tools(tools)
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "search"

    def test_union_of_whitelists(self):
        router = ToolRouter(
            skills=[
                _make_skill("s1", allowed_tools=["tool_a", "tool_b"]),
                _make_skill("s2", allowed_tools=["tool_b", "tool_c"]),
            ]
        )
        tools = [
            _make_tool("tool_a"),
            _make_tool("tool_b"),
            _make_tool("tool_c"),
            _make_tool("tool_d"),
        ]
        filtered = router.filter_tools(tools)
        names = {t["function"]["name"] for t in filtered}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_unrestricted_allows_mixed(self):
        router = ToolRouter(
            skills=[
                _make_skill("restricted", allowed_tools=["allowed_tool"]),
                _make_skill("unrestricted"),
            ]
        )
        tools = [_make_tool("allowed_tool"), _make_tool("other_tool")]
        filtered = router.filter_tools(tools)
        assert len(filtered) == 2

    def test_empty_tools_list(self):
        router = ToolRouter(skills=[_make_skill("s1", allowed_tools=["x"])])
        assert router.filter_tools([]) == []

    def test_without_name_passes_through(self):
        router = ToolRouter(skills=[_make_skill("s1", allowed_tools=["x"])])
        # A tool dict with no extractable name should pass through for debugging
        tools = [{"type": "no_function_key"}, _make_tool("x")]
        filtered = router.filter_tools(tools)
        # no_function_key tool has no name (passes), x also passes
        assert len(filtered) == 2

    def test_flat_format_tool_filtered(self):
        router = ToolRouter(skills=[_make_skill("s1", allowed_tools=["x"])])
        # Flat-format tool {"name": "..."} is recognized and filtered
        tools = [{"name": "not_allowed"}, _make_tool("x")]
        filtered = router.filter_tools(tools)
        # not_allowed is filtered out, x passes
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "x"


class TestCheck:
    def test_allowed_tool(self):
        router = ToolRouter(skills=[_make_skill("s1", allowed_tools=["tool_a"])])
        result = router.check("tool_a")
        assert result.allowed is True
        assert result.matched_skill == "s1"

    def test_denied_by_whitelist(self):
        router = ToolRouter(skills=[_make_skill("s1", allowed_tools=["tool_a"])])
        result = router.check("tool_b")
        assert result.allowed is False
        assert "not in any skill" in result.reason

    def test_no_restrictions_allows_all(self):
        router = ToolRouter(skills=[_make_skill("calc")])
        result = router.check("any_tool")
        assert result.allowed is True

    def test_policy_deny(self):
        enforcer = PolicyEnforcer(
            default_policy=InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY)
        )
        router = ToolRouter(
            skills=[_make_skill("s1", allowed_tools=["tool_a"])],
            policy_enforcer=enforcer,
        )
        # Register the skill policy
        enforcer.register_skill_policy(
            "s1",
            InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY),
        )
        result = router.check("tool_a", is_model_initiated=True)
        assert result.allowed is False
        assert result.matched_skill == "s1"

    def test_allow_consent_without_consent(self):
        enforcer = PolicyEnforcer()
        enforcer.register_skill_policy(
            "s1",
            InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT),
        )
        router = ToolRouter(
            skills=[_make_skill("s1", allowed_tools=["tool_a"])],
            policy_enforcer=enforcer,
        )
        result = router.check("tool_a", is_model_initiated=True, user_consent_given=False)
        assert result.allowed is False
        assert result.requires_consent is True

    def test_allow_consent_with_given(self):
        enforcer = PolicyEnforcer()
        enforcer.register_skill_policy(
            "s1",
            InvocationPolicy(model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT),
        )
        router = ToolRouter(
            skills=[_make_skill("s1", allowed_tools=["tool_a"])],
            policy_enforcer=enforcer,
        )
        result = router.check("tool_a", is_model_initiated=True, user_consent_given=True)
        assert result.allowed is True

    def test_user_initiated_bypasses_policy(self):
        enforcer = PolicyEnforcer()
        enforcer.register_skill_policy(
            "s1",
            InvocationPolicy(model_auto_invoke=ModelAutoInvoke.DENY),
        )
        router = ToolRouter(
            skills=[_make_skill("s1", allowed_tools=["tool_a"])],
            policy_enforcer=enforcer,
        )
        result = router.check("tool_a", is_model_initiated=False)
        assert result.allowed is True


class TestCheckBatch:
    def test_batch_routing(self):
        router = ToolRouter(skills=[_make_skill("s1", allowed_tools=["a", "b"])])
        results = router.check_batch(["a", "b", "c"])
        assert results["a"].allowed is True
        assert results["b"].allowed is True
        assert results["c"].allowed is False
