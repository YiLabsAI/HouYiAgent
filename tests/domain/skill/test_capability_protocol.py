"""Cross-Host protocol verification tests (M3+).

Verifies that the CapabilityNegotiator correctly handles:
  - Full compatibility (all requirements met)
  - Degraded compatibility (warnings but still compatible)
  - Incompatible (missing critical capabilities)
  - Claude skill engines field simulation
  - Version negotiation edge cases
"""

import pytest

from houyi.domain.skill.capability import (
    CapabilityMatchResult,
    CapabilityNegotiator,
    ConsentModel,
    ExecutionForm,
    ExtensionRequirements,
    HookHandler,
    HostCapabilities,
)


@pytest.fixture
def default_host() -> HostCapabilities:
    """Default HouYi host capabilities."""
    return HostCapabilities()


@pytest.fixture
def negotiator(default_host: HostCapabilities) -> CapabilityNegotiator:
    return CapabilityNegotiator(default_host)


class TestFullCompatibility:
    """Extension requirements fully satisfied by host."""

    def test_in_process_python_skill(self, negotiator: CapabilityNegotiator):
        """A basic Python in-process skill should be fully compatible."""
        reqs = ExtensionRequirements(
            required_execution_forms=[ExecutionForm.IN_PROCESS],
            required_hook_events=["PreToolUse", "PostToolUse"],
            required_hook_handlers=[HookHandler.HANDLER],
        )
        result = negotiator.check_compatibility(reqs)
        assert result.compatible is True
        assert result.missing_capabilities == []
        assert bool(result) is True

    def test_command_hook_skill(self, negotiator: CapabilityNegotiator):
        """A skill using command hooks should be compatible."""
        reqs = ExtensionRequirements(
            required_hook_handlers=[HookHandler.COMMAND],
            required_hook_events=["SessionStart", "Stop"],
        )
        result = negotiator.check_compatibility(reqs)
        assert result.compatible is True

    def test_minimal_requirements(self, negotiator: CapabilityNegotiator):
        """A skill with no special requirements is always compatible."""
        reqs = ExtensionRequirements()
        result = negotiator.check_compatibility(reqs)
        assert result.compatible is True
        assert result.missing_capabilities == []
        assert result.warnings == []

    def test_consent_with_interactive(self, negotiator: CapabilityNegotiator):
        """Consent-requiring skill should work with interactive consent model."""
        reqs = ExtensionRequirements(requires_consent=True)
        result = negotiator.check_compatibility(reqs)
        assert result.compatible is True


class TestDegradedCompatibility:
    """Extension works but with warnings (non-blocking)."""

    def test_evaluation_preferred_not_critical(self):
        """Evaluation preference generates warning when host lacks eval, not failure."""
        host = HostCapabilities(evaluation_support=False)
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements(requires_evaluation=True)
        result = neg.check_compatibility(reqs)
        assert result.compatible is True
        assert len(result.warnings) >= 1
        assert any("evaluation" in w.lower() for w in result.warnings)

    def test_evaluation_with_no_support(self):
        """Host without evaluation generates warning for eval-requiring skill."""
        host = HostCapabilities(evaluation_support=False)
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements(requires_evaluation=True)
        result = neg.check_compatibility(reqs)
        assert result.compatible is True  # Not a hard requirement
        assert len(result.warnings) > 0


class TestIncompatible:
    """Extension cannot run on this host."""

    def test_mcp_on_bare_host(self):
        """A skill requiring MCP execution on a host without MCP support."""
        host = HostCapabilities(
            execution_forms=[ExecutionForm.IN_PROCESS],
        )
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements(
            required_execution_forms=[ExecutionForm.MCP],
        )
        result = neg.check_compatibility(reqs)
        assert result.compatible is False
        assert len(result.missing_capabilities) >= 1
        assert any("execution" in m.lower() for m in result.missing_capabilities)

    def test_consent_without_consent_host(self):
        """A consent-requiring skill on a host with consent=none."""
        host = HostCapabilities(consent_model=ConsentModel.NONE)
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements(requires_consent=True)
        result = neg.check_compatibility(reqs)
        assert result.compatible is False
        assert any("consent" in m.lower() for m in result.missing_capabilities)

    def test_missing_hook_event(self):
        """A skill requiring a custom hook event not supported by host."""
        host = HostCapabilities(hook_events=["PreToolUse", "PostToolUse"])
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements(
            required_hook_events=["CustomEvent"],
        )
        result = neg.check_compatibility(reqs)
        assert result.compatible is False
        assert any("hook events" in m.lower() for m in result.missing_capabilities)

    def test_higher_manifest_version_required(self):
        """A skill requiring a newer manifest version than host supports."""
        host = HostCapabilities(manifest_version="0.1")
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements(min_manifest_version="2.0")
        result = neg.check_compatibility(reqs)
        assert result.compatible is False
        assert any("version" in m.lower() for m in result.missing_capabilities)


class TestClaudeSkillEnginesField:
    """Simulate Claude skill ecosystem's engines field for interoperability."""

    def test_claude_code_skill_compatible(self, negotiator: CapabilityNegotiator):
        """Claude Code skills require command hooks and basic events."""
        # Simulates Claude Code's typical requirements
        reqs = ExtensionRequirements.from_dict(
            {
                "requiredHookHandlers": ["command"],
                "requiredHookEvents": ["PreToolUse", "PostToolUse", "Stop"],
            }
        )
        result = negotiator.check_compatibility(reqs)
        assert result.compatible is True

    def test_claude_mcp_without_mcp(self):
        """A Claude MCP tool requires MCP execution form."""
        host = HostCapabilities(
            execution_forms=[ExecutionForm.IN_PROCESS, ExecutionForm.SUBPROCESS],
        )
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements.from_dict(
            {
                "requiredExecutionForms": ["mcp"],
            }
        )
        result = neg.check_compatibility(reqs)
        assert result.compatible is False

    def test_claude_mcp_with_mcp(self):
        """A Claude MCP tool works when host supports MCP."""
        host = HostCapabilities(
            execution_forms=[ExecutionForm.IN_PROCESS, ExecutionForm.MCP],
        )
        neg = CapabilityNegotiator(host)
        reqs = ExtensionRequirements.from_dict(
            {
                "requiredExecutionForms": ["mcp"],
            }
        )
        result = neg.check_compatibility(reqs)
        assert result.compatible is True

    def test_mixed_forms_partial_match(self, negotiator: CapabilityNegotiator):
        """Skill requiring either in-process OR subprocess (any match suffices)."""
        reqs = ExtensionRequirements(
            required_execution_forms=[ExecutionForm.IN_PROCESS],
        )
        result = negotiator.check_compatibility(reqs)
        assert result.compatible is True


class TestHostCapabilitiesSerialization:
    def test_to_dict_round_trip(self):
        host = HostCapabilities()
        d = host.to_dict()
        restored = HostCapabilities.from_dict(d)
        assert restored.host_name == host.host_name
        assert restored.manifest_version == host.manifest_version
        assert set(restored.execution_forms) == set(host.execution_forms)

    def test_from_dict_defaults(self):
        host = HostCapabilities.from_dict({})
        assert host.host_name == "houyi"
        assert host.manifest_version == "0.1"


class TestExtensionRequirementsSerialization:
    def test_from_dict(self):
        reqs = ExtensionRequirements.from_dict(
            {
                "minManifestVersion": "1.0",
                "requiredExecutionForms": ["in-process"],
                "requiredHookEvents": ["Stop"],
                "requiresConsent": True,
            }
        )
        assert reqs.min_manifest_version == "1.0"
        assert ExecutionForm.IN_PROCESS in reqs.required_execution_forms
        assert "Stop" in reqs.required_hook_events
        assert reqs.requires_consent is True

    def test_from_dict_defaults(self):
        reqs = ExtensionRequirements.from_dict({})
        assert reqs.min_manifest_version == "0.1"
        assert reqs.required_execution_forms == []
        assert reqs.requires_consent is False


class TestCapabilityMatchResult:
    def test_bool_true_when_compatible(self):
        result = CapabilityMatchResult(compatible=True)
        assert bool(result) is True

    def test_bool_false_when_incompatible(self):
        result = CapabilityMatchResult(compatible=False, missing_capabilities=["x"])
        assert bool(result) is False
