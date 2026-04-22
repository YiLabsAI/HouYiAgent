"""Tests for SimpleSkill capability negotiation.

Reference: SimpleSkill Specification 0.1.0 Section 5.1 (Capability Negotiation)
"""

from houyi.domain.skill.capability import (
    DEFAULT_HOUYI_CAPABILITIES,
    CapabilityMatchResult,
    CapabilityNegotiator,
    ConsentModel,
    ExecutionForm,
    ExtensionRequirements,
    HookHandler,
    HostCapabilities,
    ManifestFormat,
    ObservabilityFeature,
)


class TestHostCapabilities:
    """Test HostCapabilities dataclass."""

    def test_default_capabilities(self):
        caps = HostCapabilities()
        assert ManifestFormat.SIMPLESKILL_JSON in caps.manifest_formats
        assert ManifestFormat.SKILL_MD in caps.manifest_formats
        assert ExecutionForm.IN_PROCESS in caps.execution_forms
        assert HookHandler.COMMAND in caps.hook_handlers
        assert HookHandler.HANDLER in caps.hook_handlers
        assert caps.consent_model == ConsentModel.INTERACTIVE
        assert caps.policy_enforcement is True
        assert caps.evaluation_support is True

    def test_to_dict(self):
        caps = HostCapabilities()
        data = caps.to_dict()
        assert "manifestFormats" in data
        assert "executionForms" in data
        assert "hookHandlers" in data
        assert data["consentModel"] == "interactive"
        assert data["hostName"] == "houyi"

    def test_from_dict(self):
        data = {
            "manifestFormats": ["simpleskill.json"],
            "executionForms": ["in-process", "mcp"],
            "hookHandlers": ["command"],
            "consentModel": "policy",
            "policyEnforcement": False,
            "evaluationSupport": False,
            "hostName": "test-host",
            "hostVersion": "0.2.0",
        }
        caps = HostCapabilities.from_dict(data)
        assert ManifestFormat.SIMPLESKILL_JSON in caps.manifest_formats
        assert ExecutionForm.MCP in caps.execution_forms
        assert caps.consent_model == ConsentModel.POLICY
        assert not caps.policy_enforcement
        assert caps.host_name == "test-host"


class TestExtensionRequirements:
    """Test ExtensionRequirements dataclass."""

    def test_default_requirements(self):
        reqs = ExtensionRequirements()
        assert reqs.min_manifest_version == "0.1"
        assert reqs.required_execution_forms == []
        assert reqs.requires_consent is False

    def test_from_dict(self):
        data = {
            "minManifestVersion": "0.2",
            "requiredExecutionForms": ["mcp"],
            "requiredHookEvents": ["PreToolUse", "Stop"],
            "requiresConsent": True,
        }
        reqs = ExtensionRequirements.from_dict(data)
        assert reqs.min_manifest_version == "0.2"
        assert ExecutionForm.MCP in reqs.required_execution_forms
        assert "PreToolUse" in reqs.required_hook_events
        assert reqs.requires_consent is True


class TestCapabilityMatchResult:
    """Test CapabilityMatchResult dataclass."""

    def test_bool_conversion(self):
        result = CapabilityMatchResult(compatible=True)
        assert bool(result)

        result = CapabilityMatchResult(compatible=False)
        assert not bool(result)

    def test_with_missing_capabilities(self):
        result = CapabilityMatchResult(
            compatible=False,
            missing_capabilities=["MCP execution form required"],
        )
        assert not result
        assert len(result.missing_capabilities) == 1


class TestCapabilityNegotiator:
    """Test CapabilityNegotiator class."""

    def test_check_compatibility_basic(self):
        caps = HostCapabilities()
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements()
        result = negotiator.check_compatibility(reqs)

        assert result.compatible
        assert len(result.missing_capabilities) == 0

    def test_check_compatibility_version_mismatch(self):
        caps = HostCapabilities(manifest_version="0.1")
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(min_manifest_version="0.5")
        result = negotiator.check_compatibility(reqs)

        assert not result.compatible
        assert any("version" in m.lower() for m in result.missing_capabilities)

    def test_execution_form_mismatch(self):
        caps = HostCapabilities(execution_forms=[ExecutionForm.IN_PROCESS])
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(required_execution_forms=[ExecutionForm.MCP])
        result = negotiator.check_compatibility(reqs)

        assert not result.compatible
        assert any("execution" in m.lower() for m in result.missing_capabilities)

    def test_hook_events_mismatch(self):
        caps = HostCapabilities(hook_events=["PreToolUse"])
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(required_hook_events=["PreToolUse", "CustomEvent"])
        result = negotiator.check_compatibility(reqs)

        assert not result.compatible
        assert any("hook events" in m.lower() for m in result.missing_capabilities)

    def test_hook_handlers_mismatch(self):
        caps = HostCapabilities(hook_handlers=[HookHandler.COMMAND])
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(required_hook_handlers=[HookHandler.AGENT])
        result = negotiator.check_compatibility(reqs)

        assert not result.compatible
        assert any("handler" in m.lower() for m in result.missing_capabilities)

    def test_check_compatibility_consent_required(self):
        caps = HostCapabilities(consent_model=ConsentModel.NONE)
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(requires_consent=True)
        result = negotiator.check_compatibility(reqs)

        assert not result.compatible
        assert any("consent" in m.lower() for m in result.missing_capabilities)

    def test_check_compatibility_evaluation_warning(self):
        caps = HostCapabilities(evaluation_support=False)
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(requires_evaluation=True)
        result = negotiator.check_compatibility(reqs)

        # Evaluation is a warning, not a hard requirement
        assert result.compatible
        assert len(result.warnings) > 0

    def test_get_effective_capabilities(self):
        caps = HostCapabilities(
            execution_forms=[ExecutionForm.IN_PROCESS, ExecutionForm.MCP],
            hook_events=["PreToolUse", "PostToolUse", "Stop"],
        )
        negotiator = CapabilityNegotiator(caps)

        reqs = ExtensionRequirements(
            required_execution_forms=[ExecutionForm.MCP],
            required_hook_events=["PreToolUse"],
        )

        effective = negotiator.get_effective_capabilities(reqs)
        assert "mcp" in effective["executionForms"]
        assert "PreToolUse" in effective["hookEvents"]

    def test_version_satisfies(self):
        assert CapabilityNegotiator._version_satisfies("0.2", "0.1")
        assert CapabilityNegotiator._version_satisfies("1.0.0", "0.9")
        assert not CapabilityNegotiator._version_satisfies("0.1", "0.2")
        assert CapabilityNegotiator._version_satisfies("0.1", "0.1")


class TestDefaultHouyiCapabilities:
    """Test the default host capabilities constant."""

    def test_default_capabilities_exist(self):
        assert DEFAULT_HOUYI_CAPABILITIES is not None
        assert DEFAULT_HOUYI_CAPABILITIES.host_name == "houyi"

    def test_supports_all_forms(self):
        assert ExecutionForm.IN_PROCESS in DEFAULT_HOUYI_CAPABILITIES.execution_forms
        assert ExecutionForm.SUBPROCESS in DEFAULT_HOUYI_CAPABILITIES.execution_forms
        assert ExecutionForm.MCP in DEFAULT_HOUYI_CAPABILITIES.execution_forms

    def test_default_supports_standard_hooks(self):
        assert "PreToolUse" in DEFAULT_HOUYI_CAPABILITIES.hook_events
        assert "PostToolUse" in DEFAULT_HOUYI_CAPABILITIES.hook_events
        assert "Stop" in DEFAULT_HOUYI_CAPABILITIES.hook_events

    def test_default_has_observability(self):
        assert ObservabilityFeature.TRACE in DEFAULT_HOUYI_CAPABILITIES.observability
        assert ObservabilityFeature.EVENTS in DEFAULT_HOUYI_CAPABILITIES.observability
        assert ObservabilityFeature.METRICS in DEFAULT_HOUYI_CAPABILITIES.observability
