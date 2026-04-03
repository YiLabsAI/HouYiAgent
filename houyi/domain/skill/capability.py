"""Host capability negotiation for SimpleSkill.

This module implements capability negotiation as specified in §5.1:
- Host declares supported features
- Extensions can query host capabilities before registration
- Runtime capability matching for activation decisions

Reference: SimpleSkill Specification 0.1.0 Section 5.1 (Capability Negotiation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionForm(str, Enum):
    """Supported execution forms."""

    IN_PROCESS = "in-process"
    SUBPROCESS = "subprocess"
    MCP = "mcp"


class ManifestFormat(str, Enum):
    """Supported manifest formats."""

    SIMPLESKILL_JSON = "simpleskill.json"
    SKILL_MD = "SKILL.md"
    PACKAGE_JSON = "package.json"


class HookHandler(str, Enum):
    """Supported hook handler types."""

    COMMAND = "command"
    HANDLER = "handler"
    AGENT = "agent"


class ConsentModel(str, Enum):
    """Consent handling models."""

    INTERACTIVE = "interactive"
    POLICY = "policy"
    NONE = "none"


class ObservabilityFeature(str, Enum):
    """Observability features."""

    TRACE = "trace"
    EVENTS = "events"
    METRICS = "metrics"
    LOGS = "logs"


@dataclass
class HostCapabilities:
    """Host capability declaration."""

    manifest_formats: list[ManifestFormat] = field(
        default_factory=lambda: [
            ManifestFormat.SIMPLESKILL_JSON,
            ManifestFormat.SKILL_MD,
        ]
    )
    manifest_version: str = "0.1"
    execution_forms: list[ExecutionForm] = field(
        default_factory=lambda: [
            ExecutionForm.IN_PROCESS,
            ExecutionForm.SUBPROCESS,
        ]
    )
    hook_handlers: list[HookHandler] = field(
        default_factory=lambda: [
            HookHandler.COMMAND,
            HookHandler.HANDLER,
        ]
    )
    hook_events: list[str] = field(
        default_factory=lambda: [
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "SessionStart",
            "PreExecution",
            "PostExecution",
        ]
    )
    consent_model: ConsentModel = ConsentModel.INTERACTIVE
    policy_enforcement: bool = True
    observability: list[ObservabilityFeature] = field(
        default_factory=lambda: [
            ObservabilityFeature.TRACE,
            ObservabilityFeature.EVENTS,
        ]
    )
    evaluation_support: bool = True
    max_tool_timeout_ms: int = 300000
    max_concurrent_tools: int = 10
    host_name: str = "houyi"
    host_version: str = "0.3.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifestFormats": [f.value for f in self.manifest_formats],
            "manifestVersion": self.manifest_version,
            "executionForms": [f.value for f in self.execution_forms],
            "hookHandlers": [h.value for h in self.hook_handlers],
            "hookEvents": self.hook_events,
            "consentModel": self.consent_model.value,
            "policyEnforcement": self.policy_enforcement,
            "observability": [o.value for o in self.observability],
            "evaluationSupport": self.evaluation_support,
            "maxToolTimeoutMs": self.max_tool_timeout_ms,
            "maxConcurrentTools": self.max_concurrent_tools,
            "hostName": self.host_name,
            "hostVersion": self.host_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HostCapabilities:
        return cls(
            manifest_formats=[ManifestFormat(f) for f in data.get("manifestFormats", [])]
            or [ManifestFormat.SIMPLESKILL_JSON, ManifestFormat.SKILL_MD],
            manifest_version=data.get("manifestVersion", "0.1"),
            execution_forms=[ExecutionForm(f) for f in data.get("executionForms", [])]
            or [ExecutionForm.IN_PROCESS, ExecutionForm.SUBPROCESS],
            hook_handlers=[HookHandler(h) for h in data.get("hookHandlers", [])]
            or [HookHandler.COMMAND, HookHandler.HANDLER],
            hook_events=data.get("hookEvents", []),
            consent_model=ConsentModel(data.get("consentModel", "interactive")),
            policy_enforcement=data.get("policyEnforcement", True),
            observability=[ObservabilityFeature(o) for o in data.get("observability", [])],
            evaluation_support=data.get("evaluationSupport", True),
            max_tool_timeout_ms=data.get("maxToolTimeoutMs", 300000),
            max_concurrent_tools=data.get("maxConcurrentTools", 10),
            host_name=data.get("hostName", "houyi"),
            host_version=data.get("hostVersion", "0.3.0"),
        )


@dataclass
class ExtensionRequirements:
    """Extension requirements for capability negotiation."""

    min_manifest_version: str = "0.1"
    required_execution_forms: list[ExecutionForm] = field(default_factory=list)
    required_hook_events: list[str] = field(default_factory=list)
    required_hook_handlers: list[HookHandler] = field(default_factory=list)
    requires_consent: bool = False
    requires_evaluation: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtensionRequirements:
        return cls(
            min_manifest_version=data.get("minManifestVersion", "0.1"),
            required_execution_forms=[
                ExecutionForm(f) for f in data.get("requiredExecutionForms", [])
            ],
            required_hook_events=data.get("requiredHookEvents", []),
            required_hook_handlers=[HookHandler(h) for h in data.get("requiredHookHandlers", [])],
            requires_consent=data.get("requiresConsent", False),
            requires_evaluation=data.get("requiresEvaluation", False),
        )


@dataclass
class CapabilityMatchResult:
    """Result of capability negotiation."""

    compatible: bool
    missing_capabilities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.compatible


class CapabilityNegotiator:
    """Negotiates capabilities between host and extensions."""

    def __init__(self, host_capabilities: HostCapabilities) -> None:
        self._capabilities = host_capabilities

    @property
    def capabilities(self) -> HostCapabilities:
        return self._capabilities

    def check_compatibility(
        self,
        requirements: ExtensionRequirements,
    ) -> CapabilityMatchResult:
        missing = []
        warnings = []

        if not self._version_satisfies(
            self._capabilities.manifest_version,
            requirements.min_manifest_version,
        ):
            missing.append(
                f"Manifest version {requirements.min_manifest_version} required, "
                f"host supports {self._capabilities.manifest_version}"
            )

        if requirements.required_execution_forms:
            supported_forms = set(self._capabilities.execution_forms)
            required_forms = set(requirements.required_execution_forms)
            if not required_forms & supported_forms:
                missing.append(
                    f"Required execution forms: {[f.value for f in required_forms]}, "
                    f"host supports: {[f.value for f in supported_forms]}"
                )

        if requirements.required_hook_events:
            supported_events = set(self._capabilities.hook_events)
            required_events = set(requirements.required_hook_events)
            missing_events = required_events - supported_events
            if missing_events:
                missing.append(f"Missing hook events: {list(missing_events)}")

        if requirements.required_hook_handlers:
            supported_handlers = set(self._capabilities.hook_handlers)
            required_handlers = set(requirements.required_hook_handlers)
            missing_handlers = required_handlers - supported_handlers
            if missing_handlers:
                missing.append(f"Missing hook handlers: {[h.value for h in missing_handlers]}")

        if requirements.requires_consent and self._capabilities.consent_model == ConsentModel.NONE:
            missing.append("Extension requires consent support")

        if requirements.requires_evaluation and not self._capabilities.evaluation_support:
            warnings.append("Extension prefers evaluation support (not critical)")

        return CapabilityMatchResult(
            compatible=len(missing) == 0,
            missing_capabilities=missing,
            warnings=warnings,
        )

    def get_effective_capabilities(
        self,
        requirements: ExtensionRequirements,
    ) -> dict[str, Any]:
        return {
            "executionForms": [
                f.value
                for f in self._capabilities.execution_forms
                if not requirements.required_execution_forms
                or f in requirements.required_execution_forms
            ],
            "hookEvents": [
                e
                for e in self._capabilities.hook_events
                if not requirements.required_hook_events or e in requirements.required_hook_events
            ],
            "hookHandlers": [
                h.value
                for h in self._capabilities.hook_handlers
                if not requirements.required_hook_handlers
                or h in requirements.required_hook_handlers
            ],
            "consentModel": self._capabilities.consent_model.value,
            "evaluationSupport": self._capabilities.evaluation_support,
        }

    @staticmethod
    def _version_satisfies(host_version: str, required_version: str) -> bool:
        def parse_version(v: str) -> tuple[int, ...]:
            parts = v.lstrip(">=<").split(".")
            return tuple(int(p) for p in parts if p.isdigit())

        host_parts = parse_version(host_version)
        required_parts = parse_version(required_version)
        max_len = max(len(host_parts), len(required_parts))
        host_parts = host_parts + (0,) * (max_len - len(host_parts))
        required_parts = required_parts + (0,) * (max_len - len(required_parts))
        return host_parts >= required_parts


DEFAULT_HOUYI_CAPABILITIES = HostCapabilities(
    manifest_formats=[ManifestFormat.SIMPLESKILL_JSON, ManifestFormat.SKILL_MD],
    manifest_version="0.1",
    execution_forms=[ExecutionForm.IN_PROCESS, ExecutionForm.SUBPROCESS, ExecutionForm.MCP],
    hook_handlers=[HookHandler.COMMAND, HookHandler.HANDLER],
    hook_events=[
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SessionStart",
        "PreExecution",
        "PostExecution",
    ],
    consent_model=ConsentModel.INTERACTIVE,
    policy_enforcement=True,
    observability=[
        ObservabilityFeature.TRACE,
        ObservabilityFeature.EVENTS,
        ObservabilityFeature.METRICS,
    ],
    evaluation_support=True,
    host_name="houyi",
    host_version="0.3.0",
)
