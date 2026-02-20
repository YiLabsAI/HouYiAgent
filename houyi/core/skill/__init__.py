"""SimpleSkill standard implementation.

This package implements the SimpleSkill v0.1 specification:
- Layer A: Manifest (simpleskill.json parsing)
- Layer B: Host Runtime API (hooks, policy, consent, capability)
- Evaluation metrics

Reference implementation of the SimpleSkill standard.

Usage:
    from houyi.core.skill import (
        SkillSpec,
        HookEvent,
        SkillHooksManager,
        InvocationPolicy,
        ConsentManager,
        HostCapabilities,
        SkillMetrics,
    )
"""

from houyi.core.skill.capability import (
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
from houyi.core.skill.consent import (
    CLIConsentHandler,
    ConsentHandler,
    ConsentManager,
    ConsentRequest,
    ConsentResponse,
    ConsentResult,
    ConsentStore,
    ConsentType,
    FileConsentStore,
    InMemoryConsentStore,
    PolicyBasedConsentHandler,
)
from houyi.core.skill.hooks import (
    HookContext,
    HookEvent,
    HookResult,
    HookType,
    SkillHook,
    SkillHooksManager,
)
from houyi.core.skill.manifest import (
    ActivationEvent,
    ActivationEventType,
    Contributions,
    ManifestRegistry,
    ResourceContribution,
    SkillContribution,
    SkillManifest,
    ToolContribution,
)
from houyi.core.skill.metrics import (
    ConformanceMetrics,
    CostMetrics,
    LatencyMetrics,
    MetricsCollector,
    MetricsExporter,
    MetricsStore,
    PrivacyMetrics,
    QualityMetrics,
    ReliabilityMetrics,
    SkillMetrics,
)
from houyi.core.skill.policy import (
    ExecPerm,
    FilesystemPerm,
    InvocationDecision,
    InvocationPolicy,
    ModelAutoInvoke,
    NetworkPerm,
    Permissions,
    PolicyEnforcer,
    ResourceLimits,
    SideEffect,
)
from houyi.core.skill.preprocessor import (
    PreprocessorPipeline,
    PreprocessorResult,
    PreprocessorSpec,
    PreprocessorType,
)
from houyi.core.skill.schema import parse_hooks_config, parse_skill_md
from houyi.core.skill.spec import ExecutionMode, SkillSpec
from houyi.core.skill.tool_router import ToolRouter, ToolRouteResult

__all__ = [
    "DEFAULT_HOUYI_CAPABILITIES",
    # manifest.py
    "ActivationEvent",
    "ActivationEventType",
    "CLIConsentHandler",
    "CapabilityMatchResult",
    "CapabilityNegotiator",
    "ConformanceMetrics",
    "ConsentHandler",
    "ConsentManager",
    "ConsentModel",
    "ConsentRequest",
    "ConsentResponse",
    "ConsentResult",
    "ConsentStore",
    # consent.py
    "ConsentType",
    "Contributions",
    "CostMetrics",
    "ExecPerm",
    # capability.py
    "ExecutionForm",
    # spec.py
    "ExecutionMode",
    "ExtensionRequirements",
    "FileConsentStore",
    # policy.py
    "FilesystemPerm",
    # hooks.py
    "HookContext",
    "HookEvent",
    "HookHandler",
    "HookResult",
    "HookType",
    "HostCapabilities",
    "InMemoryConsentStore",
    "InvocationDecision",
    "InvocationPolicy",
    "LatencyMetrics",
    "ManifestFormat",
    "ManifestRegistry",
    "MetricsCollector",
    "MetricsExporter",
    "MetricsStore",
    "ModelAutoInvoke",
    "NetworkPerm",
    "ObservabilityFeature",
    "Permissions",
    "PolicyBasedConsentHandler",
    "PolicyEnforcer",
    "PreprocessorPipeline",
    "PreprocessorResult",
    "PreprocessorSpec",
    # preprocessor.py
    "PreprocessorType",
    "PrivacyMetrics",
    # metrics.py
    "QualityMetrics",
    "ReliabilityMetrics",
    "ResourceContribution",
    "ResourceLimits",
    "SideEffect",
    "SkillContribution",
    "SkillHook",
    "SkillHooksManager",
    "SkillManifest",
    "SkillMetrics",
    "SkillSpec",
    "ToolContribution",
    "ToolRouteResult",
    # tool_router.py
    "ToolRouter",
    # schema.py
    "parse_hooks_config",
    "parse_skill_md",
]
