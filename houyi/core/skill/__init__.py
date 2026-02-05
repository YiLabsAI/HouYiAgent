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
from houyi.core.skill.config import (
    DEFAULT_SKILL_CONFIG,
    ConsentConfig,
    HookConfig,
    MetricsConfig,
    PolicyConfig,
    SkillConfig,
    get_skill_config,
    load_skill_config_from_env,
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
from houyi.core.skill.schema import parse_hooks_config, parse_skill_md
from houyi.core.skill.spec import ExecutionMode, SkillSpec

__all__ = [
    # spec.py
    "ExecutionMode",
    "SkillSpec",
    # hooks.py
    "HookContext",
    "HookEvent",
    "HookResult",
    "HookType",
    "SkillHook",
    "SkillHooksManager",
    # schema.py
    "parse_hooks_config",
    "parse_skill_md",
    # manifest.py
    "ActivationEvent",
    "ActivationEventType",
    "Contributions",
    "ManifestRegistry",
    "SkillManifest",
    "ToolContribution",
    "SkillContribution",
    "ResourceContribution",
    # policy.py
    "FilesystemPerm",
    "NetworkPerm",
    "ExecPerm",
    "ResourceLimits",
    "Permissions",
    "InvocationPolicy",
    "ModelAutoInvoke",
    "SideEffect",
    "PolicyEnforcer",
    "InvocationDecision",
    # consent.py
    "ConsentType",
    "ConsentResult",
    "ConsentRequest",
    "ConsentResponse",
    "ConsentHandler",
    "ConsentStore",
    "ConsentManager",
    "InMemoryConsentStore",
    "FileConsentStore",
    "CLIConsentHandler",
    "PolicyBasedConsentHandler",
    # capability.py
    "ExecutionForm",
    "ManifestFormat",
    "HookHandler",
    "ConsentModel",
    "ObservabilityFeature",
    "HostCapabilities",
    "ExtensionRequirements",
    "CapabilityMatchResult",
    "CapabilityNegotiator",
    "DEFAULT_HOUYI_CAPABILITIES",
    # metrics.py
    "QualityMetrics",
    "LatencyMetrics",
    "CostMetrics",
    "ReliabilityMetrics",
    "PrivacyMetrics",
    "ConformanceMetrics",
    "SkillMetrics",
    "MetricsCollector",
    "MetricsExporter",
    "MetricsStore",
    # config.py
    "HookConfig",
    "ConsentConfig",
    "MetricsConfig",
    "PolicyConfig",
    "SkillConfig",
    "get_skill_config",
    "load_skill_config_from_env",
    "DEFAULT_SKILL_CONFIG",
]
