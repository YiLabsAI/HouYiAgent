"""Skill invocation policy and permissions for SimpleSkill v0.1.

This module implements:
- InvocationPolicy: Model auto-invoke governance
- Permissions: Resource access declarations
- SideEffect: Side effect classification

Reference: SimpleSkill Specification v0.1 Section 5.2 (Invocation Policy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelAutoInvoke(str, Enum):
    """Model auto-invocation policy.
    
    Controls whether the LLM can automatically invoke this skill/tool
    without explicit user confirmation.
    """
    
    ALLOW = "allow"
    """Allow model to invoke automatically."""
    
    DENY = "deny"
    """Deny automatic invocation; requires explicit user action."""
    
    ALLOW_WITH_CONSENT = "allow_with_consent"
    """Allow with runtime consent prompt."""


class SideEffect(str, Enum):
    """Side effect classification.
    
    Declares what kind of side effects the skill may produce.
    Used for governance decisions and consent requirements.
    """
    
    NONE = "none"
    """No side effects (pure computation, read-only)."""
    
    FILESYSTEM = "filesystem"
    """May read/write/delete files."""
    
    NETWORK = "network"
    """May make network requests."""
    
    EXEC = "exec"
    """May execute external processes."""
    
    MIXED = "mixed"
    """Multiple types of side effects."""


@dataclass
class FilesystemPerm:
    """Filesystem permission declaration."""
    
    read: bool = False
    """Allow reading files."""
    
    write: bool = False
    """Allow writing files."""
    
    delete: bool = False
    """Allow deleting files."""
    
    paths: list[str] = field(default_factory=list)
    """Allowed path patterns (glob). Empty = workspace only."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "write": self.write,
            "delete": self.delete,
            "paths": self.paths,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilesystemPerm":
        return cls(
            read=data.get("read", False),
            write=data.get("write", False),
            delete=data.get("delete", False),
            paths=data.get("paths", []),
        )


@dataclass
class NetworkPerm:
    """Network permission declaration."""
    
    enabled: bool = False
    """Allow network access."""
    
    domains: list[str] = field(default_factory=list)
    """Allowed domain patterns. Empty = all domains (if enabled)."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "domains": self.domains,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkPerm":
        return cls(
            enabled=data.get("enabled", False),
            domains=data.get("domains", []),
        )


@dataclass
class ExecPerm:
    """Process execution permission declaration."""
    
    enabled: bool = False
    """Allow executing external processes."""
    
    commands: list[str] = field(default_factory=list)
    """Allowed command patterns. Empty = none allowed."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "commands": self.commands,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecPerm":
        return cls(
            enabled=data.get("enabled", False),
            commands=data.get("commands", []),
        )


@dataclass
class ResourceLimits:
    """Resource usage limits."""
    
    timeout_ms: int | None = None
    """Maximum execution time in milliseconds."""
    
    memory_mb: int | None = None
    """Maximum memory usage in megabytes."""
    
    cpu_percent: int | None = None
    """Maximum CPU usage percentage."""
    
    concurrency: int | None = None
    """Maximum concurrent invocations."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_ms": self.timeout_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "concurrency": self.concurrency,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceLimits":
        return cls(
            timeout_ms=data.get("timeout_ms"),
            memory_mb=data.get("memory_mb"),
            cpu_percent=data.get("cpu_percent"),
            concurrency=data.get("concurrency"),
        )


@dataclass
class Permissions:
    """Permission declarations for a skill/tool.
    
    Reference: SimpleSkill Specification v0.1 Section 3.6 (Permissions)
    """
    
    filesystem: FilesystemPerm = field(default_factory=FilesystemPerm)
    """Filesystem access permissions."""
    
    network: NetworkPerm = field(default_factory=NetworkPerm)
    """Network access permissions."""
    
    exec: ExecPerm = field(default_factory=ExecPerm)
    """Process execution permissions."""
    
    secrets: list[str] = field(default_factory=list)
    """List of secret keys the skill needs access to."""
    
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    """Resource usage limits."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "filesystem": self.filesystem.to_dict(),
            "network": self.network.to_dict(),
            "exec": self.exec.to_dict(),
            "secrets": self.secrets,
            "resources": self.resources.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Permissions":
        return cls(
            filesystem=FilesystemPerm.from_dict(data.get("filesystem", {})),
            network=NetworkPerm.from_dict(data.get("network", {})),
            exec=ExecPerm.from_dict(data.get("exec", {})),
            secrets=data.get("secrets", []),
            resources=ResourceLimits.from_dict(data.get("resources", {})),
        )
    
    def requires_consent(self) -> bool:
        """Check if these permissions require user consent.
        
        Returns True if any sensitive permission is requested.
        """
        return (
            self.filesystem.write
            or self.filesystem.delete
            or self.network.enabled
            or self.exec.enabled
            or len(self.secrets) > 0
        )
    
    def describe(self) -> list[str]:
        """Generate human-readable permission descriptions."""
        descriptions = []
        
        if self.filesystem.read:
            paths = ", ".join(self.filesystem.paths) if self.filesystem.paths else "workspace"
            descriptions.append(f"Read files from: {paths}")
        if self.filesystem.write:
            paths = ", ".join(self.filesystem.paths) if self.filesystem.paths else "workspace"
            descriptions.append(f"Write files to: {paths}")
        if self.filesystem.delete:
            descriptions.append("Delete files")
        
        if self.network.enabled:
            if self.network.domains:
                descriptions.append(f"Network access to: {', '.join(self.network.domains)}")
            else:
                descriptions.append("Network access (unrestricted)")
        
        if self.exec.enabled:
            if self.exec.commands:
                descriptions.append(f"Execute commands: {', '.join(self.exec.commands)}")
            else:
                descriptions.append("Execute external processes")
        
        if self.secrets:
            descriptions.append(f"Access secrets: {', '.join(self.secrets)}")
        
        return descriptions


@dataclass
class InvocationPolicy:
    """Invocation policy for a skill/tool.
    
    Controls how the skill can be invoked and what governance
    rules apply.
    
    Reference: SimpleSkill Specification v0.1 Section 5.2 (Invocation Policy)
    """
    
    model_auto_invoke: ModelAutoInvoke = ModelAutoInvoke.ALLOW
    """Whether model can auto-invoke this skill."""
    
    user_invocable: bool = True
    """Whether user can directly invoke this skill."""
    
    side_effect: SideEffect = SideEffect.NONE
    """What kind of side effects this skill may produce."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "modelAutoInvoke": self.model_auto_invoke.value,
            "userInvocable": self.user_invocable,
            "sideEffect": self.side_effect.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InvocationPolicy":
        model_auto = data.get("modelAutoInvoke", data.get("model_auto_invoke", "allow"))
        side_effect = data.get("sideEffect", data.get("side_effect", "none"))
        
        return cls(
            model_auto_invoke=ModelAutoInvoke(model_auto),
            user_invocable=data.get("userInvocable", data.get("user_invocable", True)),
            side_effect=SideEffect(side_effect),
        )
    
    @classmethod
    def default_for_side_effect(cls, side_effect: SideEffect) -> "InvocationPolicy":
        """Create default policy based on side effect type.
        
        Per spec: For sideEffect != none, modelAutoInvoke SHOULD default
        to deny or allow_with_consent.
        """
        if side_effect == SideEffect.NONE:
            return cls(
                model_auto_invoke=ModelAutoInvoke.ALLOW,
                side_effect=side_effect,
            )
        else:
            return cls(
                model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
                side_effect=side_effect,
            )
    
    def should_prompt_consent(self) -> bool:
        """Check if this policy requires consent prompt before invocation."""
        return self.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT
    
    def allows_model_invoke(self) -> bool:
        """Check if model is allowed to invoke (may still need consent)."""
        return self.model_auto_invoke != ModelAutoInvoke.DENY


class PolicyEnforcer:
    """Enforces invocation policies at runtime.
    
    This class is used by the execution engine to check if a tool/skill
    invocation should be allowed, denied, or require consent.
    """
    
    def __init__(
        self,
        default_policy: InvocationPolicy | None = None,
        baseline_policy: dict[str, InvocationPolicy] | None = None,
    ):
        """Initialize policy enforcer.
        
        Args:
            default_policy: Default policy for skills without explicit policy
            baseline_policy: Host baseline policies by side effect type
        """
        self._default_policy = default_policy or InvocationPolicy()
        self._baseline_policy = baseline_policy or {}
        self._skill_policies: dict[str, InvocationPolicy] = {}
    
    def register_skill_policy(self, skill_name: str, policy: InvocationPolicy) -> None:
        """Register a policy for a specific skill."""
        self._skill_policies[skill_name] = policy
    
    def get_policy(self, skill_name: str) -> InvocationPolicy:
        """Get the effective policy for a skill."""
        return self._skill_policies.get(skill_name, self._default_policy)
    
    def check_invocation(
        self,
        skill_name: str,
        is_model_initiated: bool,
        user_consent_given: bool = False,
    ) -> "InvocationDecision":
        """Check if an invocation should be allowed.
        
        Args:
            skill_name: Name of the skill being invoked
            is_model_initiated: True if invoked by model, False if by user
            user_consent_given: True if user has already given consent
            
        Returns:
            InvocationDecision with allow/deny/prompt status
        """
        policy = self.get_policy(skill_name)
        
        # User-initiated invocations
        if not is_model_initiated:
            if policy.user_invocable:
                return InvocationDecision(allowed=True)
            else:
                return InvocationDecision(
                    allowed=False,
                    reason=f"Skill '{skill_name}' is not user-invocable",
                )
        
        # Model-initiated invocations
        if policy.model_auto_invoke == ModelAutoInvoke.DENY:
            return InvocationDecision(
                allowed=False,
                reason=f"Skill '{skill_name}' denies model auto-invocation",
            )
        
        if policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT:
            if user_consent_given:
                return InvocationDecision(allowed=True)
            else:
                return InvocationDecision(
                    allowed=False,
                    requires_consent=True,
                    reason=f"Skill '{skill_name}' requires user consent",
                )
        
        # ModelAutoInvoke.ALLOW
        return InvocationDecision(allowed=True)


@dataclass
class InvocationDecision:
    """Result of policy check for an invocation."""
    
    allowed: bool
    """Whether the invocation is allowed."""
    
    requires_consent: bool = False
    """Whether user consent is required before allowing."""
    
    reason: str | None = None
    """Reason for denial or consent requirement."""
    
    def __bool__(self) -> bool:
        return self.allowed
