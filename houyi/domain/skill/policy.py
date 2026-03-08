"""Skill invocation policy and permissions for SimpleSkill."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from houyi.domain.skill.spec import ExecutionMode
else:
    from houyi.domain.skill.spec import ExecutionMode


class ModelAutoInvoke(str, Enum):
    """Model auto-invocation policy."""

    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_CONSENT = "allow_with_consent"


class SideEffect(str, Enum):
    """Side effect classification."""

    NONE = "none"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    EXEC = "exec"
    MIXED = "mixed"


@dataclass
class FilesystemPerm:
    """Filesystem permission declaration."""

    read: bool = False
    write: bool = False
    delete: bool = False
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "write": self.write,
            "delete": self.delete,
            "paths": self.paths,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilesystemPerm:
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
    domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "domains": self.domains,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkPerm:
        return cls(
            enabled=data.get("enabled", False),
            domains=data.get("domains", []),
        )


@dataclass
class ExecPerm:
    """Process execution permission declaration."""

    enabled: bool = False
    commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "commands": self.commands,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecPerm:
        return cls(
            enabled=data.get("enabled", False),
            commands=data.get("commands", []),
        )


@dataclass
class ResourceLimits:
    """Resource usage limits."""

    timeout_ms: int | None = None
    memory_mb: int | None = None
    cpu_percent: int | None = None
    concurrency: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_ms": self.timeout_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "concurrency": self.concurrency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceLimits:
        return cls(
            timeout_ms=data.get("timeout_ms"),
            memory_mb=data.get("memory_mb"),
            cpu_percent=data.get("cpu_percent"),
            concurrency=data.get("concurrency"),
        )


@dataclass
class Permissions:
    """Permission declarations for a skill/tool."""

    filesystem: FilesystemPerm = field(default_factory=FilesystemPerm)
    network: NetworkPerm = field(default_factory=NetworkPerm)
    exec: ExecPerm = field(default_factory=ExecPerm)
    secrets: list[str] = field(default_factory=list)
    resources: ResourceLimits = field(default_factory=ResourceLimits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filesystem": self.filesystem.to_dict(),
            "network": self.network.to_dict(),
            "exec": self.exec.to_dict(),
            "secrets": self.secrets,
            "resources": self.resources.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Permissions:
        return cls(
            filesystem=FilesystemPerm.from_dict(data.get("filesystem", {})),
            network=NetworkPerm.from_dict(data.get("network", {})),
            exec=ExecPerm.from_dict(data.get("exec", {})),
            secrets=data.get("secrets", []),
            resources=ResourceLimits.from_dict(data.get("resources", {})),
        )

    def requires_consent(self) -> bool:
        return (
            self.filesystem.write
            or self.filesystem.delete
            or self.network.enabled
            or self.exec.enabled
            or len(self.secrets) > 0
        )

    def describe(self) -> list[str]:
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
    """Invocation policy for a skill/tool."""

    model_auto_invoke: ModelAutoInvoke = ModelAutoInvoke.ALLOW
    user_invocable: bool = True
    side_effect: SideEffect = SideEffect.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelAutoInvoke": self.model_auto_invoke.value,
            "userInvocable": self.user_invocable,
            "sideEffect": self.side_effect.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvocationPolicy:
        model_auto = data.get("modelAutoInvoke", data.get("model_auto_invoke", "allow"))
        side_effect = data.get("sideEffect", data.get("side_effect", "none"))

        return cls(
            model_auto_invoke=ModelAutoInvoke(model_auto),
            user_invocable=data.get("userInvocable", data.get("user_invocable", True)),
            side_effect=SideEffect(side_effect),
        )

    @classmethod
    def default_for_side_effect(cls, side_effect: SideEffect) -> InvocationPolicy:
        if side_effect == SideEffect.NONE:
            return cls(
                model_auto_invoke=ModelAutoInvoke.ALLOW,
                side_effect=side_effect,
            )
        return cls(
            model_auto_invoke=ModelAutoInvoke.ALLOW_WITH_CONSENT,
            side_effect=side_effect,
        )

    def should_prompt_consent(self) -> bool:
        return self.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT

    def allows_model_invoke(self) -> bool:
        return self.model_auto_invoke != ModelAutoInvoke.DENY


class PolicyEnforcer:
    """Enforces invocation policies at runtime."""

    def __init__(
        self,
        default_policy: InvocationPolicy | None = None,
        baseline_policy: dict[str, InvocationPolicy] | None = None,
    ):
        self._default_policy = default_policy or InvocationPolicy()
        self._baseline_policy = baseline_policy or {}
        self._skill_policies: dict[str, InvocationPolicy] = {}

    def register_skill_policy(self, skill_name: str, policy: InvocationPolicy) -> None:
        self._skill_policies[skill_name] = policy

    def get_policy(self, skill_name: str) -> InvocationPolicy:
        return self._skill_policies.get(skill_name, self._default_policy)

    def check_invocation(
        self,
        skill_name: str,
        is_model_initiated: bool,
        user_consent_given: bool = False,
    ) -> InvocationDecision:
        policy = self.get_policy(skill_name)

        if not is_model_initiated:
            if policy.user_invocable:
                return InvocationDecision(allowed=True)
            return InvocationDecision(
                allowed=False,
                reason=f"Skill '{skill_name}' is not user-invocable",
            )

        if policy.model_auto_invoke == ModelAutoInvoke.DENY:
            return InvocationDecision(
                allowed=False,
                reason=f"Skill '{skill_name}' denies model auto-invocation",
            )

        if policy.model_auto_invoke == ModelAutoInvoke.ALLOW_WITH_CONSENT:
            if user_consent_given:
                return InvocationDecision(allowed=True)
            return InvocationDecision(
                allowed=False,
                requires_consent=True,
                reason=f"Skill '{skill_name}' requires user consent",
            )

        return InvocationDecision(allowed=True)


@dataclass
class InvocationDecision:
    """Result of policy check for an invocation."""

    allowed: bool
    requires_consent: bool = False
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


__all__ = [
    "ExecPerm",
    "ExecutionMode",
    "FilesystemPerm",
    "InvocationDecision",
    "InvocationPolicy",
    "ModelAutoInvoke",
    "NetworkPerm",
    "Permissions",
    "PolicyEnforcer",
    "ResourceLimits",
    "SideEffect",
]
