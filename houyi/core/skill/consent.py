"""Consent interface for SimpleSkill v0.1.

This module implements the unified consent interface for:
- First-time extension permission grants
- Single high-risk operation confirmations
- Audit logging

Reference: SimpleSkill Specification v0.1 Section 5.3 (Consent)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from houyi.core.skill.policy import InvocationPolicy, Permissions

logger = logging.getLogger(__name__)


class ConsentType(str, Enum):
    """Type of consent being requested."""

    PERMISSION_GRANT = "permission_grant"
    """First-time permission grant for a skill/extension."""

    OPERATION_CONFIRM = "operation_confirm"
    """Single high-risk operation confirmation."""

    INVOKE_CONFIRM = "invoke_confirm"
    """Model auto-invoke with consent required."""


class ConsentResult(str, Enum):
    """Result of a consent request."""

    GRANTED = "granted"
    """User granted consent."""

    DENIED = "denied"
    """User denied consent."""

    REMEMBERED = "remembered"
    """Consent was previously granted and remembered."""

    TIMEOUT = "timeout"
    """Consent request timed out."""

    NOT_INTERACTIVE = "not_interactive"
    """Non-interactive mode, consent cannot be requested."""


@dataclass
class ConsentRequest:
    """A request for user consent."""

    consent_type: ConsentType
    """Type of consent being requested."""

    skill_name: str
    """Name of the skill requesting consent."""

    operation: str | None = None
    """Specific operation being requested (for OPERATION_CONFIRM)."""

    permissions: Permissions | None = None
    """Permissions being requested (for PERMISSION_GRANT)."""

    policy: InvocationPolicy | None = None
    """Invocation policy context."""

    context: dict[str, Any] = field(default_factory=dict)
    """Additional context for the consent request."""

    remember: bool = False
    """Whether user chose to remember this decision."""

    def describe(self) -> str:
        """Generate human-readable description of the consent request."""
        if self.consent_type == ConsentType.PERMISSION_GRANT:
            perms = self.permissions.describe() if self.permissions else []
            perm_str = "\n  - ".join(perms) if perms else "No specific permissions"
            return f"Skill '{self.skill_name}' requests the following permissions:\n  - {perm_str}"

        elif self.consent_type == ConsentType.OPERATION_CONFIRM:
            return f"Skill '{self.skill_name}' wants to perform: {self.operation}"

        elif self.consent_type == ConsentType.INVOKE_CONFIRM:
            side_effect = self.policy.side_effect.value if self.policy else "unknown"
            return f"Allow model to invoke '{self.skill_name}'?\n  Side effect: {side_effect}"

        return f"Consent requested for skill '{self.skill_name}'"


@dataclass
class ConsentResponse:
    """Response to a consent request."""

    result: ConsentResult
    """The consent decision."""

    request: ConsentRequest
    """The original request."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the consent was given/denied."""

    expires_at: datetime | None = None
    """When this consent expires (if remembered)."""

    reason: str | None = None
    """Optional reason provided by user."""

    def is_granted(self) -> bool:
        """Check if consent was granted."""
        return self.result in (ConsentResult.GRANTED, ConsentResult.REMEMBERED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/persistence."""
        return {
            "result": self.result.value,
            "skill_name": self.request.skill_name,
            "consent_type": self.request.consent_type.value,
            "operation": self.request.operation,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason": self.reason,
            "remember": self.request.remember,
        }


@runtime_checkable
class ConsentHandler(Protocol):
    """Protocol for consent handlers.

    Implementations can be CLI-based, GUI-based, or policy-based.
    """

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
        """Request consent from user.

        Args:
            request: The consent request

        Returns:
            ConsentResponse with the user's decision
        """
        ...

    def check_remembered(self, request: ConsentRequest) -> ConsentResponse | None:
        """Check if consent was previously granted and remembered.

        Args:
            request: The consent request to check

        Returns:
            ConsentResponse if remembered, None otherwise
        """
        ...


class ConsentStore(ABC):
    """Abstract base for consent persistence."""

    @abstractmethod
    def save(self, response: ConsentResponse) -> None:
        """Save a consent response."""
        ...

    @abstractmethod
    def load(self, skill_name: str, consent_type: ConsentType) -> ConsentResponse | None:
        """Load a remembered consent response."""
        ...

    @abstractmethod
    def revoke(self, skill_name: str, consent_type: ConsentType | None = None) -> None:
        """Revoke previously granted consent."""
        ...


class InMemoryConsentStore(ConsentStore):
    """In-memory consent store (for testing)."""

    def __init__(self) -> None:
        self._consents: dict[tuple[str, str], ConsentResponse] = {}

    def save(self, response: ConsentResponse) -> None:
        if response.request.remember and response.is_granted():
            key = (response.request.skill_name, response.request.consent_type.value)
            self._consents[key] = response

    def load(self, skill_name: str, consent_type: ConsentType) -> ConsentResponse | None:
        key = (skill_name, consent_type.value)
        response = self._consents.get(key)

        if not response:
            return None

        if response.expires_at:
            if datetime.now(timezone.utc) > response.expires_at:
                del self._consents[key]
                return None

        # Return with REMEMBERED status to indicate it was loaded from store
        return ConsentResponse(
            result=ConsentResult.REMEMBERED,
            request=response.request,
            timestamp=response.timestamp,
            expires_at=response.expires_at,
        )

    def revoke(self, skill_name: str, consent_type: ConsentType | None = None) -> None:
        if consent_type:
            key = (skill_name, consent_type.value)
            self._consents.pop(key, None)
        else:
            # Revoke all consents for this skill
            keys_to_remove = [k for k in self._consents if k[0] == skill_name]
            for key in keys_to_remove:
                del self._consents[key]


class FileConsentStore(ConsentStore):
    """File-based consent store for persistence."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path.home() / ".houyi" / "consent.json"
        self._path = Path(path)
        self._consents: dict[str, dict[str, Any]] = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Load consents from file."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._consents = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load consent store: {e}")
                self._consents = {}

    def _save_to_file(self) -> None:
        """Save consents to file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._consents, f, indent=2)

    def save(self, response: ConsentResponse) -> None:
        if response.request.remember and response.is_granted():
            key = f"{response.request.skill_name}:{response.request.consent_type.value}"
            self._consents[key] = response.to_dict()
            self._save_to_file()

    def load(self, skill_name: str, consent_type: ConsentType) -> ConsentResponse | None:
        key = f"{skill_name}:{consent_type.value}"
        data = self._consents.get(key)

        if not data:
            return None

        # Check expiration
        expires_at = data.get("expires_at")
        if expires_at:
            expires = datetime.fromisoformat(expires_at)
            if datetime.now(timezone.utc) > expires:
                del self._consents[key]
                self._save_to_file()
                return None

        # Reconstruct response
        request = ConsentRequest(
            consent_type=ConsentType(data["consent_type"]),
            skill_name=data["skill_name"],
            operation=data.get("operation"),
            remember=data.get("remember", False),
        )

        return ConsentResponse(
            result=ConsentResult.REMEMBERED,
            request=request,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
        )

    def revoke(self, skill_name: str, consent_type: ConsentType | None = None) -> None:
        if consent_type:
            key = f"{skill_name}:{consent_type.value}"
            self._consents.pop(key, None)
        else:
            keys_to_remove = [k for k in self._consents if k.startswith(f"{skill_name}:")]
            for key in keys_to_remove:
                del self._consents[key]
        self._save_to_file()


class ConsentManager:
    """Manages consent requests and responses.

    This is the main entry point for the consent system.
    """

    def __init__(
        self,
        handler: ConsentHandler | None = None,
        store: ConsentStore | None = None,
        interactive: bool = True,
    ) -> None:
        """Initialize consent manager.

        Args:
            handler: Consent handler for interactive prompts
            store: Consent store for persistence
            interactive: Whether interactive consent is allowed
        """
        self._handler = handler
        self._store = store or InMemoryConsentStore()
        self._interactive = interactive
        self._audit_log: list[ConsentResponse] = []

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
        """Request consent, checking remembered consents first.

        Args:
            request: The consent request

        Returns:
            ConsentResponse with the decision
        """
        # Check remembered consent
        remembered = self._store.load(request.skill_name, request.consent_type)
        if remembered:
            self._audit(remembered)
            return remembered

        # Non-interactive mode
        if not self._interactive:
            response = ConsentResponse(
                result=ConsentResult.NOT_INTERACTIVE,
                request=request,
                reason="Consent cannot be requested in non-interactive mode",
            )
            self._audit(response)
            return response

        # No handler available
        if not self._handler:
            response = ConsentResponse(
                result=ConsentResult.DENIED,
                request=request,
                reason="No consent handler configured",
            )
            self._audit(response)
            return response

        # Request consent from handler
        response = await self._handler.request_consent(request)

        # Store if remembered
        if response.is_granted() and request.remember:
            self._store.save(response)

        self._audit(response)
        return response

    def check_permission(self, skill_name: str) -> bool:
        """Check if permission was previously granted for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            True if permission was granted and remembered
        """
        response = self._store.load(skill_name, ConsentType.PERMISSION_GRANT)
        return response is not None and response.is_granted()

    def revoke_consent(
        self,
        skill_name: str,
        consent_type: ConsentType | None = None,
    ) -> None:
        """Revoke previously granted consent.

        Args:
            skill_name: Name of the skill
            consent_type: Specific type to revoke, or None for all
        """
        self._store.revoke(skill_name, consent_type)
        logger.info(f"Revoked consent for skill '{skill_name}' (type={consent_type})")

    def get_audit_log(self) -> list[ConsentResponse]:
        """Get the audit log of consent decisions."""
        return list(self._audit_log)

    def export_audit_log(self, path: Path | str) -> None:
        """Export audit log to file.

        Args:
            path: Path to export to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entries = [r.to_dict() for r in self._audit_log]
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Exported {len(entries)} consent audit entries to {path}")

    def _audit(self, response: ConsentResponse) -> None:
        """Record consent decision in audit log."""
        self._audit_log.append(response)

        # Log significant decisions
        if response.result in (ConsentResult.GRANTED, ConsentResult.DENIED):
            logger.info(
                f"Consent {response.result.value}: "
                f"skill={response.request.skill_name}, "
                f"type={response.request.consent_type.value}"
            )


class CLIConsentHandler:
    """Simple CLI-based consent handler.

    Prompts user via stdin/stdout for consent.
    For production, replace with a proper UI handler.
    """

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
        """Request consent via CLI prompt."""
        print("\n" + "=" * 50)
        print("CONSENT REQUEST")
        print("=" * 50)
        print(request.describe())
        print()

        try:
            response_str = input("Allow? [y/N/r(remember)]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ConsentResponse(
                result=ConsentResult.DENIED,
                request=request,
                reason="User cancelled",
            )

        if response_str in ("y", "yes"):
            return ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
            )
        elif response_str in ("r", "remember"):
            request.remember = True
            return ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
            )
        else:
            return ConsentResponse(
                result=ConsentResult.DENIED,
                request=request,
            )

    def check_remembered(self, request: ConsentRequest) -> ConsentResponse | None:
        """CLI handler doesn't check remembered (delegated to store)."""
        return None


class PolicyBasedConsentHandler:
    """Policy-based consent handler for non-interactive scenarios.

    Uses predefined rules to auto-grant or deny consent.
    """

    def __init__(
        self,
        auto_grant_skills: set[str] | None = None,
        auto_deny_skills: set[str] | None = None,
        default_grant: bool = False,
    ) -> None:
        """Initialize policy-based handler.

        Args:
            auto_grant_skills: Skills that are automatically granted
            auto_deny_skills: Skills that are automatically denied
            default_grant: Default decision if skill not in either list
        """
        self._auto_grant = auto_grant_skills or set()
        self._auto_deny = auto_deny_skills or set()
        self._default_grant = default_grant

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
        """Apply policy rules to decide consent."""
        skill_name = request.skill_name

        if skill_name in self._auto_grant:
            return ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
                reason="Auto-granted by policy",
            )

        if skill_name in self._auto_deny:
            return ConsentResponse(
                result=ConsentResult.DENIED,
                request=request,
                reason="Auto-denied by policy",
            )

        if self._default_grant:
            return ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
                reason="Default grant policy",
            )

        return ConsentResponse(
            result=ConsentResult.DENIED,
            request=request,
            reason="Default deny policy",
        )

    def check_remembered(self, request: ConsentRequest) -> ConsentResponse | None:
        """Policy handler doesn't use remembered consents."""
        return None
