"""Consent interface for SimpleSkill."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from houyi.domain.skill.policy import InvocationPolicy, Permissions

logger = logging.getLogger(__name__)


class ConsentType(str, Enum):
    """Type of consent being requested."""

    PERMISSION_GRANT = "permission_grant"
    OPERATION_CONFIRM = "operation_confirm"
    INVOKE_CONFIRM = "invoke_confirm"


class ConsentResult(str, Enum):
    """Result of a consent request."""

    GRANTED = "granted"
    DENIED = "denied"
    REMEMBERED = "remembered"
    TIMEOUT = "timeout"
    NOT_INTERACTIVE = "not_interactive"


@dataclass
class ConsentRequest:
    """A request for user consent."""

    consent_type: ConsentType
    skill_name: str
    operation: str | None = None
    permissions: Permissions | None = None
    policy: InvocationPolicy | None = None
    context: dict[str, Any] = field(default_factory=dict)
    remember: bool = False

    def describe(self) -> str:
        if self.consent_type == ConsentType.PERMISSION_GRANT:
            perms = self.permissions.describe() if self.permissions else []
            perm_str = "\n  - ".join(perms) if perms else "No specific permissions"
            return f"Skill '{self.skill_name}' requests the following permissions:\n  - {perm_str}"

        if self.consent_type == ConsentType.OPERATION_CONFIRM:
            return f"Skill '{self.skill_name}' wants to perform: {self.operation}"

        if self.consent_type == ConsentType.INVOKE_CONFIRM:
            side_effect = self.policy.side_effect.value if self.policy else "unknown"
            return f"Allow model to invoke '{self.skill_name}'?\n  Side effect: {side_effect}"

        return f"Consent requested for skill '{self.skill_name}'"


@dataclass
class ConsentResponse:
    """Response to a consent request."""

    result: ConsentResult
    request: ConsentRequest
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    reason: str | None = None

    def is_granted(self) -> bool:
        return self.result in (ConsentResult.GRANTED, ConsentResult.REMEMBERED)

    def to_dict(self) -> dict[str, Any]:
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
    """Protocol for consent handlers."""

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse: ...

    def check_remembered(self, request: ConsentRequest) -> ConsentResponse | None: ...


class ConsentStore(ABC):
    """Abstract base for consent persistence."""

    @abstractmethod
    def save(self, response: ConsentResponse) -> None: ...

    @abstractmethod
    def load(self, skill_name: str, consent_type: ConsentType) -> ConsentResponse | None: ...

    @abstractmethod
    def revoke(self, skill_name: str, consent_type: ConsentType | None = None) -> None: ...


class InMemoryConsentStore(ConsentStore):
    """In-memory consent store."""

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

        if response.expires_at and datetime.now(UTC) > response.expires_at:
            del self._consents[key]
            return None

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
            keys_to_remove = [k for k in self._consents if k[0] == skill_name]
            for key in keys_to_remove:
                del self._consents[key]


class FileConsentStore(ConsentStore):
    """File-based consent store."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path.home() / ".houyi" / "consent.json"
        self._path = Path(path)
        self._consents: dict[str, dict[str, Any]] = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._consents = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load consent store: %s", e)
                self._consents = {}

    def _save_to_file(self) -> None:
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

        expires_at = data.get("expires_at")
        if expires_at:
            expires = datetime.fromisoformat(expires_at)
            if datetime.now(UTC) > expires:
                del self._consents[key]
                self._save_to_file()
                return None

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
    """Manages consent requests and responses."""

    def __init__(
        self,
        handler: ConsentHandler | None = None,
        store: ConsentStore | None = None,
        interactive: bool = True,
    ) -> None:
        self._handler = handler
        self._store = store or InMemoryConsentStore()
        self._interactive = interactive
        self._audit_log: list[ConsentResponse] = []

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
        remembered = self._store.load(request.skill_name, request.consent_type)
        if remembered:
            self._audit(remembered)
            return remembered

        if not self._interactive:
            response = ConsentResponse(
                result=ConsentResult.NOT_INTERACTIVE,
                request=request,
                reason="Consent cannot be requested in non-interactive mode",
            )
            self._audit(response)
            return response

        if not self._handler:
            response = ConsentResponse(
                result=ConsentResult.DENIED,
                request=request,
                reason="No consent handler configured",
            )
            self._audit(response)
            return response

        response = await self._handler.request_consent(request)

        if response.is_granted() and request.remember:
            self._store.save(response)

        self._audit(response)
        return response

    def check_permission(self, skill_name: str) -> bool:
        response = self._store.load(skill_name, ConsentType.PERMISSION_GRANT)
        return response is not None and response.is_granted()

    def revoke_consent(
        self,
        skill_name: str,
        consent_type: ConsentType | None = None,
    ) -> None:
        self._store.revoke(skill_name, consent_type)
        logger.info("Revoked consent for skill '%s' (type=%s)", skill_name, consent_type)

    def get_audit_log(self) -> list[ConsentResponse]:
        return list(self._audit_log)

    def export_audit_log(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entries = [r.to_dict() for r in self._audit_log]
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)

        logger.info("Exported %d consent audit entries to %s", len(entries), path)

    def _audit(self, response: ConsentResponse) -> None:
        self._audit_log.append(response)

        if response.result in (ConsentResult.GRANTED, ConsentResult.DENIED):
            logger.info(
                "Consent %s: skill=%s, type=%s",
                response.result.value,
                response.request.skill_name,
                response.request.consent_type.value,
            )


class CLIConsentHandler:
    """Simple CLI-based consent handler."""

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
        print("\n" + "=" * 50)  # noqa: T201
        print("CONSENT REQUEST")  # noqa: T201
        print("=" * 50)  # noqa: T201
        print(request.describe())  # noqa: T201
        print()  # noqa: T201

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
        if response_str in ("r", "remember"):
            request.remember = True
            return ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
            )
        return ConsentResponse(
            result=ConsentResult.DENIED,
            request=request,
        )

    def check_remembered(self, request: ConsentRequest) -> ConsentResponse | None:
        return None


class PolicyBasedConsentHandler:
    """Policy-based consent handler for non-interactive scenarios."""

    def __init__(
        self,
        auto_grant_skills: set[str] | None = None,
        auto_deny_skills: set[str] | None = None,
        default_grant: bool = False,
    ) -> None:
        self._auto_grant = auto_grant_skills or set()
        self._auto_deny = auto_deny_skills or set()
        self._default_grant = default_grant

    async def request_consent(self, request: ConsentRequest) -> ConsentResponse:
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
        return None


__all__ = [
    "CLIConsentHandler",
    "ConsentHandler",
    "ConsentManager",
    "ConsentRequest",
    "ConsentResponse",
    "ConsentResult",
    "ConsentStore",
    "ConsentType",
    "FileConsentStore",
    "InMemoryConsentStore",
    "PolicyBasedConsentHandler",
]
