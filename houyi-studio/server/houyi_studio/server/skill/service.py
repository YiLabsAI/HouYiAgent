"""Skill service facade (SOLID refactored).

``SkillService`` is a **thin facade** that delegates to three single-
responsibility collaborators:

- ``SkillLoader``       — loading / unloading / validation
- ``SkillSerializer``   — SkillSpec → dict conversion
- ``DryRunValidator``   — static + live invocation checks

The class itself only adds *metrics*, *consent management*, and
*configure_skill* — concerns that genuinely cut across the above
three but are small enough to keep inline.

Dependency-Inversion: all collaborators are injected via the constructor.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY, SkillRegistry

from .dry_run import DryRunValidator
from .loader import SkillLoader
from .serializer import VALID_POLICY_ACTIONS, SkillSerializer

if TYPE_CHECKING:
    from houyi.core.skill.consent import ConsentManager
    from houyi.core.skill.metrics import MetricsStore
    from houyi.core.skill.policy import PolicyEnforcer

logger = logging.getLogger(__name__)

# ── Metrics field names ──────────────────────────────────────────────

_METRICS_FIELDS = (
    "total_calls",
    "success_count",
    "failure_count",
    "avg_latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
    "success_rate",
)


def _empty_metrics(skill_name: str) -> dict[str, Any]:
    """Return an empty metrics dict (no MetricsStore configured)."""
    result: dict[str, Any] = {"skill_name": skill_name}
    for f in _METRICS_FIELDS:
        result[f] = 0.0 if "latency" in f or f == "success_rate" else 0
    result["last_invoked"] = None
    return result


# ── Consent data-class ───────────────────────────────────────────────


@dataclass
class PendingConsentRequest:
    """Tracks a pending consent request awaiting UI response."""

    request_id: str
    skill_name: str
    tool_name: str
    reason: str
    permissions: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event: asyncio.Event = field(default_factory=asyncio.Event)
    granted: bool = False
    remember: bool = False


# ══════════════════════════════════════════════════════════════════════
# Facade
# ══════════════════════════════════════════════════════════════════════


class SkillService:
    """Facade that composes loader, serializer, and dry-run validator.

    External callers (command handlers, startup hooks) import only this
    class — internal refactoring does not change the public API.
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        metrics_store: MetricsStore | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        consent_manager: ConsentManager | None = None,
    ) -> None:
        self._registry = registry or DEFAULT_SKILL_REGISTRY
        self._metrics_store = metrics_store
        self._policy_enforcer = policy_enforcer
        self._consent_manager = consent_manager

        # Compose single-responsibility collaborators (DIP)
        self._loader = SkillLoader(self._registry)
        self._serializer = SkillSerializer()
        self._dry_run_validator = DryRunValidator(self._registry, policy_enforcer)

        self._pending_consents: dict[str, PendingConsentRequest] = {}

    # ── Governance component accessors ────────────────────────────

    @property
    def policy_enforcer(self) -> PolicyEnforcer | None:
        return self._policy_enforcer

    @property
    def consent_manager(self) -> ConsentManager | None:
        return self._consent_manager

    @property
    def metrics_store(self) -> MetricsStore | None:
        return self._metrics_store

    # ── Read operations (delegate to serializer) ──────────────────

    def list_skills(self) -> list[dict[str, Any]]:
        return [self._serializer.to_summary(s) for s in self._registry.list()]

    def get_skill_detail(self, skill_name: str) -> dict[str, Any] | None:
        skill = self._registry.get(skill_name)
        if not skill:
            return None
        return self._serializer.to_detail(skill)

    def get_skill_metrics(self, skill_name: str) -> dict[str, Any]:
        if not self._metrics_store:
            return _empty_metrics(skill_name)
        agg = self._metrics_store.aggregate(skill_name)
        result: dict[str, Any] = {"skill_name": skill_name}
        for f in _METRICS_FIELDS:
            result[f] = getattr(agg, f)
        result["last_invoked"] = agg.last_invoked.isoformat() if agg.last_invoked else None
        return result

    # ── Write operations (delegate to loader) ─────────────────────

    def is_skill_loaded(self, skill_name: str) -> bool:
        return self._loader.is_loaded(skill_name)

    def load_skill(self, source: str) -> tuple[bool, str, str | None]:
        return self._loader.load(source)

    def unload_skill(self, skill_name: str) -> tuple[bool, str | None]:
        return self._loader.unload(skill_name)

    def configure_skill(
        self,
        skill_name: str,
        *,
        policy_action: str | None = None,
        auto_invoke: bool | None = None,
    ) -> tuple[bool, str | None]:
        """Update runtime configuration for a skill."""
        skill = self._registry.get(skill_name)
        if not skill:
            return False, f"Skill not found: {skill_name}"

        changes: list[str] = []

        from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

        ip = getattr(skill, "invocation_policy", None)
        if ip is None or isinstance(ip, dict):
            ip = InvocationPolicy.from_dict(ip) if isinstance(ip, dict) else InvocationPolicy()
            skill.invocation_policy = ip

        if policy_action is not None:
            if policy_action not in VALID_POLICY_ACTIONS:
                return False, (
                    f"Invalid policy_action '{policy_action}'. "
                    f"Must be one of: {', '.join(sorted(VALID_POLICY_ACTIONS))}"
                )
            ip.model_auto_invoke = ModelAutoInvoke(policy_action)
            changes.append(f"policy → {policy_action}")
        elif auto_invoke is not None:
            ip.model_auto_invoke = ModelAutoInvoke.ALLOW if auto_invoke else ModelAutoInvoke.DENY
            changes.append(f"auto_invoke → {auto_invoke}")

        if not changes:
            return False, "No configuration changes specified"

        logger.info("Configured skill '%s': %s", skill_name, ", ".join(changes))
        return True, None

    # ── Dry-run (delegate to validator) ───────────────────────────

    async def dry_run(
        self,
        skill_name: str,
        tool_name: str,
        input_data: dict[str, Any],
        live: bool = False,
    ) -> dict[str, Any]:
        return await self._dry_run_validator.validate(
            skill_name,
            tool_name,
            input_data,
            live=live,
        )

    # ── Consent management ────────────────────────────────────────

    def create_consent_request(
        self,
        skill_name: str,
        tool_name: str,
        reason: str,
        permissions: list[str],
    ) -> str:
        request_id = f"consent_{uuid.uuid4().hex[:8]}"
        self._pending_consents[request_id] = PendingConsentRequest(
            request_id=request_id,
            skill_name=skill_name,
            tool_name=tool_name,
            reason=reason,
            permissions=permissions,
        )
        return request_id

    async def wait_for_consent(self, request_id: str, timeout: float = 60.0) -> tuple[bool, bool]:
        if request_id not in self._pending_consents:
            return False, False
        req = self._pending_consents[request_id]
        try:
            await asyncio.wait_for(req.event.wait(), timeout=timeout)
            return req.granted, req.remember
        except TimeoutError:
            logger.warning("Consent request %s timed out", request_id)
            return False, False
        finally:
            self._pending_consents.pop(request_id, None)

    def respond_to_consent(self, request_id: str, granted: bool, remember: bool = False) -> bool:
        if request_id not in self._pending_consents:
            return False
        req = self._pending_consents[request_id]
        req.granted = granted
        req.remember = remember
        req.event.set()
        return True


# ── Module-level singleton ────────────────────────────────────────────

_default_skill_service: SkillService | None = None


def get_skill_service() -> SkillService:
    global _default_skill_service
    if _default_skill_service is None:
        _default_skill_service = SkillService()
    return _default_skill_service


def set_skill_service(service: SkillService) -> None:
    global _default_skill_service
    _default_skill_service = service


__all__ = [
    "DryRunValidator",
    "PendingConsentRequest",
    "SkillLoader",
    "SkillSerializer",
    "SkillService",
    "get_skill_service",
    "set_skill_service",
]
