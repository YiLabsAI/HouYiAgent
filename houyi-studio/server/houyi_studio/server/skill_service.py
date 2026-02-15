"""Skill service for console UI integration.

This service bridges the SDK's SimpleSkill capabilities with the Console UI,
providing skill listing, detail retrieval, metrics, and consent management.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY, SkillRegistry

if TYPE_CHECKING:
    from houyi.core.skill.consent import ConsentManager
    from houyi.core.skill.metrics import MetricsStore
    from houyi.core.skill.policy import PolicyEnforcer
    from houyi.core.skill.spec import SkillSpec

logger = logging.getLogger(__name__)


@dataclass
class PendingConsentRequest:
    """Tracks a pending consent request awaiting UI response."""

    request_id: str
    skill_name: str
    tool_name: str
    reason: str
    permissions: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event: asyncio.Event = field(default_factory=asyncio.Event)
    granted: bool = False
    remember: bool = False


class SkillService:
    """Service for managing skills in the Console UI.

    This service provides:
    - Skill listing and detail retrieval (read operations)
    - Skill loading/unloading (write operations)
    - Metrics aggregation
    - Consent request management (UI bridge)
    - Dry-run validation
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        metrics_store: MetricsStore | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        consent_manager: ConsentManager | None = None,
    ) -> None:
        """Initialize skill service.

        Args:
            registry: Skill registry (defaults to DEFAULT_SKILL_REGISTRY)
            metrics_store: Metrics store for aggregating skill metrics
            policy_enforcer: Policy enforcer for dry-run validation
            consent_manager: Consent manager for user authorization flow
        """
        self._registry = registry or DEFAULT_SKILL_REGISTRY
        self._metrics_store = metrics_store
        self._policy_enforcer = policy_enforcer
        self._consent_manager = consent_manager
        self._pending_consents: dict[str, PendingConsentRequest] = {}

    # =========================================================================
    # Governance Component Accessors
    # =========================================================================

    @property
    def policy_enforcer(self) -> PolicyEnforcer | None:
        """Read-only access to the policy enforcer for invocation governance."""
        return self._policy_enforcer

    @property
    def consent_manager(self) -> ConsentManager | None:
        """Read-only access to the consent manager for user authorization."""
        return self._consent_manager

    @property
    def metrics_store(self) -> MetricsStore | None:
        """Read-only access to the metrics store for execution telemetry."""
        return self._metrics_store

    # =========================================================================
    # Read Operations
    # =========================================================================

    def list_skills(self) -> list[dict[str, Any]]:
        """List all registered skills with summary info.

        Returns:
            List of skill summaries for UI display
        """
        summaries = []
        for skill in self._registry.list():
            summaries.append(self._to_skill_summary(skill))
        return summaries

    def get_skill_detail(self, skill_name: str) -> dict[str, Any] | None:
        """Get full detail of a specific skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Skill detail dict or None if not found
        """
        skill = self._registry.get(skill_name)
        if not skill:
            return None
        return self._to_skill_detail(skill)

    def get_skill_metrics(self, skill_name: str) -> dict[str, Any] | None:
        """Get aggregated metrics for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Metrics dict or None if metrics store not configured
        """
        if not self._metrics_store:
            return {
                "skill_name": skill_name,
                "total_calls": 0,
                "success_count": 0,
                "failure_count": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "success_rate": 0.0,
                "last_invoked": None,
            }

        aggregated = self._metrics_store.aggregate(skill_name)
        return {
            "skill_name": skill_name,
            "total_calls": aggregated.total_calls,
            "success_count": aggregated.success_count,
            "failure_count": aggregated.failure_count,
            "avg_latency_ms": aggregated.avg_latency_ms,
            "p50_latency_ms": aggregated.p50_latency_ms,
            "p99_latency_ms": aggregated.p99_latency_ms,
            "success_rate": aggregated.success_rate,
            "last_invoked": (
                aggregated.last_invoked.isoformat() if aggregated.last_invoked else None
            ),
        }

    # =========================================================================
    # Write Operations
    # =========================================================================

    def load_skill(self, source: str) -> tuple[bool, str, str | None]:
        """Load a skill from a file path, URL, or directory.

        Supported sources:
        - Local file path to SKILL.md or simpleskill.json manifest
        - URL (http:// or https://) pointing to a SKILL.md file
        - Directory path containing SKILL.md files (recursive scan)

        Args:
            source: File path, URL, or directory path

        Returns:
            Tuple of (success, skill_name or error_code, error_message or None)
        """
        # URL-based import
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_from_url(source)

        path_obj = Path(source)

        if not path_obj.exists():
            return False, "file_not_found", f"Skill source not found: {source}"

        # Directory-based import (recursive scan for SKILL.md)
        if path_obj.is_dir():
            return self._load_from_directory(source)

        try:
            if path_obj.name.endswith(".json"):
                # Load from manifest
                names = self._registry.register_from_manifest(source, overwrite=True)
                if names:
                    return True, names[0], None
                return False, "no_skills", "Manifest contains no skills"
            else:
                # Load from SKILL.md
                skill_name = self._registry.register_from_skill_file(source, overwrite=True)
                return True, skill_name, None
        except Exception as e:
            logger.exception("Failed to load skill from %s", source)
            return False, "load_failed", str(e)

    def _load_from_url(self, url: str) -> tuple[bool, str, str | None]:
        """Load a skill from a URL pointing to a SKILL.md file.

        Downloads the content, caches it locally under ~/.houyi/skill_cache/,
        and registers via SkillSpec.from_url().

        Args:
            url: URL to a SKILL.md file (e.g. GitHub raw URL)

        Returns:
            Tuple of (success, skill_name or error_code, error_message or None)
        """
        try:
            from houyi.core.skill.spec import SkillSpec

            skill = SkillSpec.from_url(url, cache=True)
            self._registry.register(skill, overwrite=True)
            logger.info("Loaded skill '%s' from URL: %s", skill.name, url)
            return True, skill.name, None
        except Exception as e:
            logger.exception("Failed to load skill from URL: %s", url)
            return False, "url_load_failed", str(e)

    def _load_from_directory(self, directory: str) -> tuple[bool, str, str | None]:
        """Load skills from a directory containing SKILL.md files.

        Recursively scans the directory for SKILL.md files and registers each.

        Args:
            directory: Path to a directory containing SKILL.md files

        Returns:
            Tuple of (success, comma-separated skill names or error_code, error or None)
        """
        try:
            names = self._registry.register_from_directory(
                directory,
                pattern="SKILL.md",
                recursive=True,
                overwrite=True,
            )
            if names:
                logger.info(
                    "Loaded %d skills from directory %s: %s",
                    len(names),
                    directory,
                    ", ".join(names),
                )
                return True, ", ".join(names), None
            return False, "no_skills", f"No SKILL.md files found in {directory}"
        except Exception as e:
            logger.exception("Failed to load skills from directory: %s", directory)
            return False, "dir_load_failed", str(e)

    def unload_skill(self, skill_name: str) -> tuple[bool, str | None]:
        """Unload a skill.

        Args:
            skill_name: Name of the skill to unload

        Returns:
            Tuple of (success, error_message or None)
        """
        if not self._registry.get(skill_name):
            return False, f"Skill not found: {skill_name}"

        self._registry.unregister(skill_name)
        return True, None

    def configure_skill(
        self,
        skill_name: str,
        *,
        policy_action: str | None = None,
        auto_invoke: bool | None = None,
    ) -> tuple[bool, str | None]:
        """Update runtime configuration for a skill.

        Args:
            skill_name: Skill to configure
            policy_action: New policy action ('allow', 'allow_with_consent', 'deny')
            auto_invoke: Whether LLM can auto-invoke this skill

        Returns:
            Tuple of (success, error_message or None)
        """
        skill = self._registry.get(skill_name)
        if not skill:
            return False, f"Skill not found: {skill_name}"

        changes: list[str] = []

        if policy_action is not None:
            valid_actions = {"allow", "allow_with_consent", "deny"}
            if policy_action not in valid_actions:
                return False, (
                    f"Invalid policy_action '{policy_action}'. "
                    f"Must be one of: {', '.join(sorted(valid_actions))}"
                )
            # Update the invocation policy on the SkillSpec
            try:
                from houyi.core.skill.policy import InvocationPolicy, PolicyAction

                if not hasattr(skill, "invocation_policy") or skill.invocation_policy is None:
                    skill.invocation_policy = InvocationPolicy()
                skill.invocation_policy.default_action = PolicyAction(policy_action)
                changes.append(f"policy → {policy_action}")
            except (ImportError, Exception) as e:
                logger.warning("Could not set policy via InvocationPolicy: %s", e)
                # Store as runtime override in extra_frontmatter
                if hasattr(skill, "extra_frontmatter") and isinstance(
                    skill.extra_frontmatter, dict
                ):
                    skill.extra_frontmatter["_runtime_policy_action"] = policy_action
                changes.append(f"policy → {policy_action}")

        if auto_invoke is not None:
            try:
                from houyi.core.skill.policy import InvocationPolicy, ModelAutoInvoke

                if not hasattr(skill, "invocation_policy") or skill.invocation_policy is None:
                    skill.invocation_policy = InvocationPolicy()
                skill.invocation_policy.model_auto_invoke = (
                    ModelAutoInvoke.ALLOW if auto_invoke else ModelAutoInvoke.DENY
                )
                changes.append(f"auto_invoke → {auto_invoke}")
            except (ImportError, Exception) as e:
                logger.warning("Could not set auto_invoke via InvocationPolicy: %s", e)
                if hasattr(skill, "extra_frontmatter") and isinstance(
                    skill.extra_frontmatter, dict
                ):
                    skill.extra_frontmatter["_runtime_auto_invoke"] = auto_invoke
                changes.append(f"auto_invoke → {auto_invoke}")

        if not changes:
            return False, "No configuration changes specified"

        logger.info("Configured skill '%s': %s", skill_name, ", ".join(changes))
        return True, None

    # =========================================================================
    # Dry-run Validation
    # =========================================================================

    def dry_run(
        self, skill_name: str, tool_name: str, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform dry-run validation without actual execution.

        Validates:
        - Schema compliance
        - Policy evaluation
        - Capability requirements

        Args:
            skill_name: Skill name
            tool_name: Tool name within the skill
            input_data: Input to validate

        Returns:
            Validation result dict
        """
        result = {
            "valid": True,
            "schema_errors": [],
            "policy_result": "allow",
            "capability_gaps": [],
            "estimated_side_effects": [],
        }

        # Check skill exists
        skill = self._registry.get(skill_name)
        if not skill:
            result["valid"] = False
            result["schema_errors"].append(f"Skill not found: {skill_name}")
            return result

        # Validate input schema if available
        if hasattr(skill, "input_schema") and skill.input_schema:
            try:
                skill.input_schema.model_validate(input_data)
            except Exception as e:
                result["valid"] = False
                result["schema_errors"].append(str(e))

        # Evaluate policy if enforcer available
        if self._policy_enforcer:
            policy_result = self._policy_enforcer.evaluate(
                skill_name, tool_name, invoked_by_model=False
            )
            result["policy_result"] = policy_result.action.value
            if policy_result.action.value == "deny":
                result["valid"] = False

        # Check side effects — Permissions is a single dataclass, not a list.
        if hasattr(skill, "permissions") and skill.permissions:
            perms = skill.permissions
            if hasattr(perms, "exec") and getattr(perms.exec, "enabled", False):
                result["estimated_side_effects"].append("exec")
            if hasattr(perms, "network") and getattr(perms.network, "enabled", False):
                result["estimated_side_effects"].append("network")
            if hasattr(perms, "filesystem") and (
                getattr(perms.filesystem, "write", False)
                or getattr(perms.filesystem, "delete", False)
            ):
                result["estimated_side_effects"].append("filesystem")

        return result

    # =========================================================================
    # Consent Management (UI Bridge)
    # =========================================================================

    def create_consent_request(
        self,
        skill_name: str,
        tool_name: str,
        reason: str,
        permissions: list[str],
    ) -> str:
        """Create a consent request for UI display.

        Args:
            skill_name: Skill requesting consent
            tool_name: Tool requiring consent
            reason: Why consent is needed
            permissions: Permissions being requested

        Returns:
            Unique request ID
        """
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
        """Wait for user consent response.

        Args:
            request_id: Consent request ID
            timeout: Timeout in seconds

        Returns:
            Tuple of (granted, remember)
        """
        if request_id not in self._pending_consents:
            return False, False

        request = self._pending_consents[request_id]
        try:
            await asyncio.wait_for(request.event.wait(), timeout=timeout)
            return request.granted, request.remember
        except asyncio.TimeoutError:
            logger.warning("Consent request %s timed out", request_id)
            return False, False
        finally:
            self._pending_consents.pop(request_id, None)

    def respond_to_consent(self, request_id: str, granted: bool, remember: bool = False) -> bool:
        """Handle consent response from UI.

        Args:
            request_id: Consent request ID
            granted: Whether consent was granted
            remember: Whether to remember this decision

        Returns:
            True if request was found and updated
        """
        if request_id not in self._pending_consents:
            return False

        request = self._pending_consents[request_id]
        request.granted = granted
        request.remember = remember
        request.event.set()
        return True

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _to_skill_summary(self, skill: SkillSpec) -> dict[str, Any]:
        """Convert SkillSpec to UI summary format."""
        tools = []
        if hasattr(skill, "tools") and skill.tools:
            tools = [t.name if hasattr(t, "name") else str(t) for t in skill.tools]
        elif hasattr(skill, "name"):
            tools = [skill.name]

        policy_action = "allow"
        if hasattr(skill, "invocation_policy") and skill.invocation_policy:
            policy_action = getattr(skill.invocation_policy, "default_action", "allow")
            if hasattr(policy_action, "value"):
                policy_action = policy_action.value

        side_effect = "none"
        if hasattr(skill, "permissions") and skill.permissions:
            perms = skill.permissions
            # Permissions is a single dataclass, not a list.
            # Derive the dominant side-effect from its fields.
            if hasattr(perms, "exec") and getattr(perms.exec, "enabled", False):
                side_effect = "exec"
            elif hasattr(perms, "network") and getattr(perms.network, "enabled", False):
                side_effect = "network"
            elif hasattr(perms, "filesystem") and (
                getattr(perms.filesystem, "write", False)
                or getattr(perms.filesystem, "delete", False)
            ):
                side_effect = "filesystem"

        return {
            "name": skill.name,
            "display_name": getattr(skill, "display_name", skill.name),
            "description": getattr(skill, "description", None),
            "tools": tools,
            "policy_action": policy_action,
            "side_effect": side_effect,
            "certification": getattr(skill, "certification", "unverified"),
        }

    def _to_skill_detail(self, skill: SkillSpec) -> dict[str, Any]:
        """Convert SkillSpec to full detail format."""
        summary = self._to_skill_summary(skill)

        # Add full tool schemas
        tools = []
        if hasattr(skill, "tools") and skill.tools:
            for tool in skill.tools:
                tool_info = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", None),
                }
                if hasattr(tool, "input_schema"):
                    tool_info["input_schema"] = (
                        tool.input_schema.model_json_schema()
                        if hasattr(tool.input_schema, "model_json_schema")
                        else {}
                    )
                tools.append(tool_info)
        else:
            tools.append(
                {
                    "name": skill.name,
                    "description": getattr(skill, "description", None),
                    "input_schema": (
                        skill.input_schema.model_json_schema()
                        if hasattr(skill, "input_schema")
                        and hasattr(skill.input_schema, "model_json_schema")
                        else {}
                    ),
                }
            )

        # Add permissions — Permissions is a single dataclass, convert via describe().
        permissions = []
        if hasattr(skill, "permissions") and skill.permissions:
            perms = skill.permissions
            if hasattr(perms, "describe"):
                # Use the describe() method to get human-readable permission lines
                for desc in perms.describe():
                    permissions.append({"name": desc, "description": desc, "is_sensitive": True})
            elif isinstance(perms, dict):
                for k, v in perms.items():
                    permissions.append({"name": k, "description": str(v), "is_sensitive": False})

        # Add policy detail
        policy = {}
        if hasattr(skill, "invocation_policy") and skill.invocation_policy:
            ip = skill.invocation_policy
            policy = {
                "default_action": getattr(ip, "default_action", "allow"),
                "model_auto_invoke": getattr(ip, "model_auto_invoke", True),
                "require_consent_for": getattr(ip, "require_consent_for", []),
            }
            if hasattr(policy["default_action"], "value"):
                policy["default_action"] = policy["default_action"].value

        # Add hooks
        hooks = []
        if hasattr(skill, "hooks") and skill.hooks:
            hooks = [getattr(h, "hook_type", str(h)) for h in skill.hooks]

        return {
            **summary,
            "version": getattr(skill, "version", None) or "0.0.0",
            "author": getattr(skill, "author", None),
            "tools": tools,
            "permissions": permissions,
            "policy": policy,
            "hooks": hooks,
        }


# Default service instance
_default_skill_service: SkillService | None = None


def get_skill_service() -> SkillService:
    """Get the default skill service instance."""
    global _default_skill_service
    if _default_skill_service is None:
        _default_skill_service = SkillService()
    return _default_skill_service


def set_skill_service(service: SkillService) -> None:
    """Set the default skill service instance."""
    global _default_skill_service
    _default_skill_service = service


__all__ = [
    "PendingConsentRequest",
    "SkillService",
    "get_skill_service",
    "set_skill_service",
]
