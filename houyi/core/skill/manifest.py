"""SimpleSkill Manifest parser and validator.

This module implements Layer A (Manifest) of SimpleSkill:
- simpleskill.json schema parsing
- Contributions index (tools/skills/resources)
- Activation events
- Trust and provider information

Reference: SimpleSkill Specification v0.1 Section 3 (Layer A: Manifest)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from houyi.core.skill.policy import InvocationPolicy, Permissions

logger = logging.getLogger(__name__)


class ActivationEventType(str, Enum):
    """Activation event types for lazy loading."""

    ON_COMMAND = "onCommand"
    """Activate when a specific command is invoked."""

    ON_LANGUAGE = "onLanguage"
    """Activate for a specific file language."""

    ON_FILE_SYSTEM = "onFileSystem"
    """Activate when certain file patterns are accessed."""

    ON_STARTUP = "onStartup"
    """Activate immediately on host startup."""

    ON_TOOL_USE = "onToolUse"
    """Activate when a specific tool is about to be used."""

    ALWAYS = "*"
    """Always activated."""


@dataclass
class ActivationEvent:
    """Activation event specification."""

    event_type: ActivationEventType
    """Type of activation event."""

    pattern: str | None = None
    """Pattern for the event (e.g., command name, language id, file glob)."""

    @classmethod
    def parse(cls, event_str: str) -> ActivationEvent:
        """Parse activation event from string.

        Examples:
            "onCommand:planning.createPlan"
            "onLanguage:python"
            "onFileSystem:**/*.md"
            "onStartup"
            "*"
        """
        if event_str == "*":
            return cls(event_type=ActivationEventType.ALWAYS)

        if ":" in event_str:
            event_type_str, pattern = event_str.split(":", 1)
            try:
                event_type = ActivationEventType(event_type_str)
            except ValueError:
                logger.warning("Unknown activation event type: %s", event_type_str)
                event_type = ActivationEventType.ON_STARTUP
            return cls(event_type=event_type, pattern=pattern)

        try:
            event_type = ActivationEventType(event_str)
            return cls(event_type=event_type)
        except ValueError:
            logger.warning("Unknown activation event: %s", event_str)
            return cls(event_type=ActivationEventType.ON_STARTUP)


@dataclass
class ToolContribution:
    """Tool contribution specification."""

    id: str
    """Unique tool identifier."""

    description: str
    """Human-readable description."""

    input_schema: dict[str, Any] = field(default_factory=dict)
    """JSON Schema for input validation."""

    output_schema: dict[str, Any] = field(default_factory=dict)
    """JSON Schema for output validation."""

    execution: dict[str, Any] = field(default_factory=dict)
    """Execution form and entry point."""

    permissions: Permissions | None = None
    """Tool-specific permissions (cannot exceed manifest permissions)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolContribution:
        """Create from dictionary."""
        permissions = None
        if "permissions" in data:
            permissions = Permissions.from_dict(data["permissions"])

        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
            output_schema=data.get("outputSchema", {}),
            execution=data.get("execution", {}),
            permissions=permissions,
        )


@dataclass
class SkillContribution:
    """Skill contribution specification.

    A skill contribution can reference an external SKILL.md file via ``path``,
    or provide inline metadata when no path is given.
    """

    id: str
    """Unique skill identifier."""

    description: str
    """Human-readable description."""

    path: str = ""
    """Relative path to SKILL.md (empty if metadata is inline)."""

    invocation_policy: InvocationPolicy = field(default_factory=InvocationPolicy)
    """Invocation policy for this skill."""

    tool_refs: list[str] = field(default_factory=list)
    """Tools this skill may orchestrate."""

    resources: list[str] = field(default_factory=list)
    """Resources this skill may access."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillContribution:
        """Create from dictionary."""
        policy = InvocationPolicy()
        if "invocationPolicy" in data:
            policy = InvocationPolicy.from_dict(data["invocationPolicy"])

        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            path=data.get("path", ""),
            invocation_policy=policy,
            tool_refs=data.get("toolRefs", []),
            resources=data.get("resources", []),
        )


@dataclass
class ResourceContribution:
    """Resource contribution specification."""

    id: str
    """Unique resource identifier."""

    uri: str
    """Resource URI."""

    name: str
    """Human-readable name."""

    description: str = ""
    """Description of the resource."""

    mime_type: str | None = None
    """MIME type of the resource content."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceContribution:
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mime_type=data.get("mimeType"),
        )


@dataclass
class Contributions:
    """Contributions index from manifest."""

    tools: list[ToolContribution] = field(default_factory=list)
    """Tool contributions."""

    skills: list[SkillContribution] = field(default_factory=list)
    """Skill contributions."""

    resources: list[ResourceContribution] = field(default_factory=list)
    """Resource contributions."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contributions:
        """Create from dictionary."""
        tools = [ToolContribution.from_dict(t) for t in data.get("tools", [])]
        skills = [SkillContribution.from_dict(s) for s in data.get("skills", [])]
        resources = [ResourceContribution.from_dict(r) for r in data.get("resources", [])]
        return cls(tools=tools, skills=skills, resources=resources)


@dataclass
class Trust:
    """Trust and provenance information."""

    provider: str = ""
    """Provider identifier (e.g., "github:user/repo")."""

    signature: str | None = None
    """Optional cryptographic signature."""

    checksum: str | None = None
    """Content checksum for integrity verification."""

    verified: bool = False
    """Whether the package has been verified."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trust:
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", ""),
            signature=data.get("signature"),
            checksum=data.get("checksum"),
            verified=data.get("verified", False),
        )


@dataclass
class SkillManifest:
    """SimpleSkill manifest (simpleskill.json).

    Reference: SimpleSkill Specification v0.1 Section 3.1-3.7
    """

    id: str
    """Unique extension identifier (e.g., "houyi.planning-with-files")."""

    version: str
    """SemVer version string."""

    name: str
    """Human-readable name."""

    description: str
    """Human-readable description."""

    engines: dict[str, str] = field(default_factory=dict)
    """Required host engines (e.g., {"host": "houyi>=0.1"})."""

    activation_events: list[ActivationEvent] = field(default_factory=list)
    """Events that trigger extension activation."""

    contributions: Contributions = field(default_factory=Contributions)
    """Tools, skills, and resources contributed by this extension."""

    permissions: Permissions = field(default_factory=Permissions)
    """Permission requirements."""

    trust: Trust = field(default_factory=Trust)
    """Trust and provenance information."""

    main: str | None = None
    """Main entry point (for in-process execution)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    manifest_path: Path | None = None
    """Path to the manifest file (if loaded from file)."""

    @classmethod
    def from_file(cls, path: str | Path) -> SkillManifest:
        """Load manifest from file.

        Args:
            path: Path to simpleskill.json or package.json

        Returns:
            Parsed SkillManifest

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parsing fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls.from_dict(data)
        manifest.manifest_path = path
        return manifest

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        """Create manifest from dictionary.

        Args:
            data: Parsed JSON data

        Returns:
            SkillManifest instance
        """
        # Parse activation events
        activation_events = []
        for event_str in data.get("activationEvents", []):
            activation_events.append(ActivationEvent.parse(event_str))

        # Parse contributions
        contributions = Contributions()
        if "contributions" in data:
            contributions = Contributions.from_dict(data["contributions"])

        # Parse permissions
        permissions = Permissions()
        if "permissions" in data:
            permissions = Permissions.from_dict(data["permissions"])

        # Parse trust
        trust = Trust()
        if "trust" in data:
            trust = Trust.from_dict(data["trust"])

        return cls(
            id=data.get("id", ""),
            version=data.get("version", "0.0.0"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            engines=data.get("engines", {}),
            activation_events=activation_events,
            contributions=contributions,
            permissions=permissions,
            trust=trust,
            main=data.get("main"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "engines": self.engines,
            "activationEvents": [
                f"{e.event_type.value}:{e.pattern}" if e.pattern else e.event_type.value
                for e in self.activation_events
            ],
            "contributions": {
                "tools": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                        "outputSchema": t.output_schema,
                        "execution": t.execution,
                    }
                    for t in self.contributions.tools
                ],
                "skills": [
                    {
                        "id": s.id,
                        "description": s.description,
                        **({"path": s.path} if s.path else {}),
                        "invocationPolicy": s.invocation_policy.to_dict(),
                        "toolRefs": s.tool_refs,
                        "resources": s.resources,
                    }
                    for s in self.contributions.skills
                ],
                "resources": [
                    {
                        "id": r.id,
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description,
                        "mimeType": r.mime_type,
                    }
                    for r in self.contributions.resources
                ],
            },
            "permissions": self.permissions.to_dict(),
            "trust": {
                "provider": self.trust.provider,
                "signature": self.trust.signature,
                "checksum": self.trust.checksum,
                "verified": self.trust.verified,
            },
            "main": self.main,
            "metadata": self.metadata,
        }

    def validate(self) -> list[str]:
        """Validate manifest and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Required fields
        if not self.id:
            errors.append("Missing required field: id")
        if not self.version:
            errors.append("Missing required field: version")
        if not self.name:
            errors.append("Missing required field: name")

        # ID format (should be like "publisher.extension-name")
        if self.id and not re.match(r"^[\w-]+\.[\w-]+$", self.id):
            errors.append(f"Invalid id format: {self.id} (expected: publisher.extension-name)")

        # Version format (SemVer)
        if self.version and not re.match(r"^\d+\.\d+\.\d+", self.version):
            errors.append(f"Invalid version format: {self.version} (expected: SemVer)")

        # Check contributions have IDs
        for tool in self.contributions.tools:
            if not tool.id:
                errors.append("Tool contribution missing id")
        for skill in self.contributions.skills:
            if not skill.id:
                errors.append("Skill contribution missing id")

        return errors

    def get_tool(self, tool_id: str) -> ToolContribution | None:
        """Get tool contribution by ID."""
        for tool in self.contributions.tools:
            if tool.id == tool_id:
                return tool
        return None

    def get_skill(self, skill_id: str) -> SkillContribution | None:
        """Get skill contribution by ID."""
        for skill in self.contributions.skills:
            if skill.id == skill_id:
                return skill
        return None

    def matches_activation_event(
        self,
        event_type: ActivationEventType,
        context: str | None = None,
    ) -> bool:
        """Check if manifest should be activated for given event.

        Args:
            event_type: Type of activation event
            context: Context for the event (command name, language id, etc.)

        Returns:
            True if manifest should be activated
        """
        for event in self.activation_events:
            # Always matches
            if event.event_type == ActivationEventType.ALWAYS:
                return True

            # Type matches
            if event.event_type != event_type:
                continue

            # No pattern means any context matches
            if not event.pattern:
                return True

            # Pattern matching
            if context:
                # Glob-style matching for file patterns
                if event_type == ActivationEventType.ON_FILE_SYSTEM:
                    if self._glob_match(event.pattern, context):
                        return True
                # Exact match for others
                elif event.pattern == context:
                    return True

        return False

    @staticmethod
    def _glob_match(pattern: str, path: str) -> bool:
        """Simple glob matching for file paths."""
        # Convert glob to regex
        regex = pattern.replace(".", r"\.")
        regex = regex.replace("**", ".*")
        regex = regex.replace("*", "[^/]*")
        regex = f"^{regex}$"
        return bool(re.match(regex, path))


class ManifestRegistry:
    """Registry for managing multiple manifests."""

    def __init__(self) -> None:
        self._manifests: dict[str, SkillManifest] = {}
        self._tool_index: dict[str, str] = {}  # tool_id -> manifest_id
        self._skill_index: dict[str, str] = {}  # skill_id -> manifest_id

    def register(self, manifest: SkillManifest) -> None:
        """Register a manifest.

        Args:
            manifest: Manifest to register

        Raises:
            ValueError: If manifest ID already registered
        """
        if manifest.id in self._manifests:
            raise ValueError(f"Manifest already registered: {manifest.id}")

        self._manifests[manifest.id] = manifest

        # Index tools and skills
        for tool in manifest.contributions.tools:
            full_id = f"{manifest.id}.{tool.id}"
            self._tool_index[full_id] = manifest.id
            self._tool_index[tool.id] = manifest.id  # Also index short ID

        for skill in manifest.contributions.skills:
            full_id = f"{manifest.id}.{skill.id}"
            self._skill_index[full_id] = manifest.id
            self._skill_index[skill.id] = manifest.id

        logger.debug("Registered manifest: %s (v%s)", manifest.id, manifest.version)

    def unregister(self, manifest_id: str) -> None:
        """Unregister a manifest."""
        manifest = self._manifests.pop(manifest_id, None)
        if not manifest:
            return

        # Remove from indices
        for tool in manifest.contributions.tools:
            self._tool_index.pop(f"{manifest.id}.{tool.id}", None)
            self._tool_index.pop(tool.id, None)

        for skill in manifest.contributions.skills:
            self._skill_index.pop(f"{manifest.id}.{skill.id}", None)
            self._skill_index.pop(skill.id, None)

    def get_manifest(self, manifest_id: str) -> SkillManifest | None:
        """Get manifest by ID."""
        return self._manifests.get(manifest_id)

    def get_manifest_for_tool(self, tool_id: str) -> SkillManifest | None:
        """Get manifest that provides a tool."""
        manifest_id = self._tool_index.get(tool_id)
        if manifest_id:
            return self._manifests.get(manifest_id)
        return None

    def get_manifest_for_skill(self, skill_id: str) -> SkillManifest | None:
        """Get manifest that provides a skill."""
        manifest_id = self._skill_index.get(skill_id)
        if manifest_id:
            return self._manifests.get(manifest_id)
        return None

    def list_manifests(self) -> list[SkillManifest]:
        """List all registered manifests."""
        return list(self._manifests.values())

    def find_by_activation(
        self,
        event_type: ActivationEventType,
        context: str | None = None,
    ) -> list[SkillManifest]:
        """Find manifests that should be activated for an event."""
        return [
            m for m in self._manifests.values() if m.matches_activation_event(event_type, context)
        ]
