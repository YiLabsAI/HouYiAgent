"""Entity resolution for memory extraction.

This module provides pluggable entity resolution strategies that map
turn/speaker context to a canonical entity identifier used as the
subject in extracted atomic facts.

Design goals:
- Framework ships with sensible defaults (single-user "user")
- Built-in resolvers for common patterns (multi-speaker dialogue, namespaced IDs)
- Composable: chain multiple resolvers via pipeline (first-match or transform-chain)
- Easy to extend for custom identity schemes (user_id, agent_id, etc.)
- No breaking changes: resolver is optional, defaults work out-of-box

Resolver composition patterns:
- FirstMatchChain: try resolvers in order until one returns non-default
- TransformPipeline: each resolver transforms the output of the previous
- ConditionalResolver: route to different resolvers based on turn metadata
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TurnContext:
    """Context for a single turn that may contain extractable facts.

    This is a framework-neutral representation of a conversation turn,
    decoupled from LoCoMo or any specific benchmark format.
    """

    text: str
    """The raw text content of the turn."""

    speaker_id: str | None = None
    """Opaque speaker identifier from the source system.

    Examples:
    - Role indicator: "speaker_a", "speaker_b"
    - Display name: "Alice", "Bob"
    - Stable ID: "user_123", "agent_456"
    - UUID: "550e8400-e29b-41d4-a716-446655440000"
    """

    session_id: str | None = None
    """Session/conversation identifier."""

    turn_id: str | None = None
    """Unique identifier for this specific turn."""

    metadata: dict[str, Any] | None = None
    """Additional context from the source system."""


class EntityResolver(Protocol):
    """Protocol for resolving a turn's speaker to a canonical entity id."""

    def resolve(self, turn: TurnContext) -> str:
        """Return the canonical entity identifier for the turn's speaker.

        The returned string is used as the 'subject' field in extracted
        AtomicFacts. Common patterns:
        - "user" for single-user scenarios (default)
        - "Caroline", "Jon" for multi-speaker benchmarks
        - "user_123", "agent_456" for production systems

        Args:
            turn: The turn context containing speaker and metadata.

        Returns:
            A non-empty entity identifier.
        """
        ...


class DefaultEntityResolver:
    """Default single-user resolver.

    Always returns "user". This is the framework's out-of-box default,
    suitable for:
    - Single-user chatbots
    - Simple agent scenarios where entity identity isn't relevant
    - Backwards compatibility (existing code continues to work)
    """

    def resolve(self, turn: TurnContext) -> str:
        return "user"


class SpeakerNameResolver:
    """Resolver that uses the speaker_id directly as the entity.

    Suitable for:
    - Multi-speaker dialogues like LoCoMo where speaker_id is already
      the canonical name ("Caroline", "Jon")
    - Systems where speaker identifiers are stable and meaningful

    Falls back to "user" if speaker_id is missing.
    """

    def resolve(self, turn: TurnContext) -> str:
        if turn.speaker_id and turn.speaker_id.strip():
            return turn.speaker_id.strip()
        return "user"


class RoleBasedEntityResolver:
    """Resolver for role-based multi-speaker scenarios.

    Maps role indicators (e.g., "speaker_a", "speaker_b") to canonical
    entity names. Also handles cases where the speaker_id is already
    the canonical name.

    Common use cases:
    - Multi-speaker dialogue datasets with role indicators
    - Interview transcripts (interviewer/interviewee)
    - Customer service logs (agent/customer)

    Example mapping:
    - "speaker_a" -> "Alice" (primary speaker)
    - "speaker_b" -> "Bob" (secondary speaker)
    - "Alice" -> "Alice" (already canonical)
    """

    def __init__(self, primary: str, secondary: str | None = None) -> None:
        """Initialize with primary and optional secondary speaker names.

        Args:
            primary: The canonical name for the primary speaker (e.g., "Alice")
            secondary: Optional name for the secondary speaker (e.g., "Bob")
        """
        self._primary = primary
        self._secondary = secondary
        self._role_to_name = {"speaker_a": primary, "primary": primary}
        if secondary:
            self._role_to_name["speaker_b"] = secondary
            self._role_to_name["secondary"] = secondary

    def resolve(self, turn: TurnContext) -> str:
        """Resolve speaker_id to canonical entity name."""
        spk = (turn.speaker_id or "").strip()

        # Case 1: speaker_id is already the canonical name
        if spk == self._primary:
            return self._primary
        if spk == self._secondary:
            return self._secondary or "user"

        # Case 2: speaker_id is a role indicator
        if spk in self._role_to_name:
            return self._role_to_name[spk]

        # Case 3: unknown or empty, use primary as default
        if spk:
            # Try fuzzy match against known names
            for _role, name in self._role_to_name.items():
                if name and name.lower() in spk.lower():
                    return name

        return self._primary


class NamespacePrefixedResolver:
    """Resolver that prefixes entity with namespace for multi-tenant scenarios.

    Useful for:
    - Multi-tenant SaaS where user "alice" in workspace "ws_1" needs
      to be distinct from "alice" in workspace "ws_2"
    - Agent systems with multiple distinct agent instances

    Format: "{namespace}::{entity}" or custom via template.
    """

    def __init__(self, namespace: str, template: str = "{ns}::{entity}") -> None:
        """Initialize with namespace and optional template.

        Args:
            namespace: The namespace to prefix (e.g., workspace_id)
            template: Format string with {ns} and {entity} placeholders
        """
        self._namespace = namespace
        self._template = template
        self._inner: EntityResolver = DefaultEntityResolver()

    def with_inner(self, resolver: EntityResolver) -> NamespacePrefixedResolver:
        """Chain with another resolver for the base entity name."""
        self._inner = resolver
        return self

    def resolve(self, turn: TurnContext) -> str:
        entity = self._inner.resolve(turn)
        return self._template.format(ns=self._namespace, entity=entity)


# ---------------------------------------------------------------------------
# Resolver composition (pipeline/chain patterns)
# ---------------------------------------------------------------------------


class FirstMatchChain:
    """Chain multiple resolvers, returning the first non-default result.

    Useful when you want to try multiple resolution strategies in order:
    1. Check if speaker_id is a known user ID
    2. Fall back to extracting name from text
    3. Finally default to "user"

    Example:
        resolver = FirstMatchChain([
            UserIdResolver(db),      # returns "user_123" or None
            NameFromTextResolver(),  # returns "Alice" or None
            DefaultEntityResolver(), # always returns "user"
        ])
    """

    def __init__(
        self,
        resolvers: list[EntityResolver],
        default: str = "user",
        skip_default: str | None = "user",
    ) -> None:
        """Initialize with a list of resolvers.

        Args:
            resolvers: Ordered list of resolvers to try
            default: Fallback if all resolvers return skip_default
            skip_default: Value to treat as "not resolved" (usually "user")
        """
        self._resolvers = resolvers
        self._default = default
        self._skip_default = skip_default

    def resolve(self, turn: TurnContext) -> str:
        """Try resolvers in order until one returns non-default."""
        for resolver in self._resolvers:
            result = resolver.resolve(turn)
            if self._skip_default is None or result != self._skip_default:
                return result
        return self._default


class TransformPipeline:
    """Pipeline where each resolver transforms the output of the previous.

    Useful for layering transformations:
    - Base: resolve speaker_id to user name
    - Layer 1: apply namespace prefix
    - Layer 2: apply alias mapping

    Example:
        resolver = TransformPipeline([
            SpeakerNameResolver(),                    # "Alice"
            NamespacePrefixedResolver("ws_1"),        # "ws_1::Alice"
            AliasMappingResolver({"ws_1::Alice": "A"}) # "A"
        ])
    """

    def __init__(self, resolvers: list[EntityResolver]) -> None:
        """Initialize with ordered list of transforming resolvers.

        Each resolver receives the turn context (with metadata containing
        the previous result in "_prev_entity").
        """
        self._resolvers = resolvers

    def resolve(self, turn: TurnContext) -> str:
        """Pass result through each resolver in sequence."""
        current = turn
        for resolver in self._resolvers:
            result = resolver.resolve(current)
            # Create new context with previous result for next resolver
            metadata = dict(current.metadata or {})
            metadata["_prev_entity"] = result
            current = TurnContext(
                text=current.text,
                speaker_id=current.speaker_id,
                session_id=current.session_id,
                turn_id=current.turn_id,
                metadata=metadata,
            )
        return current.metadata.get("_prev_entity", "user") if current.metadata else "user"


class ConditionalResolver:
    """Route to different resolvers based on turn metadata conditions.

    Useful for multi-tenant scenarios where resolution strategy varies
    by workspace, user type, or session characteristics.

    Example:
        resolver = ConditionalResolver(
            condition=lambda turn: turn.metadata.get("workspace_type") == "personal",
            if_true=SpeakerNameResolver(),
            if_false=NamespacePrefixedResolver("corp", "{ns}::{entity}")
        )
    """

    def __init__(
        self,
        condition: Callable[[TurnContext], bool],
        if_true: EntityResolver,
        if_false: EntityResolver,
    ) -> None:
        """Initialize with condition and two resolvers.

        Args:
            condition: Function that receives TurnContext and returns bool
            if_true: Resolver used when condition is True
            if_false: Resolver used when condition is False
        """
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    def resolve(self, turn: TurnContext) -> str:
        """Route to appropriate resolver based on condition."""
        resolver = self._if_true if self._condition(turn) else self._if_false
        return resolver.resolve(turn)


class AliasMappingResolver:
    """Resolver that applies a static alias mapping.

    Useful for:
    - Normalizing entity names ("Bob" -> "Robert")
    - Mapping external IDs to internal IDs
    - Consolidating multiple identifiers for the same entity
    """

    def __init__(
        self,
        alias_map: dict[str, str],
        inner: EntityResolver | None = None,
        passthrough: bool = True,
    ) -> None:
        """Initialize with alias mapping.

        Args:
            alias_map: Dict mapping input -> output entity names
            inner: Optional inner resolver to get base entity first
            passthrough: If True, return original when no alias found;
                        if False, return "user" when no alias found
        """
        self._alias_map = alias_map
        self._inner = inner
        self._passthrough = passthrough

    def resolve(self, turn: TurnContext) -> str:
        """Apply alias mapping to resolved (or provided) entity."""
        if self._inner:
            base = self._inner.resolve(turn)
        else:
            base = turn.speaker_id or "user"

        return self._alias_map.get(base, base if self._passthrough else "user")


# ---------------------------------------------------------------------------
# Framework default instance
# ---------------------------------------------------------------------------

_DEFAULT_RESOLVER = DefaultEntityResolver()


def get_default_resolver() -> EntityResolver:
    """Return the framework's default single-user resolver."""
    return _DEFAULT_RESOLVER
