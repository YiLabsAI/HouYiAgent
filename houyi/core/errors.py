"""Core exception hierarchy for HouYi SDK.

All exceptions that SDK core (e.g., skill_executor) needs to catch must be
defined here. Tool plugins and extensions should subclass these base types
rather than defining independent exception trees.

Dependency rule: houyi.core depends on nothing; plugins depend on houyi.core.
"""

from __future__ import annotations


class HouYiError(Exception):
    """Base exception for all HouYi SDK errors."""

    pass


class DependencyMissingError(HouYiError):
    """A required optional dependency is not installed.

    Raised by tool plugins when an optional package (e.g., tavily-python,
    readability-lxml) is missing. The skill executor treats this as a
    deterministic error and will not retry.
    """

    pass
