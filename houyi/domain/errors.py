"""Domain exception hierarchy for HouYi SDK."""


class HouYiError(Exception):
    """Base exception for all HouYi SDK errors."""


class DependencyMissingError(HouYiError):
    """A required optional dependency is not installed."""


__all__ = ["DependencyMissingError", "HouYiError"]
