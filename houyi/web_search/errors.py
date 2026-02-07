from __future__ import annotations

from houyi.core.errors import DependencyMissingError as _CoreDependencyMissingError


class WebSearchError(Exception):
    pass


class ProviderError(WebSearchError):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderInvalidResponse(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class DependencyMissingError(WebSearchError, _CoreDependencyMissingError):
    pass


class ContentFetchError(WebSearchError):
    pass
