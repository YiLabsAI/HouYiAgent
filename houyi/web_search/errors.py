from __future__ import annotations


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


class DependencyMissingError(WebSearchError):
    pass


class ContentFetchError(WebSearchError):
    pass
