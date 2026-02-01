"""Unit tests for web search errors."""

from __future__ import annotations

import pytest

from houyi.web_search import errors


def test_web_search_error_hierarchy() -> None:
    """Ensure custom errors inherit from WebSearchError."""

    assert issubclass(errors.DependencyMissingError, errors.WebSearchError)
    assert issubclass(errors.ProviderAuthError, errors.WebSearchError)
    assert issubclass(errors.ProviderTimeoutError, errors.WebSearchError)
    assert issubclass(errors.ProviderRateLimitError, errors.WebSearchError)
    assert issubclass(errors.ProviderInvalidResponse, errors.WebSearchError)
    assert issubclass(errors.ContentFetchError, errors.WebSearchError)


def test_web_search_error_message() -> None:
    """Errors should preserve message text."""

    error = errors.ProviderAuthError("missing")
    with pytest.raises(errors.ProviderAuthError) as exc:
        raise error
    assert "missing" in str(exc.value)


def test_web_search_error_instantiation() -> None:
    """Instantiate all error types for coverage."""

    errors.DependencyMissingError("dep")
    errors.ProviderTimeoutError("timeout")
    errors.ProviderRateLimitError("rate")
    errors.ProviderInvalidResponse("invalid")
    errors.ContentFetchError("content")
