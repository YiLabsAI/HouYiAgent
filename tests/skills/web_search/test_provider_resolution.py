from __future__ import annotations

from houyi.skills.web_search.provider_resolution import (
    normalize_web_search_provider_name,
    resolve_supported_web_search_provider,
)


def test_normalize_provider() -> None:
    assert normalize_web_search_provider_name(" Google_Scholar ") == "google_scholar"


def test_supported_requested_provider() -> None:
    assert resolve_supported_web_search_provider("tavily", configured_provider="serper") == "tavily"


def test_unknown_request() -> None:
    assert (
        resolve_supported_web_search_provider("google_scholar", configured_provider="serper")
        == "serper"
    )


def test_fallback_provider() -> None:
    assert (
        resolve_supported_web_search_provider(
            "google_scholar",
            configured_provider="unknown",
            fallback_provider="ddg",
        )
        == "ddg"
    )


def test_defaults_to_serper() -> None:
    assert resolve_supported_web_search_provider("google_scholar") == "serper"


def test_defaults_to_ddg() -> None:
    assert (
        resolve_supported_web_search_provider(
            "google_scholar",
            allowed_providers={"ddg", "tavily"},
        )
        == "ddg"
    )
