from __future__ import annotations

import urllib.error
from typing import Any

from houyi.skills.location import skill as location_skill


def _call_get_location(**kwargs: Any) -> dict[str, Any]:
    func = getattr(location_skill.get_location, "_original_func", location_skill.get_location)
    return func(**kwargs)


def test_split_city_country_code_iso_suffix() -> None:
    city, code = location_skill._split_city_country_code("beijing, CN")
    assert city == "beijing"
    assert code == "CN"


def test_split_city_country_code_non_iso_suffix_kept() -> None:
    city, code = location_skill._split_city_country_code("beijing, China")
    assert city == "beijing, China"
    assert code is None


def test_build_geocoding_url_with_country_code() -> None:
    url = location_skill._build_geocoding_url("beijing", country_code="CN")
    assert "name=beijing" in url
    assert "countryCode=CN" in url


def test_get_location_uses_country_code_query(monkeypatch) -> None:
    seen_urls: list[str] = []

    def _fake_fetch(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen_urls.append(url)
        return {
            "results": [
                {
                    "name": "Beijing",
                    "latitude": 39.9,
                    "longitude": 116.4,
                    "country": "China",
                    "country_code": "CN",
                }
            ]
        }

    monkeypatch.setattr(location_skill, "_fetch_with_retry", _fake_fetch)

    out = _call_get_location(city="beijing, CN")
    assert out["found"] is True
    assert len(seen_urls) == 1
    assert "countryCode=CN" in seen_urls[0]


def test_get_location_fallbacks_without_country_code(monkeypatch) -> None:
    seen_urls: list[str] = []

    def _fake_fetch(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen_urls.append(url)
        if "countryCode=CN" in url:
            return {"results": []}
        return {
            "results": [
                {
                    "name": "Beijing",
                    "latitude": 39.9,
                    "longitude": 116.4,
                    "country": "China",
                    "country_code": "CN",
                }
            ]
        }

    monkeypatch.setattr(location_skill, "_fetch_with_retry", _fake_fetch)

    out = _call_get_location(city="beijing, CN")
    assert out["found"] is True
    assert len(seen_urls) == 2
    assert "countryCode=CN" in seen_urls[0]
    assert "countryCode=" not in seen_urls[1]


def test_get_location_not_found_after_fallback(monkeypatch) -> None:
    def _fake_fetch(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"results": []}

    monkeypatch.setattr(location_skill, "_fetch_with_retry", _fake_fetch)

    out = _call_get_location(city="beijing, CN")
    assert out["found"] is False
    assert out["error"] == "City not found: beijing, CN"


def test_sanitize_city_name_handles_invalid_inputs() -> None:
    assert location_skill._sanitize_city_name(None) == "Hangzhou"
    assert location_skill._sanitize_city_name("   ") == "Hangzhou"
    assert location_skill._sanitize_city_name(123) == "Hangzhou"  # type: ignore[arg-type]


def test_get_location_handles_network_error(monkeypatch) -> None:
    def _raise(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(location_skill, "_fetch_with_retry", _raise)
    out = _call_get_location(city="Beijing")
    assert out["found"] is False
    assert out["error"] == "Network error"


def test_get_location_handles_coordinates_not_available(monkeypatch) -> None:
    monkeypatch.setattr(
        location_skill, "_fetch_with_retry", lambda *a, **k: {"results": [{"name": "X"}]}
    )
    out = _call_get_location(city="Beijing")
    assert out["found"] is False
    assert out["error"] == "Coordinates not available"


def test_location_post_tool_use_formats_success_summary() -> None:
    class _Ctx:
        tool_result: dict[str, Any] = {
            "found": True,
            "city": "Beijing",
            "lat": 39.9,
            "lon": 116.4,
            "country": "China",
        }

    result = location_skill._location_post_tool_use(_Ctx())
    assert result["success"] is True
    assert "Found: Beijing" in str(result["output"])
    assert result.get("inject_to_prompt") is True
