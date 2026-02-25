from __future__ import annotations

from typing import Any

from houyi.skills.weather import skill as weather_skill


class _Ctx:
    def __init__(self, tool_args: dict[str, Any]) -> None:
        self.tool_args = tool_args


def _call_get_weather(**kwargs: Any) -> str:
    func = getattr(weather_skill.get_weather, "_original_func", weather_skill.get_weather)
    return func(**kwargs)


def test_get_weather_rejects_invalid_provider() -> None:
    out = _call_get_weather(lat=39.9, lon=116.4, provider="bad")
    assert "provider must be one of" in out


def test_get_weather_requires_city_or_coords() -> None:
    out = _call_get_weather(date="today")
    assert out == "Error: provide either both lat/lon or city"


def test_get_weather_coords_openmeteo_success(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_build_open_meteo_result",
        lambda lat, lon, d: f"ok:{lat:.1f},{lon:.1f}:{d}",
    )
    out = _call_get_weather(lat=39.9, lon=116.4, date="today", provider="openmeteo")
    assert out.startswith("ok:39.9,116.4:")


def test_get_weather_city_auto_uses_geocoding_then_openmeteo(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_resolve_coordinates_from_city",
        lambda city, country=None: {
            "found": True,
            "city": city,
            "country": country,
            "lat": 40.0,
            "lon": 116.0,
        },
    )
    monkeypatch.setattr(
        weather_skill,
        "_build_open_meteo_result",
        lambda lat, lon, d: f"openmeteo:{lat},{lon}:{d}",
    )
    out = _call_get_weather(city="Beijing", country="CN", date="today", provider="auto")
    assert out.startswith("openmeteo:40.0,116.0:")


def test_get_weather_city_openmeteo_reports_geocode_error(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_resolve_coordinates_from_city",
        lambda city, country=None: {"found": False, "error": "City not found"},
    )
    out = _call_get_weather(city="Unknown", provider="openmeteo")
    assert out == "Weather unavailable: City not found"


def test_get_weather_auto_fallbacks_to_wttr_when_openmeteo_fails(monkeypatch) -> None:
    def _raise(*args: Any, **kwargs: Any) -> str:
        raise OSError("network")

    monkeypatch.setattr(
        weather_skill,
        "_resolve_coordinates_from_city",
        lambda city, country=None: {
            "found": True,
            "city": city,
            "country": None,
            "lat": 31.2,
            "lon": 121.5,
        },
    )
    monkeypatch.setattr(weather_skill, "_build_open_meteo_result", _raise)
    monkeypatch.setattr(weather_skill, "_build_wttr_result", lambda query: f"wttr:{query}")

    out = _call_get_weather(city="Shanghai", provider="auto")
    assert out == "wttr:Shanghai"


def test_get_weather_wttr_with_coords(monkeypatch) -> None:
    monkeypatch.setattr(weather_skill, "_build_wttr_result", lambda query: f"wttr:{query}")
    out = _call_get_weather(lat=31.2, lon=121.5, provider="wttr")
    assert out == "wttr:31.2,121.5"


def test_weather_pre_tool_use_blocks_partial_coords() -> None:
    result = weather_skill._weather_pre_tool_use(_Ctx({"lat": 30.0}))
    assert result["should_block"] is True
    assert "lat and lon" in result["output"]


def test_weather_pre_tool_use_blocks_empty_mode() -> None:
    result = weather_skill._weather_pre_tool_use(_Ctx({}))
    assert result["should_block"] is True
    assert "provide either both lat/lon or city" in result["output"]


def test_weather_pre_tool_use_accepts_city_mode() -> None:
    result = weather_skill._weather_pre_tool_use(_Ctx({"city": "Hangzhou", "provider": "auto"}))
    assert result["success"] is True
    assert "mode=city" in result["output"]
