from __future__ import annotations

from typing import Any

from houyi.skills.weather import skill as weather_skill


class _Ctx:
    def __init__(self, tool_args: dict[str, Any]) -> None:
        self.tool_args = tool_args


def _call_get_weather(**kwargs: Any) -> str:
    func = getattr(weather_skill.get_weather, "_original_func", weather_skill.get_weather)
    return func(**kwargs)


def test_rejects_invalid_provider() -> None:
    out = _call_get_weather(lat=39.9, lon=116.4, provider="bad")
    assert "provider must be one of" in out


def test_requires_city_or_coords() -> None:
    out = _call_get_weather(date="today")
    assert out == "Error: provide either both lat/lon or city"


def test_coords_openmeteo_success(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_build_open_meteo_result",
        lambda lat, lon, d: f"ok:{lat:.1f},{lon:.1f}:{d}",
    )
    out = _call_get_weather(lat=39.9, lon=116.4, date="today", provider="openmeteo")
    assert out.startswith("ok:39.9,116.4:")


def test_city_auto_uses_openmeteo(monkeypatch) -> None:
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


def test_openmeteo_reports_geocode_error(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_resolve_coordinates_from_city",
        lambda city, country=None: {"found": False, "error": "City not found"},
    )
    out = _call_get_weather(city="Unknown", provider="openmeteo")
    assert out == "Weather unavailable: City not found"


def test_auto_fallbacks_to_wttr(monkeypatch) -> None:
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


def test_wttr_with_coords(monkeypatch) -> None:
    monkeypatch.setattr(weather_skill, "_build_wttr_result", lambda query: f"wttr:{query}")
    out = _call_get_weather(lat=31.2, lon=121.5, provider="wttr")
    assert out == "wttr:31.2,121.5"


def test_use_blocks_partial_coords() -> None:
    result = weather_skill._weather_pre_tool_use(_Ctx({"lat": 30.0}))
    assert result["should_block"] is True
    assert "lat and lon" in result["output"]


def test_use_blocks_empty_mode() -> None:
    result = weather_skill._weather_pre_tool_use(_Ctx({}))
    assert result["should_block"] is True
    assert "provide either both lat/lon or city" in result["output"]


def test_use_accepts_city_mode() -> None:
    result = weather_skill._weather_pre_tool_use(_Ctx({"city": "Hangzhou", "provider": "auto"}))
    assert result["success"] is True
    assert "mode=city" in result["output"]


def test_coordinates_out_of_range() -> None:
    ok, msg = weather_skill._validate_coordinates(100.0, 10.0)
    assert ok is False
    assert "out of range" in msg


def test_normalize_date_input() -> None:
    assert weather_skill._normalize_date_input("today") == weather_skill.get_date._original_func()
    assert weather_skill._normalize_date_input(
        " tomorrow "
    ) == weather_skill.get_date._original_func(1)


def test_invalid_values() -> None:
    assert weather_skill._safe_float("12.3") == 12.3
    assert weather_skill._safe_float(None) is None


def test_normalize_provider() -> None:
    provider, error = weather_skill._normalize_provider(" wttr ")
    assert provider == "wttr"
    assert error is None


def test_request_invalid_geo(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_resolve_coordinates_from_city",
        lambda city, country=None: {"found": True, "lat": "bad", "lon": None},
    )

    request, error = weather_skill._resolve_weather_request(
        lat=None,
        lon=None,
        date_input="today",
        city="Beijing",
        country=None,
        provider="auto",
    )

    assert request == {}
    assert error == "Weather unavailable: city lookup returned invalid coordinates"


def test_try_openmeteo_error() -> None:
    message = weather_skill._provider_unavailable_message(
        provider="openmeteo",
        request={"resolved_lat": None, "resolved_lon": None, "location_label": ""},
    )
    assert "Open-Meteo requires" in message


def test_try_wttr_error() -> None:
    message = weather_skill._provider_unavailable_message(
        provider="wttr",
        request={"resolved_lat": None, "resolved_lon": None, "location_label": ""},
    )
    assert "wttr.in requires" in message


def test_try_wttr_network(monkeypatch) -> None:
    def _raise(query: str) -> str:
        _ = query
        raise OSError("down")

    monkeypatch.setattr(weather_skill, "_build_wttr_result", _raise)

    out = weather_skill._try_wttr(
        request={"location_label": "Beijing", "resolved_lat": None, "resolved_lon": None}
    )

    assert out == "Weather unavailable: wttr.in OSError"


def test_weather_code_fallback() -> None:
    assert weather_skill._weather_code_to_description(None) == "Unknown"
    assert weather_skill._weather_code_to_description(999) == "Weather code 999"


def test_runcates_output() -> None:
    class _Ctx2:
        tool_result: str = "x" * 500

    result = weather_skill._weather_post_tool_use(_Ctx2())
    assert result["success"] is True
    assert "[PostToolUse]" in str(result["output"])


def test_unavailable_when_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        weather_skill,
        "_resolve_coordinates_from_city",
        lambda city, country=None: {
            "found": True,
            "city": city,
            "country": country,
            "lat": 31.2,
            "lon": 121.5,
        },
    )

    def _raise(*args: Any, **kwargs: Any) -> str:
        raise OSError("fail")

    monkeypatch.setattr(weather_skill, "_build_open_meteo_result", _raise)
    monkeypatch.setattr(weather_skill, "_build_wttr_result", _raise)
    out = _call_get_weather(city="Shanghai", provider="auto")
    assert "Weather unavailable" in out
