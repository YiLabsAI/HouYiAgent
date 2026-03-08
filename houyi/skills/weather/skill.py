"""Weather skill implementation.

Provides weather query capabilities using Open-Meteo API.
Each function decorated with @tool becomes a separate skill.

Tools:
- get_date: Get current or offset date (pure, no network)
- get_weather: Get real weather data from Open-Meteo API (network call)

Default hooks:
- PreToolUse: Validate coordinates before API call
- PostToolUse: Log result summary after API call
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any

from houyi import tool
from houyi.domain.skill.policy import InvocationPolicy, NetworkPerm, Permissions, SideEffect

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_DELAY = 1.0
WTTR_ENDPOINT = "https://wttr.in"
OPEN_METEO_FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# Valid coordinate ranges
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)


def _validate_coordinates(lat: float, lon: float) -> tuple[bool, str]:
    """Validate latitude and longitude values."""
    if not isinstance(lat, (int, float)):
        return False, f"Invalid latitude type: {type(lat).__name__}, expected number"
    if not isinstance(lon, (int, float)):
        return False, f"Invalid longitude type: {type(lon).__name__}, expected number"
    if not LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
        return False, f"Latitude {lat} out of range [{LAT_RANGE[0]}, {LAT_RANGE[1]}]"
    if not LON_RANGE[0] <= lon <= LON_RANGE[1]:
        return False, f"Longitude {lon} out of range [{LON_RANGE[0]}, {LON_RANGE[1]}]"
    return True, ""


def _fetch_with_retry(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Fetch URL with retry logic."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            last_error = e
            if attempt < max_retries:
                logger.debug("Request failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
                time.sleep(RETRY_DELAY)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Invalid JSON response: %s", e)
            break

    raise urllib.error.URLError(f"Request failed: {last_error}")


def _fetch_text_with_retry(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Fetch text URL with retry logic."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8").strip()
        except urllib.error.URLError as e:
            last_error = e
            if attempt < max_retries:
                logger.debug(
                    "Text request failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                time.sleep(RETRY_DELAY)

    raise urllib.error.URLError(f"Request failed: {last_error}")


def _normalize_date_input(date_input: str | None) -> str:
    """Normalize empty/relative date input into ISO date string."""
    if not date_input:
        return get_date._original_func()  # type: ignore[attr-defined]
    date_input = str(date_input).strip()
    if not date_input:
        return get_date._original_func()  # type: ignore[attr-defined]
    if not date_input[0].isdigit():
        return get_date._original_func(date_input)  # type: ignore[attr-defined]
    return date_input


def _resolve_coordinates_from_city(city: str, country: str | None = None) -> dict[str, Any]:
    """Resolve city/country to coordinates via location skill."""
    from houyi.skills.location.skill import get_location

    query = city.strip()
    if country and country.strip():
        query = f"{query}, {country.strip()}"

    resolver = getattr(get_location, "_original_func", None)
    if not callable(resolver):
        return {"found": False, "error": "Location resolver unavailable"}

    result = resolver(query)
    if not isinstance(result, dict):
        return {"found": False, "error": "Invalid geocoding response"}
    return result


def _safe_float(value: Any) -> float | None:
    """Convert value to float when possible, otherwise return None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_open_meteo_result(lat: float, lon: float, resolved_date: str) -> str:
    """Fetch weather from Open-Meteo and format output."""
    url = (
        OPEN_METEO_FORECAST_ENDPOINT + f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,weathercode"
        "&timezone=auto"
        f"&start_date={resolved_date}&end_date={resolved_date}"
    )

    logger.debug("Fetching weather from Open-Meteo: %s", url)

    payload = _fetch_with_retry(url)
    daily = payload.get("daily", {})
    if not daily:
        return f"No weather data available for ({lat:.4f}, {lon:.4f}) on {resolved_date}"

    tmax = (daily.get("temperature_2m_max") or [None])[0]
    tmin = (daily.get("temperature_2m_min") or [None])[0]
    code = (daily.get("weathercode") or [None])[0]

    weather_desc = _weather_code_to_description(code)
    tmax_str = f"{tmax}°C" if tmax is not None else "N/A"
    tmin_str = f"{tmin}°C" if tmin is not None else "N/A"

    return (
        f"Weather for ({lat:.4f}, {lon:.4f}) on {resolved_date}: "
        f"{weather_desc}, high {tmax_str}, low {tmin_str}"
    )


def _build_wttr_result(location_query: str) -> str:
    """Fetch compact current weather from wttr.in."""
    encoded_location = urllib.parse.quote_plus(location_query.strip())
    url = f"{WTTR_ENDPOINT}/{encoded_location}?format=%l:+%c+%t+%h+%w"
    logger.debug("Fetching weather from wttr.in: %s", url)
    text = _fetch_text_with_retry(url)
    if not text:
        raise urllib.error.URLError("Empty wttr.in response")
    return f"Current weather (wttr.in): {text}"


def _weather_code_to_description(code: int | None) -> str:
    """Convert WMO weather code to human-readable description."""
    if code is None:
        return "Unknown"

    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Freezing light drizzle",
        57: "Freezing dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Freezing light rain",
        67: "Freezing heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return weather_codes.get(code, f"Weather code {code}")


def _normalize_provider(provider: str) -> tuple[str, str | None]:
    normalized_provider = (provider or "auto").strip().lower()
    if normalized_provider not in {"auto", "openmeteo", "wttr"}:
        return "", "Error: provider must be one of auto/openmeteo/wttr"
    return normalized_provider, None


def _resolve_weather_request(
    *,
    lat: float | None,
    lon: float | None,
    date_input: str | None,
    city: str | None,
    country: str | None,
    provider: str,
) -> tuple[dict[str, Any], str | None]:
    has_coords = lat is not None and lon is not None
    resolved_lat: float | None = None
    resolved_lon: float | None = None
    if has_coords:
        if lat is None or lon is None:
            return {}, "Error: lat/lon must be provided together"
        resolved_lat = _safe_float(lat)
        resolved_lon = _safe_float(lon)
        if resolved_lat is None or resolved_lon is None:
            return {}, "Error: lat/lon must be numeric"
        is_valid, error = _validate_coordinates(resolved_lat, resolved_lon)
        if not is_valid:
            return {}, f"Error: {error}"

    city_value = (city or "").strip()
    country_value = (country or "").strip()
    location_label = ", ".join(v for v in [city_value, country_value] if v)

    if not has_coords and not city_value:
        return {}, "Error: provide either both lat/lon or city"

    resolved_date = _normalize_date_input(date_input)
    if (resolved_lat is None or resolved_lon is None) and city_value:
        location_result = _resolve_coordinates_from_city(city_value, country_value)
        if location_result.get("found"):
            maybe_lat = _safe_float(location_result.get("lat"))
            maybe_lon = _safe_float(location_result.get("lon"))
            if maybe_lat is None or maybe_lon is None:
                return {}, "Weather unavailable: city lookup returned invalid coordinates"
            resolved_lat = maybe_lat
            resolved_lon = maybe_lon
            resolved_city = str(location_result.get("city") or city_value)
            resolved_country = str(location_result.get("country") or country_value).strip()
            location_label = ", ".join(v for v in [resolved_city, resolved_country] if v)
        elif provider == "openmeteo":
            return {}, f"Weather unavailable: {location_result.get('error', 'city lookup failed')}"

    return {
        "resolved_lat": resolved_lat,
        "resolved_lon": resolved_lon,
        "resolved_date": resolved_date,
        "location_label": location_label,
    }, None


def _try_openmeteo(*, provider: str, request: dict[str, Any]) -> str | None:
    resolved_lat = request["resolved_lat"]
    resolved_lon = request["resolved_lon"]
    resolved_date = request["resolved_date"]
    if resolved_lat is None or resolved_lon is None:
        return None

    try:
        return _build_open_meteo_result(resolved_lat, resolved_lon, resolved_date)
    except urllib.error.URLError:
        if provider == "openmeteo":
            return (
                f"Weather unavailable for ({resolved_lat:.4f}, {resolved_lon:.4f}) "
                f"on {resolved_date}: Network error"
            )
    except Exception as e:
        logger.error("Unexpected Open-Meteo error: %s", e, exc_info=True)
        if provider == "openmeteo":
            return (
                f"Weather unavailable for ({resolved_lat:.4f}, {resolved_lon:.4f}) "
                f"on {resolved_date}: {type(e).__name__}"
            )
    return None


def _try_wttr(*, request: dict[str, Any]) -> str | None:
    location_label = request["location_label"]
    resolved_lat = request["resolved_lat"]
    resolved_lon = request["resolved_lon"]
    if not location_label and (resolved_lat is None or resolved_lon is None):
        return None

    try:
        wttr_query = location_label or f"{resolved_lat},{resolved_lon}"
        return _build_wttr_result(wttr_query)
    except urllib.error.URLError:
        return "Weather unavailable: wttr.in network error"
    except Exception as e:
        logger.error("Unexpected wttr.in error: %s", e, exc_info=True)
        return f"Weather unavailable: wttr.in {type(e).__name__}"


def _provider_unavailable_message(*, provider: str, request: dict[str, Any]) -> str:
    resolved_lat = request["resolved_lat"]
    resolved_lon = request["resolved_lon"]
    location_label = request["location_label"]
    if provider == "openmeteo":
        return "Weather unavailable: Open-Meteo requires lat/lon or resolvable city"
    if provider == "wttr":
        if location_label or (resolved_lat is not None and resolved_lon is not None):
            return "Weather unavailable: wttr.in network error"
        return "Weather unavailable: wttr.in requires city/country or coordinates"
    return "Weather unavailable: no compatible weather provider path was available"


@tool
def get_date(offset_days: int | str = 0) -> str:
    """Get ISO date with optional day offset or relative string.

    Args:
        offset_days: Number of days to offset from today, or a relative string
                    like "today", "tomorrow", "yesterday", or an ISO date string.

    Returns:
        ISO format date string (YYYY-MM-DD).

    Examples:
        get_date()           -> "2026-02-05" (today)
        get_date(1)          -> "2026-02-06" (tomorrow)
        get_date("tomorrow") -> "2026-02-06"
    """
    if offset_days is None:
        offset_days = 0

    if isinstance(offset_days, str):
        normalized = offset_days.strip().lower()
        relative_map = {
            "today": 0,
            "now": 0,
            "tomorrow": 1,
            "tmr": 1,
            "tmr.": 1,
            "yesterday": -1,
            "yday": -1,
        }
        if normalized in relative_map:
            offset_days = relative_map[normalized]
        else:
            try:
                return date.fromisoformat(normalized).isoformat()
            except ValueError:
                logger.warning("Invalid date string '%s', defaulting to today", offset_days)
                offset_days = 0

    try:
        offset_days = int(offset_days)
    except (ValueError, TypeError):
        offset_days = 0

    # Clamp to reasonable range
    offset_days = max(-365, min(365, offset_days))
    return (date.today() + timedelta(days=offset_days)).isoformat()


@tool
def get_weather(
    lat: float | None = None,
    lon: float | None = None,
    date: str | None = None,
    city: str | None = None,
    country: str | None = None,
    provider: str = "auto",
) -> str:
    """Get weather using coordinates or city with provider fallback.

    Supports two input modes:
    1) Coordinates mode: ``lat`` + ``lon`` (+ optional ``date``)
    2) City mode: ``city`` (+ optional ``country`` + optional ``date``)

    Provider choices:
    - ``openmeteo``: structured daily forecast (date-aware)
    - ``wttr``: current weather text summary
    - ``auto`` (default): try Open-Meteo first, fallback to wttr when possible

    Args:
        lat: Latitude coordinate (-90 to 90).
        lon: Longitude coordinate (-180 to 180).
        date: ISO date string or relative string like "today"/"tomorrow".
        city: City name for friendly input mode.
        country: Optional country/region to disambiguate city.
        provider: ``auto`` | ``openmeteo`` | ``wttr``.

    Returns:
        Human-readable weather summary.
    """
    normalized_provider, provider_error = _normalize_provider(provider)
    if provider_error:
        return provider_error

    request, request_error = _resolve_weather_request(
        lat=lat,
        lon=lon,
        date_input=date,
        city=city,
        country=country,
        provider=normalized_provider,
    )
    if request_error:
        return request_error

    if normalized_provider in {"openmeteo", "auto"}:
        result = _try_openmeteo(provider=normalized_provider, request=request)
        if result is not None:
            return result

    if normalized_provider in {"wttr", "auto"}:
        result = _try_wttr(request=request)
        if result is not None:
            return result

    return _provider_unavailable_message(provider=normalized_provider, request=request)


get_weather.invocation_policy = InvocationPolicy.default_for_side_effect(SideEffect.NETWORK)
get_weather.permissions = Permissions(network=NetworkPerm(enabled=True))


# ── Default lifecycle hooks ──────────────────────────────────────────

from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook


def _weather_pre_tool_use(context: Any) -> dict[str, Any]:
    """PreToolUse hook: validate weather input mode before API call."""
    args = context.tool_args or {}
    lat = args.get("lat")
    lon = args.get("lon")
    city = (args.get("city") or "").strip() if isinstance(args.get("city"), str) else ""

    if (lat is None) != (lon is None):
        return {
            "success": False,
            "output": "[PreToolUse] Blocked: lat and lon must be provided together",
            "should_block": True,
        }

    if lat is not None and lon is not None:
        ok, err = _validate_coordinates(lat, lon)
        if not ok:
            return {
                "success": False,
                "output": f"[PreToolUse] Blocked: {err}",
                "should_block": True,
            }

    if lat is None and lon is None and not city:
        return {
            "success": False,
            "output": "[PreToolUse] Blocked: provide either both lat/lon or city",
            "should_block": True,
        }

    date_val = args.get("date", "")
    provider_val = args.get("provider", "auto")
    mode_desc = "coords" if lat is not None and lon is not None else "city"
    return {
        "success": True,
        "output": (
            f"[PreToolUse] ✓ Weather input validated (mode={mode_desc}, provider={provider_val}, date={date_val})"
        ),
    }


def _weather_post_tool_use(context: Any) -> dict[str, Any]:
    """PostToolUse hook: log a summary of the weather result."""
    result_str = str(context.tool_result or "")[:200]
    return {
        "success": True,
        "output": f"[PostToolUse] Weather result: {result_str}",
        "inject_to_prompt": True,
    }


# Attach hooks to the get_weather SkillSpec
get_weather.hooks = [
    SkillHook(
        event=HookEvent.PRE_TOOL_USE,
        hook_type=HookType.HANDLER,
        handler=_weather_pre_tool_use,
        matcher="get_weather",
    ),
    SkillHook(
        event=HookEvent.POST_TOOL_USE,
        hook_type=HookType.HANDLER,
        handler=_weather_post_tool_use,
        matcher="get_weather",
    ),
]


# Export tools (each is a SkillSpec)
__all__ = ["get_date", "get_weather"]
