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
import urllib.request
from datetime import date, timedelta
from typing import Any

from houyi import tool

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_DELAY = 1.0

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
def get_weather(lat: float, lon: float, date: str) -> str:
    """Get real weather data from Open-Meteo API.

    Fetches actual weather forecast data for the specified location and date.
    Open-Meteo is a free API that doesn't require authentication.

    Args:
        lat: Latitude coordinate (-90 to 90).
        lon: Longitude coordinate (-180 to 180).
        date: ISO format date string or relative string like "today", "tomorrow".

    Returns:
        Weather description with max/min temperatures.
    """
    is_valid, error = _validate_coordinates(lat, lon)
    if not is_valid:
        return f"Error: {error}"

    # Normalize date
    if not date:
        date = get_date._original_func()  # type: ignore[attr-defined]
    elif isinstance(date, str) and date.strip() and not date[0].isdigit():
        date = get_date._original_func(date)  # type: ignore[attr-defined]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,weathercode"
        "&timezone=auto"
        f"&start_date={date}&end_date={date}"
    )

    logger.debug("Fetching weather from: %s", url)

    try:
        payload = _fetch_with_retry(url)
        daily = payload.get("daily", {})
        if not daily:
            return f"No weather data available for ({lat:.4f}, {lon:.4f}) on {date}"

        tmax = (daily.get("temperature_2m_max") or [None])[0]
        tmin = (daily.get("temperature_2m_min") or [None])[0]
        code = (daily.get("weathercode") or [None])[0]

        weather_desc = _weather_code_to_description(code)
        tmax_str = f"{tmax}°C" if tmax is not None else "N/A"
        tmin_str = f"{tmin}°C" if tmin is not None else "N/A"

        return f"Weather for ({lat:.4f}, {lon:.4f}) on {date}: {weather_desc}, high {tmax_str}, low {tmin_str}"

    except urllib.error.URLError:
        return f"Weather unavailable for ({lat:.4f}, {lon:.4f}) on {date}: Network error"
    except Exception as e:
        logger.error("Unexpected error fetching weather: %s", e, exc_info=True)
        return f"Weather unavailable for ({lat:.4f}, {lon:.4f}) on {date}: {type(e).__name__}"


# ── Default lifecycle hooks ──────────────────────────────────────────
# These demonstrate the hooks system and provide useful default behaviour.
# Users can extend or replace these hooks in their own SKILL.md or code.

from houyi.core.skill.hooks import HookEvent, HookType, SkillHook


def _weather_pre_tool_use(context: Any) -> dict[str, Any]:
    """PreToolUse hook: validate coordinates before making the API call."""
    args = context.tool_args or {}
    lat = args.get("lat", 0)
    lon = args.get("lon", 0)
    ok, err = _validate_coordinates(lat, lon)
    if not ok:
        return {
            "success": False,
            "output": f"[PreToolUse] Blocked: {err}",
            "should_block": True,
        }
    date_val = args.get("date", "")
    return {
        "success": True,
        "output": (f"[PreToolUse] ✓ Coordinates ({lat}, {lon}) validated, date={date_val}"),
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
