"""Location skill implementation.

Provides geocoding capabilities using Open-Meteo Geocoding API.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from houyi import tool
from houyi.domain.skill.policy import InvocationPolicy, NetworkPerm, Permissions, SideEffect

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_DELAY = 1.0
MAX_CITY_LENGTH = 200
DEFAULT_CITY = "Hangzhou"
GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
GEOCODING_DEFAULT_PARAMS: dict[str, str] = {
    "count": "1",
    "language": "en",
    "format": "json",
}


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


def _sanitize_city_name(city: str | None) -> str:
    """Sanitize and validate city name input."""
    if not city:
        return DEFAULT_CITY
    if not isinstance(city, str):
        logger.warning("Invalid city type: %s, using default", type(city).__name__)
        return DEFAULT_CITY

    city = city.strip()
    if not city:
        return DEFAULT_CITY
    if len(city) > MAX_CITY_LENGTH:
        logger.warning("City name too long (%d chars), truncating", len(city))
        city = city[:MAX_CITY_LENGTH]

    return city


def _split_city_country_code(city: str) -> tuple[str, str | None]:
    """Split inputs like ``Beijing, CN`` into city + ISO country code."""
    parts = [p.strip() for p in city.split(",") if p.strip()]
    if len(parts) < 2:
        return city, None

    maybe_code = parts[-1]
    if len(maybe_code) == 2 and maybe_code.isalpha():
        city_part = ", ".join(parts[:-1]).strip()
        if city_part:
            return city_part, maybe_code.upper()
    return city, None


def _build_geocoding_url(city_name: str, country_code: str | None = None) -> str:
    """Build Open-Meteo geocoding URL with optional country code filter."""
    params: dict[str, str] = {
        "name": city_name,
        **GEOCODING_DEFAULT_PARAMS,
    }
    if country_code:
        params["countryCode"] = country_code
    return f"{GEOCODING_ENDPOINT}?{urllib.parse.urlencode(params)}"


@tool
def get_location(city: str | None = None) -> dict[str, Any]:
    """Get coordinates for a city using Open-Meteo Geocoding API.

    Args:
        city: City name to geocode. Defaults to "Hangzhou" if not provided.

    Returns:
        Dictionary with city name, latitude, longitude, and additional info.
        On error, returns dict with error field.

    Examples:
        get_location("Beijing")  -> {"city": "Beijing", "lat": 39.9042, ...}
        get_location()           -> {"city": "Hangzhou", "lat": 30.2741, ...}
    """
    city = _sanitize_city_name(city)
    query_city, country_code = _split_city_country_code(city)

    try:
        # Validate that the effective query string is encodable.
        urllib.parse.quote(query_city, safe="")
    except Exception as e:
        logger.warning("Failed to encode city name '%s': %s", city, e)
        return {
            "city": city,
            "lat": None,
            "lon": None,
            "found": False,
            "error": "Invalid city name",
        }

    url = _build_geocoding_url(query_city, country_code=country_code)
    logger.debug("Geocoding city: %s (query=%s, country_code=%s)", city, query_city, country_code)

    try:
        payload = _fetch_with_retry(url)
        results = payload.get("results")

        # Some queries like "beijing, CN" may over-constrain results with countryCode.
        # Retry once without country filter before returning not-found.
        if (not isinstance(results, list) or len(results) == 0) and country_code:
            fallback_url = _build_geocoding_url(query_city, country_code=None)
            payload = _fetch_with_retry(fallback_url)
            results = payload.get("results")

        if not results or not isinstance(results, list) or len(results) == 0:
            logger.info("No geocoding results for: %s", city)
            return {
                "city": city,
                "lat": None,
                "lon": None,
                "found": False,
                "error": f"City not found: {city}",
            }

        result = results[0]
        lat = result.get("latitude")
        lon = result.get("longitude")

        if lat is None or lon is None:
            return {
                "city": city,
                "lat": None,
                "lon": None,
                "found": False,
                "error": "Coordinates not available",
            }

        return {
            "city": result.get("name", city),
            "lat": float(lat),
            "lon": float(lon),
            "country": result.get("country"),
            "country_code": result.get("country_code"),
            "timezone": result.get("timezone"),
            "population": result.get("population"),
            "admin1": result.get("admin1"),
            "found": True,
        }

    except urllib.error.URLError:
        return {"city": city, "lat": None, "lon": None, "found": False, "error": "Network error"}
    except (ValueError, TypeError) as e:
        return {
            "city": city,
            "lat": None,
            "lon": None,
            "found": False,
            "error": f"Invalid response: {e}",
        }
    except Exception as e:
        logger.error("Unexpected geocoding error for %s: %s", city, e, exc_info=True)
        return {
            "city": city,
            "lat": None,
            "lon": None,
            "found": False,
            "error": f"Unexpected: {type(e).__name__}",
        }


get_location.invocation_policy = InvocationPolicy.default_for_side_effect(SideEffect.NETWORK)
get_location.permissions = Permissions(network=NetworkPerm(enabled=True))


# ── Default lifecycle hooks ──────────────────────────────────────────

from houyi.domain.skill.hooks import HookEvent, HookType, SkillHook


def _location_pre_tool_use(context: Any) -> dict[str, Any]:
    """PreToolUse hook: validate and sanitize city name."""
    args = context.tool_args or {}
    city = args.get("city", DEFAULT_CITY)
    sanitized = _sanitize_city_name(city)
    return {
        "success": True,
        "output": f"[PreToolUse] ✓ Geocoding city: {sanitized}",
    }


def _location_post_tool_use(context: Any) -> dict[str, Any]:
    """PostToolUse hook: log geocoding result summary."""
    result = context.tool_result
    if isinstance(result, dict) and result.get("found"):
        return {
            "success": True,
            "output": (
                f"[PostToolUse] Found: {result.get('city')} "
                f"({result.get('lat')}, {result.get('lon')}) "
                f"in {result.get('country', '?')}"
            ),
            "inject_to_prompt": True,
        }
    return {
        "success": True,
        "output": f"[PostToolUse] Geocoding result: {str(result)[:200]}",
    }


get_location.hooks = [
    SkillHook(
        event=HookEvent.PRE_TOOL_USE,
        hook_type=HookType.HANDLER,
        handler=_location_pre_tool_use,
        matcher="get_location",
    ),
    SkillHook(
        event=HookEvent.POST_TOOL_USE,
        hook_type=HookType.HANDLER,
        handler=_location_post_tool_use,
        matcher="get_location",
    ),
]


# Export (get_location is a SkillSpec)
__all__ = ["get_location"]
