"""Location skill implementation.

Provides geocoding capabilities using Open-Meteo Geocoding API.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from houyi import tool

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_DELAY = 1.0
MAX_CITY_LENGTH = 200
DEFAULT_CITY = "Hangzhou"


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

    try:
        encoded_city = urllib.request.quote(city, safe="")
    except Exception as e:
        logger.warning("Failed to encode city name '%s': %s", city, e)
        return {
            "city": city,
            "lat": None,
            "lon": None,
            "found": False,
            "error": "Invalid city name",
        }

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    logger.debug("Geocoding city: %s", city)

    try:
        payload = _fetch_with_retry(url)
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


# Export (get_location is a SkillSpec)
__all__ = ["get_location"]
