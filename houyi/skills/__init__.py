"""Built-in skills package.

This package contains built-in skill implementations:
- planning: Task planning and tracking with markdown files
- weather: Weather query tools using Open-Meteo API
- location: Location/geocoding tools using Open-Meteo Geocoding API

Skills can be loaded individually or registered automatically.
"""

from __future__ import annotations

__all__ = ["location", "planning", "weather"]
