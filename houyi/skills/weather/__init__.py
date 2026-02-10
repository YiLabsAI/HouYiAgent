"""Weather skill package.

Provides weather-related tools:
- get_date: Get current or offset date
- get_weather: Get mock weather data
- get_weather_live: Get real weather from Open-Meteo API

Each tool is a SkillSpec that can be registered individually.
"""

from __future__ import annotations

from .skill import get_date, get_weather, get_weather_live

__all__ = ["get_date", "get_weather", "get_weather_live"]
