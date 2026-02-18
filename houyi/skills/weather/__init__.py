"""Weather skill package.

Provides weather-related tools:
- get_date: Get current or offset date (pure, no network)
- get_weather: Get real weather from Open-Meteo API (with default hooks)

Each tool is a SkillSpec that can be registered individually.
"""

from __future__ import annotations

from .skill import get_date, get_weather

__all__ = ["get_date", "get_weather"]
