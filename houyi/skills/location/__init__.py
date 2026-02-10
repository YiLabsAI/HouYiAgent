"""Location skill package.

Provides location/geocoding tools:
- get_location: Get coordinates for a city

The tool is a SkillSpec that can be registered individually.
"""

from __future__ import annotations

from .skill import get_location

__all__ = ["get_location"]
