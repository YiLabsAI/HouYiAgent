---
name: weather
version: 0.2.0
description: Weather query tools using Open-Meteo API
author: Houyi Team
hooks:
  - event: PreToolUse
    matcher: get_weather
  - event: PostToolUse
    matcher: get_weather
invocationPolicy:
  sideEffect: network
  modelAutoInvoke: allow_with_consent
permissions:
  network:
    enabled: true
---

# Weather Skill

Provides weather-related query capabilities with real-time data from Open-Meteo API.

## Tools

### get_date

Get current date or calculate date with offset.

**Parameters:**
- `offset_days` (int | str, optional): Days to offset, or relative string like "today", "tomorrow", "yesterday"

**Returns:** ISO format date string (YYYY-MM-DD)

**Examples:**
```
get_date()           # Returns today's date
get_date(1)          # Returns tomorrow's date
get_date(-1)         # Returns yesterday's date
get_date("tomorrow") # Returns tomorrow's date
```

### get_weather

Get real weather data from Open-Meteo API (free, no API key required).

**Parameters:**
- `lat` (float): Latitude coordinate (-90 to 90)
- `lon` (float): Longitude coordinate (-180 to 180)
- `date` (str): ISO date string or relative string like "today", "tomorrow"

**Returns:** Weather description with max/min temperatures

**Example:**
```
get_weather(39.9042, 116.4074, "today")
# Returns: "Weather for (39.9042, 116.4074) on 2026-02-03: Clear sky, high 5°C, low -3°C"
```

## Lifecycle Hooks

This skill registers default hooks that users can extend:

- **PreToolUse**: Validates coordinates are within valid ranges before making the API call. Blocks execution if coordinates are invalid.
- **PostToolUse**: Logs a summary of the weather result. Output is injected into the prompt for downstream use.

## Data Source

Uses [Open-Meteo](https://open-meteo.com/) — a free, open-source weather API. No API key required.
