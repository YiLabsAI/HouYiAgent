---
name: weather
version: 0.3.0
description: Weather query tools using Open-Meteo with wttr.in fallback
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

Provides weather-related query capabilities with two input modes and provider fallback.

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

Get weather by coordinates or city name.

**Parameters:**
- `lat` (float, optional): Latitude coordinate (-90 to 90)
- `lon` (float, optional): Longitude coordinate (-180 to 180)
- `city` (str, optional): City name (friendly mode)
- `country` (str, optional): Country/region disambiguation for city mode
- `date` (str, optional): ISO date string or relative string like "today", "tomorrow"
- `provider` (str, optional): `auto` (default), `openmeteo`, or `wttr`

At least one mode is required:
- Coordinates mode: provide both `lat` and `lon`
- City mode: provide `city`

**Returns:** Human-readable weather summary

**Example:**
```
get_weather(lat=39.9042, lon=116.4074, date="today")
# Open-Meteo date-aware forecast

get_weather(city="Hangzhou", date="today")
# City mode (internally resolves coordinates first)

get_weather(city="London", provider="wttr")
# Current weather summary from wttr.in
```

## Lifecycle Hooks

This skill registers default hooks that users can extend:

- **PreToolUse**: Validates dual-mode input contract (`lat`+`lon` or `city`) and coordinate ranges.
- **PostToolUse**: Logs a summary of the weather result. Output is injected into the prompt for downstream use.

## Data Source

- [Open-Meteo](https://open-meteo.com/) — structured forecast (date-aware)
- [wttr.in](https://wttr.in/:help) — compact current-weather text summary
