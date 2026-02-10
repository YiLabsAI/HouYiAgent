---
name: weather
version: 0.1.0
description: Weather query tools using Open-Meteo API
author: Houyi Team
---

# Weather Skill

Provides weather-related query capabilities.

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

Get mock weather data (for testing).

**Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude
- `date` (str): ISO date string

**Returns:** Mock weather description

### get_weather_live

Get real weather data from Open-Meteo API.

**Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude
- `date` (str): ISO date string or relative string

**Returns:** Weather description with temperatures

**Example:**
```
get_weather_live(39.9042, 116.4074, "today")
# Returns: "Weather for (39.9042, 116.4074) on 2026-02-05: Clear sky, high 5°C, low -3°C"
```

## Data Source

Uses [Open-Meteo](https://open-meteo.com/) - a free, open-source weather API.
No API key required.
