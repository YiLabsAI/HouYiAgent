"""Console integration fixture for tool injection.

This module registers console E2E tools for integration tests and local UI runs.
Use by importing this module in tests or by letting console server startup load it
when HOUYI_DISABLE_E2E_TOOLS is not set.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import tempfile
import urllib.request
from datetime import date, timedelta

from houyi_studio.server.app import app

from houyi import tool
from houyi.core.skill_registry import DEFAULT_SKILL_REGISTRY

PID_FILE = os.path.join(tempfile.gettempdir(), "houyi-console-e2e.pid")

_QUIET_VALUES = {"1", "true", "yes", "on"}


def _is_quiet() -> bool:
    return (os.getenv("HOUYI_E2E_QUIET") or "").strip().lower() in _QUIET_VALUES


if _is_quiet():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def _write_pid_file() -> None:
    with open(PID_FILE, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def _cleanup_pid_file() -> None:
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except OSError as exc:
        print(f"[console_e2e_server] Failed to remove PID file {PID_FILE}: {exc}")


@tool
def weather() -> str:
    """Return a mock weather response."""
    return "Sunny"


@tool
def get_location(city: str | None = None) -> dict[str, float | str]:
    """Return mock coordinates for a city."""
    if not city:
        city = "Hangzhou"
    return {"result": {"city": city, "lat": 39.9042, "lon": 116.4074}}


@tool
def get_date(offset_days: int | str = 0) -> str:
    """Return ISO date with optional day offset or relative string."""
    if isinstance(offset_days, str):
        normalized = offset_days.strip().lower()
        if normalized in {"today", "now"}:
            offset_days = 0
        elif normalized in {"tomorrow", "tmr", "tmr."}:
            offset_days = 1
        elif normalized in {"yesterday", "yday"}:
            offset_days = -1
        else:
            try:
                return date.fromisoformat(normalized).isoformat()
            except ValueError:
                offset_days = 0
    return (date.today() + timedelta(days=int(offset_days))).isoformat()


@tool
def get_weather(lat: float, lon: float, date: str) -> str:
    """Return mock weather for coordinates and date (call after location/date tools)."""
    return f"Mock weather for {lat},{lon} on {date}: Sunny"


@tool
def get_weather_live(lat: float, lon: float, date: str) -> str:
    """Fetch real weather data from Open-Meteo (call after location/date tools)."""
    disable_flag = (os.getenv("HOUYI_DISABLE_LIVE_WEATHER") or "").strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        print(
            f"[console_e2e_server] Live weather disabled via HOUYI_DISABLE_LIVE_WEATHER={disable_flag}"
        )
        return f"Mock live weather for {lat},{lon} on {date}: max=25, min=15"
    if isinstance(date, str):
        date_func = getattr(get_date, "_original_func", get_date)
        normalized_date = date_func(date) if callable(date_func) else date
    else:
        normalized_date = date
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&timezone=UTC"
        f"&start_date={normalized_date}&end_date={normalized_date}"
    )
    print(f"[console_e2e_server] Fetching live weather: {url}")
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload.get("daily", {})
    tmax = (daily.get("temperature_2m_max") or [None])[0]
    tmin = (daily.get("temperature_2m_min") or [None])[0]
    return f"Real weather for {lat},{lon} on {date}: max={tmax}, min={tmin}"


@tool
def boom() -> str:
    """Always fail for error flow coverage."""
    raise ValueError("boom")


DEFAULT_SKILL_REGISTRY.register(weather, overwrite=True)
DEFAULT_SKILL_REGISTRY.register(get_location, overwrite=True)
DEFAULT_SKILL_REGISTRY.register(get_date, overwrite=True)
DEFAULT_SKILL_REGISTRY.register(get_weather, overwrite=True)
DEFAULT_SKILL_REGISTRY.register(get_weather_live, overwrite=True)
DEFAULT_SKILL_REGISTRY.register(boom, overwrite=True)


if __name__ == "__main__":
    import uvicorn

    def _handle_exit(signum, _frame) -> None:
        print(f"[console_e2e_server] Received signal {signum}, shutting down.")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_exit)
    signal.signal(signal.SIGINT, _handle_exit)

    _write_pid_file()
    atexit.register(_cleanup_pid_file)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
