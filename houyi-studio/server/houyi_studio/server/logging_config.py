"""Logging configuration utilities for console server."""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any

LOG_LEVEL_ENV_VAR = "HOUYI_LOG_LEVEL"
LOG_PAYLOAD_LIMIT_ENV_VAR = "HOUYI_LOG_PAYLOAD_LIMIT"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_PAYLOAD_LIMIT = 512

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class UvicornLoggerNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.error":
            record.name = "uvicorn"
        return True


def _resolve_log_level(level: str | None) -> str:
    if not level:
        return DEFAULT_LOG_LEVEL
    normalized = level.strip().upper()
    if normalized not in logging._nameToLevel:  # type: ignore[attr-defined]
        return DEFAULT_LOG_LEVEL
    return normalized


def _access_log_level(level: str) -> str:
    return "INFO" if level == "DEBUG" else "WARNING"


def _uvicorn_error_level(level: str) -> str:
    return "INFO" if level == "DEBUG" else level


def _watchfiles_level(level: str) -> str:
    return "INFO" if level in {"DEBUG", "INFO"} else "WARNING"


def build_logging_config(level: str | None = None) -> dict[str, Any]:
    resolved_level = _resolve_log_level(level)
    access_level = _access_log_level(resolved_level)
    error_level = _uvicorn_error_level(resolved_level)
    watchfiles_level = _watchfiles_level(resolved_level)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "uvicorn_logger_name": {
                "()": "houyi_studio.server.logging_config.UvicornLoggerNameFilter",
            }
        },
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": LOG_DATE_FORMAT,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["uvicorn_logger_name"],
            }
        },
        "root": {
            "level": resolved_level,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": resolved_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": error_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": access_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "watchfiles": {
                "level": watchfiles_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "watchfiles.main": {
                "level": watchfiles_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "ddgs": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "ddgs.ddgs": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "primp": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "primp.impersonate": {
                "level": "ERROR",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }


def configure_logging(level: str | None = None) -> str:
    resolved_level = _resolve_log_level(level or os.getenv(LOG_LEVEL_ENV_VAR))
    logging.config.dictConfig(build_logging_config(resolved_level))
    return resolved_level


def get_log_level() -> str:
    return logging.getLevelName(logging.getLogger().getEffectiveLevel())


def set_log_level(level: str) -> str:
    resolved_level = _resolve_log_level(level)
    access_level = _access_log_level(resolved_level)
    error_level = _uvicorn_error_level(resolved_level)
    watchfiles_level = _watchfiles_level(resolved_level)

    logging.getLogger().setLevel(resolved_level)
    logging.getLogger("uvicorn").setLevel(resolved_level)
    logging.getLogger("uvicorn.error").setLevel(error_level)
    logging.getLogger("uvicorn.asgi").setLevel(resolved_level)
    logging.getLogger("uvicorn.access").setLevel(access_level)
    logging.getLogger("watchfiles").setLevel(watchfiles_level)
    logging.getLogger("watchfiles.main").setLevel(watchfiles_level)
    logging.getLogger("ddgs").setLevel(logging.WARNING)
    logging.getLogger("ddgs.ddgs").setLevel(logging.WARNING)
    logging.getLogger("primp").setLevel(logging.WARNING)
    logging.getLogger("primp.impersonate").setLevel(logging.ERROR)

    return resolved_level


def truncate_payload(value: Any, limit: int | None = None) -> str:
    payload_limit = limit
    if payload_limit is None:
        raw_limit = os.getenv(LOG_PAYLOAD_LIMIT_ENV_VAR)
        try:
            payload_limit = int(raw_limit) if raw_limit else DEFAULT_PAYLOAD_LIMIT
        except ValueError:
            payload_limit = DEFAULT_PAYLOAD_LIMIT

    text = value if isinstance(value, str) else str(value)
    if payload_limit is not None and payload_limit > 0 and len(text) > payload_limit:
        remaining = len(text) - payload_limit
        return f"{text[:payload_limit]}...(+{remaining} chars)"
    return text
