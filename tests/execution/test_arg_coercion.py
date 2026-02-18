"""Tests for houyi.execution.arg_coercion — configurable argument coercion."""

from __future__ import annotations

from houyi.execution.arg_coercion import (
    _REGISTRY,
    coerce_args,
    register_arg_coercion,
)


class TestCoerceArgs:
    def test_no_coercion_returns_original(self):
        args = {"q": "hello"}
        assert coerce_args("unknown_tool", args, {}) is args

    def test_weather_coercion_injects_date(self):
        args = {"lat": 39.9, "lon": 116.4}
        outputs = {"get_date": "2026-01-01"}
        result = coerce_args("get_weather", args, outputs)
        assert result["date"] == "2026-01-01"
        assert result["lat"] == 39.9

    def test_weather_coercion_injects_location(self):
        args = {"date": "2026-01-01"}
        outputs = {"get_location": {"lat": 35.6, "lon": 139.7}}
        result = coerce_args("get_weather", args, outputs)
        assert result["lat"] == 35.6
        assert result["lon"] == 139.7

    def test_custom_registration(self):
        def _custom(args, outputs):
            return {**args, "injected": True}

        register_arg_coercion("custom_tool", _custom)
        try:
            result = coerce_args("custom_tool", {"x": 1}, {})
            assert result == {"x": 1, "injected": True}
        finally:
            _REGISTRY.pop("custom_tool", None)

    def test_coercion_error_returns_original(self):
        def _bad(args, outputs):
            raise ValueError("boom")

        register_arg_coercion("bad_tool", _bad)
        try:
            args = {"x": 1}
            result = coerce_args("bad_tool", args, {})
            assert result is args
        finally:
            _REGISTRY.pop("bad_tool", None)
