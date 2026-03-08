import logging

from houyi.application.workflow.config_service import ConfigService


class TestConfigService:
    def test_normalize_run_settings_none(self):
        service = ConfigService()
        assert service.normalize_run_settings(None) == {}
        assert service.normalize_run_settings({}) == {}

    def test_resolve_tool_settings_defaults(self):
        service = ConfigService()
        resolved = service.resolve_tool_settings(None)
        assert resolved["enable_tool_calls"] is True
        assert resolved["tool_names"] == []
        assert resolved["tool_choice"] is None
        assert resolved["max_tool_calls"] == 6
        assert resolved["temperature"] is None
        assert resolved["parallel_tool_calls"] is None

    def test_resolve_tool_settings_parses_tool_names_string_json(self):
        service = ConfigService()
        resolved = service.resolve_tool_settings({"tool_names": '["a", "b"]'})
        assert resolved["tool_names"] == ["a", "b"]

    def test_resolve_tool_settings_parses_tool_names_csv(self):
        service = ConfigService()
        resolved = service.resolve_tool_settings({"tool_names": "a, b,,c"})
        assert resolved["tool_names"] == ["a", "b", "c"]

    def test_resolve_tool_settings_coerces_max_tool_calls(self):
        service = ConfigService()
        assert service.resolve_tool_settings({"max_tool_calls": "3"})["max_tool_calls"] == 3
        assert service.resolve_tool_settings({"max_tool_calls": 4})["max_tool_calls"] == 4
        assert service.resolve_tool_settings({"max_tool_calls": 2.2})["max_tool_calls"] == 2
        assert service.resolve_tool_settings({"max_tool_calls": "bad"})["max_tool_calls"] == 6

    def test_normalize_tool_choice_rejects_boolean(self, caplog):
        service = ConfigService()
        caplog.set_level(logging.WARNING)
        resolved = service.resolve_tool_settings({"tool_choice": True})
        assert resolved["tool_choice"] is None
        assert any("Invalid tool_choice boolean" in r.message for r in caplog.records)

    def test_enable_tool_calls_forced_when_tool_names_present(self):
        service = ConfigService()
        resolved = service.resolve_tool_settings({"enable_tool_calls": False, "tool_names": ["a"]})
        assert resolved["enable_tool_calls"] is True

    def test_normalize_run_settings_preserves_retry_policy(self):
        service = ConfigService()
        resolved = service.normalize_run_settings({"retry_policy": {"max_retries": 2}})
        assert resolved["retry_policy"] == {"max_retries": 2}
