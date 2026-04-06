from houyi.application.runtime.registry import AgentRegistry, AgentTypeConfig


def _cfg(agent_type: str = "t", name: str = "T") -> AgentTypeConfig:
    return AgentTypeConfig(agent_type=agent_type, name=name)


class TestAgentTypeConfig:
    def test_config_defaults(self) -> None:
        cfg = _cfg()
        assert cfg.description == ""
        assert cfg.icon == ""
        assert cfg.status == "active"
        assert cfg.default_spec is None
        assert cfg.supported_tools == []
        assert cfg.metadata == {}


class TestAgentRegistry:
    def test_register_and_get(self) -> None:
        reg = AgentRegistry()
        cfg = _cfg("deep_researcher", "Deep Researcher")
        reg.register("deep_researcher", cfg)
        assert reg.get("deep_researcher") is cfg

    def test_get_missing(self) -> None:
        assert AgentRegistry().get("nope") is None

    def test_list_available(self) -> None:
        reg = AgentRegistry()
        cfg = _cfg("a", "A")
        reg.register("a", cfg)
        assert cfg in reg.list_available()

    def test_unregister(self) -> None:
        reg = AgentRegistry()
        reg.register("x", _cfg("x", "X"))
        reg.unregister("x")
        assert reg.get("x") is None

    def test_unregister_missing(self) -> None:
        AgentRegistry().unregister("nope")  # should not raise

    def test_len(self) -> None:
        reg = AgentRegistry()
        assert len(reg) == 0
        reg.register("a", _cfg("a", "A"))
        assert len(reg) == 1

    def test_contains(self) -> None:
        reg = AgentRegistry()
        reg.register("a", _cfg("a", "A"))
        assert "a" in reg
        assert "b" not in reg

    def test_overwrite(self) -> None:
        reg = AgentRegistry()
        reg.register("a", _cfg("a", "Old"))
        reg.register("a", _cfg("a", "New"))
        assert reg.get("a").name == "New"  # type: ignore[union-attr]
