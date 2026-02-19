# Integration Tests

## Console Tool Injection Fixture

Console integration tests that rely on tool calls load the fixture at:

```
/tests/integration/fixtures/console_tools.py
```

This module registers integration-test tools (get_date/get_location/get_weather, etc.) in
`DEFAULT_SKILL_REGISTRY` and can also be loaded by the console server startup
when `HOUYI_DISABLE_E2E_TOOLS` is not set.

### Running the tool-call scenario

```bash
uv run pytest tests/integration/test_llm_tool_scenario.py -k tool_scenario -q
```

If your environment does not allow external calls, set:

```bash
export HOUYI_DISABLE_LIVE_WEATHER=1
```

The tests will still exercise tool bindings with mocked weather output.
