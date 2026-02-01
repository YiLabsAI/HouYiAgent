"""Unit tests for GoogleVertexGeminiAdapter conversions."""

from __future__ import annotations

from houyi.llm.vertex_gemini_adapter import GoogleVertexGeminiAdapter


def _build_adapter() -> GoogleVertexGeminiAdapter:
    # Bypass __init__ to avoid optional runtime dependency.
    return GoogleVertexGeminiAdapter.__new__(GoogleVertexGeminiAdapter)


def test_vertex_gemini_tool_schema_conversion() -> None:
    """Should convert OpenAI tool schema to Vertex function declarations."""

    adapter = _build_adapter()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    converted = adapter._convert_tools_to_vertex(tools)
    assert converted
    declarations = converted[0]["function_declarations"]
    assert declarations[0]["name"] == "weather"


def test_vertex_gemini_tool_choice_conversion() -> None:
    """Should map tool_choice to Vertex function calling config."""

    adapter = _build_adapter()
    assert adapter._convert_tool_choice("required")["function_calling_config"]["mode"] == "ANY"
    assert adapter._convert_tool_choice("none")["function_calling_config"]["mode"] == "NONE"
    assert adapter._convert_tool_choice("auto")["function_calling_config"]["mode"] == "AUTO"
