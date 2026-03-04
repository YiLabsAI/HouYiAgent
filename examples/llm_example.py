"""Example: Using real LLM with HouYi.

This example shows how to use OpenAI or Anthropic LLMs with HouYi.

Requirements:
    uv sync --extra model-adapters
    # or install directly:
    #   pip install "openai>=1.0.0" "anthropic>=0.7.0"
"""

import os

from houyi import Agent, tool
from houyi.llm.anthropic_adapter import AnthropicAdapter
from houyi.llm.openai_adapter import OpenAIAdapter


# Define a skill
@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    # In production, this would call a real search API
    return [f"Result 1 for '{query}'", f"Result 2 for '{query}'", "Additional information"]


# Example 1: Using OpenAI (requires OPENAI_API_KEY)
def example_openai():
    """Example using OpenAI GPT-4."""
    # Set API key (or use environment variable)
    # os.environ["OPENAI_API_KEY"] = "sk-..."

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI example")
        return

    print("=== Example 1: OpenAI GPT-4 ===")

    agent = Agent(
        role="Research Assistant",
        skills=[search],
        llm=OpenAIAdapter(model="gpt-4"),
    )

    # This will call real GPT-4
    result = agent.run("What are the latest developments in AI?")
    print(f"Result: {result}\n")


# Example 2: Using Anthropic Claude (requires ANTHROPIC_API_KEY)
def example_anthropic():
    """Example using Anthropic Claude."""
    # Set API key (or use environment variable)
    # os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic example")
        return

    print("=== Example 2: Anthropic Claude 3.5 ===")

    agent = Agent(
        role="Research Assistant",
        skills=[search],
        llm=AnthropicAdapter(model="claude-3-5-sonnet-20241022"),
    )

    # This will call real Claude
    result = agent.run("What are the latest developments in AI?")
    print(f"Result: {result}\n")


# Example 3: Direct LLM adapter usage
def example_direct_adapter():
    """Example using LLM adapter directly."""
    import asyncio

    from houyi.llm.base import LLMMessage, MessageRole
    from houyi.llm.openai_adapter import OpenAIAdapter

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set, skipping direct adapter example")
        return

    print("=== Example 3: Direct LLM Adapter ===")

    async def run():
        adapter = OpenAIAdapter(model="gpt-4")

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=MessageRole.USER, content="What is HouYi framework?"),
        ]

        response = await adapter.chat(messages)
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage}")

    asyncio.run(run())


# Example 4: Streaming response
def example_streaming():
    """Example using streaming LLM response."""
    import asyncio

    from houyi.llm.base import LLMMessage, MessageRole
    from houyi.llm.openai_adapter import OpenAIAdapter

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set, skipping streaming example")
        return

    print("=== Example 4: Streaming Response ===")

    async def run():
        adapter = OpenAIAdapter(model="gpt-4")

        messages = [LLMMessage(role=MessageRole.USER, content="Write a haiku about AI agents.")]

        print("Streaming: ", end="", flush=True)
        async for chunk in adapter.stream_chat(messages):
            print(chunk.content_delta, end="", flush=True)
        print("\n")

    asyncio.run(run())


if __name__ == "__main__":
    print("HouYi LLM Integration Examples\n")
    print("Note: Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run examples\n")

    print("Studio chat tool-calling request example:")
    print(
        """
POST /api/chat/conversations/<conversation_id>/messages
{
  "content": "Find TODO comments in this repo and summarize them",
  "enable_tool_calls": true,
  "tool_call_strategy": "balanced",
  "enable_skills": ["houyi_find_files", "houyi_grep", "houyi_read_file"],
  "enable_web_search": false,
  "max_tool_iterations": 6
}
""".strip()
    )
    print()

    # Run examples
    example_openai()
    example_anthropic()
    example_direct_adapter()
    example_streaming()

    print("\n✅ Examples completed!")
