"""Example: Observability with HouYi.

This example shows how to use tracing and different exporters.
"""

from houyi import Agent, Task, Team, tool


# Define skills
@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for '{query}'", "Additional info"]


@tool
def analyze(data: list[str]) -> str:
    """Analyze data."""
    return f"Analysis of {len(data)} items"


# Example 1: Default observability (console output)
def example_default():
    """Example with default console tracing."""
    print("=== Example 1: Default Console Tracing ===\n")

    agent = Agent(role="Researcher", skills=[search])

    result = agent.run("What is HouYi?")
    print(f"Result: {result}\n")


# Example 2: Verbose console output
def example_verbose():
    """Example with verbose console tracing."""
    print("=== Example 2: Verbose Console Tracing ===\n")

    agent = Agent(
        role="Researcher",
        skills=[search],
        observability={"enabled": True, "exporters": [{"type": "console", "verbose": True}]},
    )

    result = agent.run("What is HouYi?")
    print(f"Result: {result}\n")


# Example 3: Multiple exporters
def example_multiple_exporters():
    """Example with multiple exporters."""
    print("=== Example 3: Multiple Exporters ===\n")

    agent = Agent(
        role="Researcher",
        skills=[search],
        observability={
            "enabled": True,
            "exporters": [{"type": "console"}, {"type": "json", "filepath": "traces.json"}],
        },
    )

    result = agent.run("What is HouYi?")
    print(f"Result: {result}")
    print("✅ Traces also exported to traces.json\n")


# Example 4: Jaeger exporter (placeholder)
def example_jaeger():
    """Example with Jaeger exporter."""
    print("=== Example 4: Jaeger Exporter (Placeholder) ===\n")

    agent = Agent(
        role="Researcher",
        skills=[search],
        observability={
            "enabled": True,
            "exporters": [
                {"type": "console"},
                {"type": "jaeger", "endpoint": "http://localhost:4318"},
            ],
        },
    )

    result = agent.run("What is HouYi?")
    print(f"Result: {result}\n")


# Example 5: Disabled observability
def example_disabled():
    """Example with observability disabled."""
    print("=== Example 5: Disabled Observability ===\n")

    agent = Agent(role="Researcher", skills=[search], observability={"enabled": False})

    result = agent.run("What is HouYi?")
    print(f"Result: {result}")
    print("(No trace output)\n")


# Example 6: Multi-agent team with tracing
def example_team():
    """Example with multi-agent team tracing."""
    print("=== Example 6: Multi-Agent Team Tracing ===\n")

    researcher = Agent(
        role="Researcher",
        skills=[search],
        observability={"enabled": True, "exporters": [{"type": "console", "verbose": True}]},
    )

    analyst = Agent(role="Analyst", skills=[analyze])

    team = Team(
        agents=[researcher, analyst],
        tasks=[
            Task("Research HouYi", agent=researcher),
            Task("Analyze findings", agent=analyst, context=[0]),
        ],
    )

    result = team.run()
    print(f"\nResult: {result}\n")


if __name__ == "__main__":
    print("HouYi Observability Examples\n")
    print("=" * 60 + "\n")

    # Run examples
    example_default()
    example_verbose()
    example_multiple_exporters()
    example_jaeger()
    example_disabled()
    example_team()

    print("=" * 60)
    print("✅ All examples completed!")
